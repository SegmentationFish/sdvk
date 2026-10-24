// SDXL (NoobAI-XL vPred-1.0, Q8_0 GGUF) text-to-image on the ofc Vulkan engine.
// fp32 activations; Q8_0 weights stay as int8 payload + fp32 scale in VRAM and are
// dequantized inside the gemm shaders (gemm_nt_q8 / gemm_nn_q8).
//
// Pipeline: CLIP-L (768, 12 layers) + OpenCLIP-bigG (1280, 32 layers, pooled) ->
// UNet2DConditionModel (SDXL: added_time_embed 2816, cross-attn 2048, heads 5/10/20) ->
// EulerDiscreteScheduler (epsilon) -> VAE decoder (fp16 weights, dequantized on load).
//
// Weight phases share one param arena (reset between phases) to stay inside 7.75 GiB VRAM:
//   A) text encoders   B) UNet   C) VAE
#include "Modules/GPU/gpu_ops.hpp"
#include "Modules/Diffusion/Safetensors.hpp"
#include "Modules/Model/Gguf.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ofc;
using namespace ofc::gpu;

// ================================================================ globals

static GgufFile gGG;                                            // UNet weights (Q8_0 / F16 / F32)
static std::vector<std::pair<std::string, Safetensors*>> gSTs;  // (prefix, file) for dim lookup

static std::unordered_map<std::string, BufRef> gW;   // fp32 weights
static std::unordered_map<std::string, BufRef> gWQ;  // Q8_0 int8 payloads
static std::unordered_map<std::string, BufRef> gWS;  // Q8_0 fp32 scales

static uint32_t ceilDiv(uint64_t a, uint32_t b) { return (uint32_t)((a + b - 1) / b); }
static BufRef allocN(uint64_t nFloats) { return G.eng.stepArena().alloc(nFloats * 4); }
static uint64_t arenaMark() { return G.eng.stepArena().used(); }
static void arenaRewind(uint64_t bytes) { G.eng.stepArena().rewind(bytes); }

static void die(const std::string& m) {
    std::printf("FAIL: %s\n", m.c_str());
    std::exit(1);
}

// ================================================================ weight access

static const BufRef& W(const std::string& name) {
    auto it = gW.find(name);
    if (it == gW.end()) die("missing fp32 weight " + name);
    return it->second;
}
static bool hasW(const std::string& name) { return gW.count(name) != 0; }

static const BufRef& WQ(const std::string& name) {
    auto it = gWQ.find(name);
    if (it == gWQ.end()) die("missing Q8 payload " + name);
    return it->second;
}
static const BufRef& WS(const std::string& name) {
    auto it = gWS.find(name);
    if (it == gWS.end()) die("missing Q8 scale " + name);
    return it->second;
}
static bool isQ8(const std::string& name) { return gWQ.count(name) != 0; }
// a weight may live in the fp32 map or the Q8_0 payload maps
static bool hasAnyW(const std::string& name) { return gW.count(name) || gWQ.count(name); }

// GGUF logical (rows, cols): comfy.gguf.orig_shape.* wins, else ne[1]xne[0] (1D -> ne0 x 1, like sdims)
static void ggdims(const std::string& name, uint32_t& rows, uint32_t& cols) {
    const GgufTensor* t = gGG.find(name);
    if (!t) die("missing gguf tensor " + name);
    const GgufMeta* m = gGG.meta("comfy.gguf.orig_shape." + name);
    if (m && m->isArray() && m->u64s.size() >= 2) {
        rows = (uint32_t)m->u64s[0];
        cols = 1;
        for (size_t i = 1; i < m->u64s.size(); i++) cols *= (uint32_t)m->u64s[i];
        return;
    }
    if (t->dims.size() <= 1) {
        rows = (uint32_t)(t->dims.empty() ? 1 : t->dims[0]);
        cols = 1;
        return;
    }
    rows = (uint32_t)t->dims[1];
    cols = (uint32_t)t->dims[0];
}

static const STTensor* findST(const std::string& name) {
    for (auto& e : gSTs) {
        if (!e.first.empty() && name.rfind(e.first, 0) == 0) {
            // prefixed lookup: only the file that owns this prefix
            return e.second->find(name.substr(e.first.size()));
        }
    }
    for (auto& e : gSTs) {
        const STTensor* t = e.second->find(name);
        if (t) return t;
    }
    return nullptr;
}
static void sdims(const std::string& name, uint32_t& rows, uint32_t& cols) {
    const STTensor* t = findST(name);
    if (!t) die("missing safetensors tensor " + name);
    rows = (uint32_t)t->shape[0];
    cols = 1;
    for (size_t i = 1; i < t->shape.size(); i++) cols *= (uint32_t)t->shape[i];
}
static bool hasST(const std::string& name) { return findST(name) != nullptr; }

// dims of a weight that may live in the GGUF (Q8) or in a safetensors file (fp32)
// Route by which container holds the tensor: the GGUF also stores some UNet tensors as F16/F32.
static void linearDims(const std::string& name, uint32_t& rows, uint32_t& cols) {
    if (gGG.find(name)) ggdims(name, rows, cols);
    else sdims(name, rows, cols);
}
static uint32_t wdimT(const std::string& name) {
    uint32_t r, c;
    linearDims(name, r, c);
    return r;
}

static void clearWeights() {
    gW.clear();
    gWQ.clear();
    gWS.clear();
    gSTs.clear();
}

// ================================================================ weight loading

static void loadGgufUnet() {
    uint64_t pending = 0, q8n = 0, f32n = 0;
    for (auto& t : gGG.tensors()) {
        if (t.type == GgmlType::Q8_0) {
            std::vector<int8_t> q;
            std::vector<float> s;
            std::string err;
            if (!gGG.readQ8(t.name, q, s, &err)) die(err);
            BufRef bq = G.paramArena.alloc((q.size() + 255) & ~255ull);
            BufRef bs = G.paramArena.alloc((s.size() * 4 + 255) & ~255ull);
            G.eng.stageUpload(bq, q.data(), q.size());
            G.eng.stageUpload(bs, s.data(), s.size() * 4);
            gWQ[t.name] = bq;
            gWS[t.name] = bs;
            q8n++;
            pending += q.size() + s.size() * 4;
        } else {
            std::string err;
            auto v = gGG.readFloat(t.name, &err);
            if (v.empty()) die(err + " (" + t.name + ")");
            BufRef b = G.paramArena.alloc((v.size() * 4 + 255) & ~255ull);
            G.eng.stageUpload(b, v.data(), v.size() * 4);
            gW[t.name] = b;
            f32n++;
            pending += v.size() * 4;
        }
        if (pending >= (96ull << 20)) {
            G.eng.flushUploads();
            pending = 0;
        }
    }
    G.eng.flushUploads();
    std::printf("unet weights: %llu Q8_0 + %llu fp (param arena %.0f MB)\n", (unsigned long long)q8n,
                (unsigned long long)f32n, G.paramArena.used() / 1048576.0);
}

static void loadStWeights(Safetensors& st, const std::string& pfx) {
    uint64_t pending = 0;
    for (auto& t : st.tensors()) {
        auto v = st.readF32(t.name);
        if (v.empty()) die("read failed: " + t.name);
        BufRef b = G.paramArena.alloc((v.size() * 4 + 255) & ~255ull);
        G.eng.stageUpload(b, v.data(), v.size() * 4);
        gW[pfx + t.name] = b;
        pending += v.size() * 4;
        if (pending >= (96ull << 20)) {
            G.eng.flushUploads();
            pending = 0;
        }
    }
    G.eng.flushUploads();
    gSTs.push_back({pfx, &st});
    std::printf("weights loaded (%s): %.0f MB\n", pfx.c_str(), G.paramArena.used() / 1048576.0);
}

// ================================================================ dispatch helpers

struct Pg {
    uint32_t M, N, K, mode, sA, sB, sC, batch, lda, ldb, ldc, pad;
};
struct Pnd {
    uint32_t n, d0, d1, d2, d3, os0, os1, os2, os3, ss0, ss1, ss2, ss3, mode;
    float scale;
};
struct P2 {
    uint32_t a, b;
};

static void gemmNt(const BufRef& A, const BufRef& B, const BufRef& C, uint32_t M, uint32_t N, uint32_t K) {
    Pg pc{M, N, K, 0, 0, 0, 0, 1, 0, 0, 0, 0};
    G.eng.dispatch("gemm_nt", {A, B, C}, &pc, ceilDiv(N, 64), ceilDiv(M, 64), 1);
}

static void gemmNtQ8(const BufRef& A, const BufRef& Bq, const BufRef& C, const BufRef& Bs, uint32_t M, uint32_t N,
                     uint32_t K) {
    Pg pc{M, N, K, 0, 0, 0, 0, 1, 0, 0, 0, 0};
    G.eng.dispatch("gemm_nt_q8", {A, Bq, C, Bs}, &pc, ceilDiv(N, 64), ceilDiv(M, 64), 1);
}

static void gemmNn(const BufRef& A, const BufRef& B, const BufRef& C, uint32_t M, uint32_t N, uint32_t K,
                   uint32_t ldc) {
    Pg pc{M, N, K, 0, 0, 0, 0, 1, 0, 0, ldc, 0};
    G.eng.dispatch("gemm_nn", {A, B, C}, &pc, ceilDiv(N, 64), ceilDiv(M, 64), 1);
}

static void gemmNnQ8(const BufRef& Aq, const BufRef& As, const BufRef& B, const BufRef& C, uint32_t M, uint32_t N,
                     uint32_t K, uint32_t ldc) {
    Pg pc{M, N, K, 0, 0, 0, 0, 1, 0, 0, ldc, 0};
    G.eng.dispatch("gemm_nn_q8", {Aq, B, C, As}, &pc, ceilDiv(N, 64), ceilDiv(M, 64), 1);
}

static BufRef copyNd(const BufRef& src, uint32_t n, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t os0, uint32_t os1,
                     uint32_t os2, uint32_t ss0, uint32_t ss1, uint32_t ss2, float scale = 1.0f) {
    BufRef dst = allocN(n);
    Pnd pc{n, d0, d1, d2, 1, os0, os1, os2, 0, ss0, ss1, ss2, 0, 0, scale};
    G.eng.dispatch("copy_nd", {dst, src}, &pc, ceilDiv(n, 256), 1, 1);
    return dst;
}

static void copyInto(const BufRef& dst, uint64_t dstOff, const BufRef& src, uint64_t bytes) {
    BufRef d{dst.buf, dst.offset + dstOff, bytes};
    BufRef s{src.buf, src.offset, bytes};
    G.eng.copyBuf(d, s);
}

// device-side trace: snapshot a buffer into the scratch arena (copy recorded in order),
// downloaded after waitStep by flushTrace()
static bool gTrace = false;
static const char* gDbSel = "";  // dump the internals of this one stage only
struct TrEnt {
    std::string name;
    BufRef b;
    uint64_t floats;
};
static std::vector<TrEnt> gTr;
static void tr(const std::string& name, const BufRef& b, uint64_t floats) {
    if (!gTrace) return;
    BufRef c = G.scratchArena.alloc((floats * 4 + 255) & ~255ull);
    copyInto(c, 0, b, floats * 4);
    gTr.push_back({name, c, floats});
}
static void flushTrace() {
    for (auto& e : gTr) {
        std::vector<float> v(e.floats);
        G.eng.downloadNow(e.b, v.data(), v.size() * 4);
        FILE* f = std::fopen(("dbg_" + e.name + ".f32").c_str(), "wb");
        std::fwrite(v.data(), 4, v.size(), f);
        std::fclose(f);
    }
    gTr.clear();
}

static BufRef add2(const BufRef& a, const BufRef& b, uint64_t n) {
    BufRef c = allocN(n);
    P2 pc{(uint32_t)n, 0};
    G.eng.dispatch("add2", {c, a, b}, &pc, ceilDiv(n, 256), 1, 1);
    return c;
}

static BufRef mul2(const BufRef& a, const BufRef& b, uint64_t n) {
    BufRef c = allocN(n);
    struct {
        uint32_t n, d0, d1, d2, d3, as0, as1, as2, as3, bs0, bs1, bs2, bs3, mode;
    } pc;
    std::memset(&pc, 0, sizeof pc);
    pc.n = (uint32_t)n;
    pc.d0 = (uint32_t)n;
    pc.d1 = 1;
    pc.d2 = 1;
    pc.d3 = 1;
    pc.as0 = 1;
    pc.bs0 = 1;
    pc.mode = 1;
    G.eng.dispatch("ew2", {c, a, b}, &pc, ceilDiv(n, 256), 1, 1);
    return c;
}

static BufRef unary(const BufRef& x, uint64_t n, uint32_t mode) {
    BufRef y = allocN(n);
    struct {
        uint32_t n, mode;
    } pc{(uint32_t)n, mode};
    G.eng.dispatch("unary_fwd", {x, y}, &pc, ceilDiv(n, 256), 1, 1);
    return y;
}
static BufRef silu(const BufRef& x, uint64_t n) { return unary(x, n, 2); }
static BufRef geluErf(const BufRef& x, uint64_t n) { return unary(x, n, 4); }

static BufRef softmax(const BufRef& x, uint32_t rows, uint32_t cols) {
    BufRef y = allocN((uint64_t)rows * cols);
    P2 pc{rows, cols};
    G.eng.dispatch("softmax_fwd", {x, y}, &pc, rows, 1, 1);
    return y;
}

static void biasAddLast(const BufRef& y, const BufRef& b, uint32_t rows, uint32_t d) {
    struct {
        uint32_t n, d0, d1, d2, d3, as0, as1, as2, as3, bs0, bs1, bs2, bs3, mode;
    } pc;
    std::memset(&pc, 0, sizeof pc);
    pc.n = rows * d;
    pc.d0 = d;
    pc.d1 = rows;
    pc.d2 = 1;
    pc.d3 = 1;
    pc.as0 = 1;
    pc.as1 = d;
    pc.bs0 = 1;
    pc.mode = 0;
    G.eng.dispatch("ew2", {y, y, b}, &pc, ceilDiv((uint64_t)rows * d, 256), 1, 1);
}

static void biasAddCh(const BufRef& y, const BufRef& b, uint32_t C, uint32_t P) {
    struct {
        uint32_t C, P;
    } pb{C, P};
    G.eng.dispatch("bias_add", {y, b}, &pb, ceilDiv((uint64_t)C * P, 256), 1, 1);
}

static void scaleInPlace(const BufRef& x, uint64_t n, float s) {
    Pnd pc{(uint32_t)n, (uint32_t)n, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, s};
    G.eng.dispatch("copy_nd", {x, x}, &pc, ceilDiv(n, 256), 1, 1);
}

static BufRef transpose2(const BufRef& x, uint32_t outer, uint32_t inner) {
    return copyNd(x, outer * inner, inner, outer, 1, outer, 1, 0, 1, inner, 0);
}

// linear over a weight set that may be Q8 (gguf) or fp32 (safetensors)
static BufRef linear(const BufRef& x, const std::string& p, uint32_t rows) {
    uint32_t nout, nin;
    linearDims(p + ".weight", nout, nin);
    BufRef y = allocN((uint64_t)rows * nout);
    if (isQ8(p + ".weight")) {
        gemmNtQ8(x, WQ(p + ".weight"), y, WS(p + ".weight"), rows, nout, nin);
    } else {
        gemmNt(x, W(p + ".weight"), y, rows, nout, nin);
    }
    if (hasW(p + ".bias")) biasAddLast(y, W(p + ".bias"), rows, nout);
    return y;
}

static BufRef layernorm(const BufRef& x, const std::string& p, uint32_t rows, uint32_t d) {
    BufRef y = allocN((uint64_t)rows * d);
    struct {
        uint32_t rows, d;
        float eps;
        uint32_t affine;
    } pc{rows, d, 1e-5f, 1};
    G.eng.dispatch("layernorm_fwd", {x, W(p + ".weight"), W(p + ".bias"), y}, &pc, rows, 1, 1);
    return y;
}

static BufRef gn(const BufRef& x, const std::string& p, uint32_t C, uint32_t P, uint32_t groups = 32,
                 float eps = 1e-5f) {
    BufRef y = allocN((uint64_t)C * P);
    struct {
        uint32_t C, P, G;
        float eps;
    } pc{C, P, groups, eps};
    G.eng.dispatch("groupnorm", {x, W(p + ".weight"), W(p + ".bias"), y}, &pc, groups, 1, 1);
    return y;
}

// ================================================================ convolutions

static BufRef gCol;
static const uint64_t kColFloats = 12000000;  // 48 MB reusable im2col buffer

// out[cout,P] = wt[cout, cin*9] * col[cin*9, P]; row-chunked to bound the col buffer
static BufRef conv3Raw(const BufRef& in, const std::string& wt, uint32_t cin, uint32_t cout, uint32_t H, uint32_t Wd) {
    uint32_t P = H * Wd, K = cin * 9;
    BufRef out = allocN((uint64_t)cout * P);
    uint32_t rows = (uint32_t)std::max<uint64_t>(1, std::min<uint64_t>(H, kColFloats / (uint64_t)K / Wd));
    for (uint32_t r0 = 0; r0 < H; r0 += rows) {
        uint32_t rr = std::min(rows, H - r0), np = rr * Wd;
        struct {
            uint32_t C, H, W, row0, rows;
        } pc{cin, H, Wd, r0, rr};
        BufRef col{gCol.buf, gCol.offset, (uint64_t)K * np * 4};
        G.eng.dispatch("im2col", {in, col}, &pc, ceilDiv((uint64_t)K * np, 256), 1, 1);
        BufRef cs{out.buf, out.offset + (uint64_t)r0 * Wd * 4, out.size - (uint64_t)r0 * Wd * 4};
        if (isQ8(wt)) gemmNnQ8(WQ(wt), WS(wt), col, cs, cout, np, K, P);
        else gemmNn(W(wt), col, cs, cout, np, K, P);
    }
    std::string biasName = wt;  // callers pass "<prefix>.weight"; the bias shares that prefix
    biasName.erase(biasName.size() - 7);
    biasName += ".bias";
    biasAddCh(out, W(biasName), cout, P);
    return out;
}

static BufRef conv3s2(const BufRef& in, const std::string& p, uint32_t H, uint32_t Wd) {
    uint32_t cout, kCols;  // conv weight dims are (cout, cin*k)
    linearDims(p + ".weight", cout, kCols);
    uint32_t cin = kCols / 9;
    uint32_t OH = H / 2, OW = Wd / 2, K = cin * 9, P = OH * OW;
    BufRef out = allocN((uint64_t)cout * P);
    struct {
        uint32_t C, H, W, OH, OW;
    } pc{cin, H, Wd, OH, OW};
    BufRef col{gCol.buf, gCol.offset, (uint64_t)K * P * 4};
    G.eng.dispatch("im2col_s2", {in, col}, &pc, ceilDiv((uint64_t)K * P, 256), 1, 1);
    if (isQ8(p + ".weight")) gemmNnQ8(WQ(p + ".weight"), WS(p + ".weight"), col, out, cout, P, K, P);
    else gemmNn(W(p + ".weight"), col, out, cout, P, K, P);
    biasAddCh(out, W(p + ".bias"), cout, P);
    return out;
}

static BufRef conv1(const BufRef& in, const std::string& p, uint32_t cin, uint32_t cout, uint32_t P) {
    BufRef out = allocN((uint64_t)cout * P);
    if (isQ8(p + ".weight")) {
        gemmNnQ8(WQ(p + ".weight"), WS(p + ".weight"), in, out, cout, P, cin, P);
    } else {
        gemmNn(W(p + ".weight"), in, out, cout, P, cin, P);
    }
    biasAddCh(out, W(p + ".bias"), cout, P);
    return out;
}

static BufRef up2(const BufRef& x, uint32_t C, uint32_t H, uint32_t Wd) {
    BufRef y = allocN((uint64_t)C * (H * 2) * (Wd * 2));
    struct {
        uint32_t C, H, W;
    } pc{C, H, Wd};
    G.eng.dispatch("upsample2x", {x, y}, &pc, ceilDiv((uint64_t)C * H * 2 * Wd * 2, 256), 1, 1);
    return y;
}

// ================================================================ attention

static BufRef gS;
static uint64_t gSCap = 0;

static void ensureAttnScratch(uint64_t floats) {
    if (gSCap >= floats) return;
    gS = G.scratchArena.alloc(floats * 4);
    gSCap = floats;
}

static BufRef headsView(const BufRef& x, uint32_t T, uint32_t C, uint32_t heads, uint32_t hOff, uint32_t hCnt,
                        float scale) {
    uint32_t hd = C / heads;
    BufRef src{x.buf, x.offset + (uint64_t)hOff * hd * 4, x.size - (uint64_t)hOff * hd * 4};
    return copyNd(src, hCnt * T * hd, hd, T, hCnt, 1, hd, T * hd, 1, C, hd, scale);
}

static void headsMergeInto(const BufRef& dst, const BufRef& x, uint32_t T, uint32_t C, uint32_t heads, uint32_t hOff,
                           uint32_t hCnt) {
    uint32_t hd = C / heads;
    BufRef d{dst.buf, dst.offset + (uint64_t)hOff * hd * 4, dst.size - (uint64_t)hOff * hd * 4};
    Pnd pc{hCnt * T * hd, hd, T, hCnt, 1, 1, C, hd, 0, 1, hd, T * hd, 0, 0, 1.0f};
    G.eng.dispatch("copy_nd", {d, x}, &pc, ceilDiv((uint64_t)hCnt * T * hd, 256), 1, 1);
}

static BufRef attnCore(const BufRef& q, const BufRef& k, const BufRef& v, uint32_t Tq, uint32_t Tk, uint32_t C,
                       uint32_t heads, bool causal = false) {
    uint32_t hd = C / heads;
    uint64_t perHead = (uint64_t)Tq * Tk;
    uint32_t chunk = (uint32_t)std::max<uint64_t>(1, std::min<uint64_t>(16, 16000000ull / perHead));
    ensureAttnScratch((uint64_t)chunk * perHead);
    BufRef out = allocN((uint64_t)Tq * C);
    float scale = 1.0f / std::sqrt((float)hd);
    for (uint32_t h0 = 0; h0 < heads; h0 += chunk) {
        uint32_t hc = std::min(chunk, heads - h0);
        BufRef qh = headsView(q, Tq, C, heads, h0, hc, scale);
        BufRef kh = headsView(k, Tk, C, heads, h0, hc, 1.0f);
        BufRef vh = headsView(v, Tk, C, heads, h0, hc, 1.0f);
        BufRef S{gS.buf, gS.offset, (uint64_t)hc * Tq * Tk * 4};
        Pg pc{Tq, Tk, hd, 0, Tq * hd, Tk * hd, Tq * Tk, hc, 0, 0, 0, 0};
        G.eng.dispatch("gemm_nt", {qh, kh, S}, &pc, ceilDiv(Tk, 64), ceilDiv(Tq, 64), hc);
        if (causal) {
            struct {
                uint32_t batch, rows, cols;
            } cm{hc, Tq, Tk};
            G.eng.dispatch("causal_mask", {S}, &cm, ceilDiv((uint64_t)hc * Tq * Tk, 256), 1, 1);
        }
        P2 ps{hc * Tq, Tk};
        G.eng.dispatch("softmax_fwd", {S, S}, &ps, hc * Tq, 1, 1);
        BufRef O = allocN((uint64_t)hc * Tq * hd);
        Pg po{Tq, hd, Tk, 0, Tq * Tk, Tk * hd, Tq * hd, hc, 0, 0, 0, 0};
        G.eng.dispatch("gemm_nn", {S, vh, O}, &po, ceilDiv(hd, 64), ceilDiv(Tq, 64), hc);
        headsMergeInto(out, O, Tq, C, heads, h0, hc);
    }
    return out;
}

static BufRef crossAttention(const BufRef& xq, const BufRef& xkv, const std::string& p, uint32_t Tq, uint32_t Tk,
                             uint32_t C, uint32_t heads) {
    BufRef q = linear(xq, p + ".to_q", Tq);
    BufRef k = linear(xkv, p + ".to_k", Tk);
    BufRef v = linear(xkv, p + ".to_v", Tk);
    BufRef o = attnCore(q, k, v, Tq, Tk, C, heads);
    return linear(o, p + ".to_out.0", Tq);
}

// ================================================================ CLIP BPE tokenizer

struct TokSet {
    std::map<std::string, int> vocab;
    std::map<std::pair<std::string, std::string>, int> merges;
    std::unordered_map<uint32_t, uint8_t> charByte;
};

static void utf8Append(std::string& s, uint32_t cp) {
    if (cp < 0x80) s.push_back((char)cp);
    else if (cp < 0x800) {
        s.push_back((char)(0xC0 | (cp >> 6)));
        s.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        s.push_back((char)(0xE0 | (cp >> 12)));
        s.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        s.push_back((char)(0xF0 | (cp >> 18)));
        s.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        s.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back((char)(0x80 | (cp & 0x3F)));
    }
}

static std::vector<uint32_t> utf8Decode(const std::string& s, size_t& i) {
    std::vector<uint32_t> out;
    while (i < s.size()) {
        unsigned char c = (unsigned char)s[i];
        uint32_t cp;
        size_t n;
        if (c < 0x80) {
            cp = c;
            n = 1;
        } else if ((c >> 5) == 0x6) {
            cp = c & 0x1F;
            n = 2;
        } else if ((c >> 4) == 0xE) {
            cp = c & 0x0F;
            n = 3;
        } else {
            cp = c & 0x07;
            n = 4;
        }
        if (i + n > s.size()) break;
        for (size_t k = 1; k < n; k++) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
        out.push_back(cp);
        i += n;
    }
    return out;
}

static void buildByteMap(TokSet& ts) {
    std::vector<int> bs, cs;
    for (int b = '!'; b <= '~'; b++) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
    cs = bs;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }
    for (size_t i = 0; i < bs.size(); i++) ts.charByte[(uint32_t)cs[i]] = (uint8_t)bs[i];
}

static std::string decodeToken(const TokSet& ts, const std::string& enc) {
    std::string raw;
    size_t i = 0;
    auto cps = utf8Decode(enc, i);
    for (uint32_t cp : cps) {
        auto it = ts.charByte.find(cp);
        if (it == ts.charByte.end()) continue;
        raw.push_back((char)it->second);
    }
    return raw;
}

static void loadVocab(TokSet& ts, const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::printf("FAIL: cannot open %s\n", path.c_str());
        std::exit(1);
    }
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t i = 0;
    auto ws = [&]() {
        while (i < s.size() && std::isspace((unsigned char)s[i])) i++;
    };
    auto readStr = [&]() {
        std::string out;
        if (i >= s.size() || s[i] != '"') {
            std::printf("vocab: bad json\n");
            std::exit(1);
        }
        i++;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                char c = s[i + 1];
                if (c == 'n') {
                    out.push_back('\n');
                    i += 2;
                } else if (c == 't') {
                    out.push_back('\t');
                    i += 2;
                } else if (c == 'r') {
                    out.push_back('\r');
                    i += 2;
                } else if (c == 'u' && i + 5 < s.size()) {
                    uint32_t cp = (uint32_t)std::stoul(s.substr(i + 2, 4), nullptr, 16);
                    i += 6;
                    if (cp >= 0xD800 && cp <= 0xDBFF && i + 5 < s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                        uint32_t lo = (uint32_t)std::stoul(s.substr(i + 2, 4), nullptr, 16);
                        i += 6;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    utf8Append(out, cp);
                } else {
                    out.push_back(c);
                    i += 2;
                }
            } else out.push_back(s[i++]);
        }
        i++;
        return out;
    };
    ws();
    if (i >= s.size() || s[i] != '{') {
        std::printf("vocab: expected object\n");
        std::exit(1);
    }
    i++;
    while (i < s.size()) {
        ws();
        if (s[i] == '}') break;
        if (s[i] == ',') {
            i++;
            continue;
        }
        std::string key = readStr();
        ws();
        if (i < s.size() && s[i] == ':') i++;
        ws();
        char* endp = nullptr;
        long id = std::strtol(s.c_str() + i, &endp, 10);
        i = (size_t)(endp - s.c_str());
        ts.vocab[decodeToken(ts, key)] = (int)id;
    }
}

static void loadMerges(TokSet& ts, const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::printf("FAIL: cannot open %s\n", path.c_str());
        std::exit(1);
    }
    std::string line;
    int rank = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        std::string a = decodeToken(ts, line.substr(0, sp));
        std::string b = decodeToken(ts, line.substr(sp + 1));
        ts.merges[{a, b}] = rank++;
    }
}

static std::vector<int> bpeWord(const TokSet& ts, const std::string& word) {
    if (word.empty()) return {};
    std::vector<std::string> syms;
    for (unsigned char c : word) syms.push_back(std::string(1, (char)c));
    syms.back() += "</w>";  // CLIP bpe(): word = tuple(token[:-1]) + (token[-1] + "</w>",)
    while (syms.size() > 1) {
        int bestRank = INT32_MAX;
        size_t bestAt = 0;
        for (size_t i = 0; i + 1 < syms.size(); i++) {
            auto it = ts.merges.find({syms[i], syms[i + 1]});
            if (it != ts.merges.end() && it->second < bestRank) {
                bestRank = it->second;
                bestAt = i;
            }
        }
        if (bestRank == INT32_MAX) break;
        syms[bestAt] += syms[bestAt + 1];
        syms.erase(syms.begin() + bestAt + 1);
    }
    std::vector<int> ids;
    for (auto& s : syms) {
        auto it = ts.vocab.find(s);
        if (it != ts.vocab.end()) ids.push_back(it->second);
    }
    return ids;
}

static uint32_t utf8One(const std::string& s, size_t& i) {
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) {
        i++;
        return c;
    }
    uint32_t cp;
    size_t n;
    if ((c >> 5) == 0x6) {
        cp = c & 0x1F;
        n = 2;
    } else if ((c >> 4) == 0xE) {
        cp = c & 0x0F;
        n = 3;
    } else {
        cp = c & 0x07;
        n = 4;
    }
    if (i + n > s.size()) {
        i++;
        return 0;
    }
    for (size_t k = 1; k < n; k++) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
    i += n;
    return cp;
}

static std::vector<std::string> preTokenize(const std::string& text) {
    // CLIP pattern: 's|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+  (case-insensitive)
    std::string low;
    for (char c : text) low.push_back((char)std::tolower((unsigned char)c));
    std::vector<std::string> words;
    static const char* sufs[] = {"s", "t", "re", "ve", "m", "ll", "d"};
    size_t i = 0;
    while (i < low.size()) {
        unsigned char c = (unsigned char)low[i];
        if (std::isspace(c)) {
            i++;
            continue;
        }
        if (c == '\'' && i + 1 < low.size()) {
            bool matched = false;
            for (const char* suf : sufs) {
                size_t sl = std::strlen(suf);
                if (low.compare(i + 1, sl, suf) == 0) {
                    words.push_back("'" + std::string(suf));
                    i += 1 + sl;
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
        }
        if (c < 0x80 && std::isalpha(c)) {
            std::string w;
            while (i < low.size() && std::isalpha((unsigned char)low[i])) {
                w.push_back(low[i]);
                i++;
            }
            words.push_back(w);
            continue;
        }
        if (c < 0x80 && std::isdigit(c)) {
            std::string w;
            while (i < low.size() && std::isdigit((unsigned char)low[i])) {
                w.push_back(low[i]);
                i++;
            }
            words.push_back(w);
            continue;
        }
        if (c < 0x80) {
            std::string w;
            while (i < low.size()) {
                unsigned char d = (unsigned char)low[i];
                if (d >= 0x80 || std::isalnum(d) || std::isspace(d)) break;
                w.push_back((char)d);
                i++;
            }
            words.push_back(w);
            continue;
        }
        // non-ASCII: per character (keeps CJK prompts usable)
        uint32_t cp = utf8One(low, i);
        std::string one;
        utf8Append(one, cp);
        words.push_back(one);
    }
    return words;
}

static std::vector<int> clipTokenize(const TokSet& ts, const std::string& text) {
    std::vector<int> ids;
    ids.push_back(49406);
    for (auto& w : preTokenize(text)) {
        auto t = bpeWord(ts, w);
        for (int id : t) ids.push_back(id);
        if (ids.size() >= 76) break;
    }
    if (ids.size() > 75) ids.resize(75);
    ids.push_back(49407);
    while (ids.size() < 77) ids.push_back(49407);
    return ids;
}

// ================================================================ CLIP text encoder

// pfx covers "<tag>.text_model."; proj is "<tag>.text_projection" (empty = none)
// idx: device buffer with the 77 token ids (uploaded by the caller outside the step);
// embOut/pooledOut: device buffers written here, downloaded by the caller after waitStep
static void clipEncode(const std::string& pfx, const std::string& proj, const BufRef& idx, const std::vector<int>& ids,
                       uint32_t D, uint32_t layers, uint32_t heads, uint32_t actMode, const BufRef& embOut,
                       const BufRef& pooledOut) {
    const uint32_t T = 77;
    uint32_t ffn = wdimT(pfx + "encoder.layers.0.mlp.fc1.weight");

    BufRef xSlot = allocN((uint64_t)T * D);
    {
        BufRef tok = allocN((uint64_t)T * D);
        P2 pg{T, D};
        G.eng.dispatch("gather_rows", {W(pfx + "embeddings.token_embedding.weight"), idx, tok}, &pg,
                       ceilDiv((uint64_t)T * D, 256), 1, 1);
        BufRef x0 = add2(tok, W(pfx + "embeddings.position_embedding.weight"), (uint64_t)T * D);
        if (gTrace && pfx == "te1.text_model.") tr("emb", x0, (uint64_t)T * D);
        copyInto(xSlot, 0, x0, (uint64_t)T * D * 4);
        arenaRewind(xSlot.offset + xSlot.size);
    }

    for (uint32_t l = 0; l < layers; l++) {
        std::string lp = pfx + "encoder.layers." + std::to_string(l) + ".";
        std::string tp = (pfx == "te1.text_model.") ? "t1_" : "t2_";
        bool tl = gTrace && l == 0;
        BufRef h = layernorm(xSlot, lp + "layer_norm1", T, D);
        if (tl) tr(tp + "ln1", h, (uint64_t)T * D);
        BufRef q = linear(h, lp + "self_attn.q_proj", T);
        if (tl) tr(tp + "q", q, (uint64_t)T * D);
        BufRef k = linear(h, lp + "self_attn.k_proj", T);
        BufRef v = linear(h, lp + "self_attn.v_proj", T);
        BufRef o = attnCore(q, k, v, T, T, D, heads, true);
        if (tl) tr(tp + "attn", o, (uint64_t)T * D);
        o = linear(o, lp + "self_attn.out_proj", T);
        if (tl) tr(tp + "oproj", o, (uint64_t)T * D);
        BufRef x1 = add2(xSlot, o, (uint64_t)T * D);
        if (tl) tr(tp + "x1", x1, (uint64_t)T * D);
        h = layernorm(x1, lp + "layer_norm2", T, D);
        if (tl) tr(tp + "ln2", h, (uint64_t)T * D);
        h = linear(h, lp + "mlp.fc1", T);
        if (tl) tr(tp + "fc1", h, (uint64_t)T * ffn);
        h = unary(h, (uint64_t)T * ffn, actMode);
        if (tl) tr(tp + "act", h, (uint64_t)T * ffn);
        h = linear(h, lp + "mlp.fc2", T);
        BufRef x2 = add2(x1, h, (uint64_t)T * D);
        if (tl) tr(tp + "x2", x2, (uint64_t)T * D);
        if (l + 2 == layers) copyInto(embOut, 0, x2, (uint64_t)T * D * 4);
        copyInto(xSlot, 0, x2, (uint64_t)T * D * 4);
        arenaRewind(xSlot.offset + xSlot.size);
    }

    if (!proj.empty()) {
        BufRef f = layernorm(xSlot, pfx + "final_layer_norm", T, D);
        uint32_t am = 0;
        for (uint32_t i = 1; i < T; i++)
            if (ids[i] > ids[am]) am = i;
        BufRef row{f.buf, f.offset + (uint64_t)am * D * 4, (uint64_t)D * 4};
        BufRef p = linear(row, proj, 1);
        copyInto(pooledOut, 0, p, (uint64_t)D * 4);
        arenaRewind(xSlot.offset + xSlot.size);
    }
}

// ================================================================ SDXL UNet

static BufRef resBlock(const BufRef& x, const std::string& p, const BufRef& temb, uint32_t H, uint32_t Wd,
                       const char* tag = "") {
    // conv weight dims are (cout, cin*k)
    uint32_t cout, kCols;
    linearDims(p + ".in_layers.2.weight", cout, kCols);
    uint32_t cin = kCols / 9;
    uint32_t P = H * Wd;
    bool sel = gTrace && *gDbSel && tag && *tag && strstr(gDbSel, tag) != nullptr;
    auto dt = [&](const char* suf, const BufRef& b, uint64_t floats) {
        if (sel) tr(std::string(tag) + suf, b, floats);
    };
    BufRef out = allocN((uint64_t)cout * P);
    BufRef h = gn(x, p + ".in_layers.0", cin, P, 32, 1e-5f);
    dt("_n1", h, (uint64_t)cin * P);
    h = silu(h, (uint64_t)cin * P);
    h = conv3Raw(h, p + ".in_layers.2.weight", cin, cout, H, Wd);
    dt("_c1", h, (uint64_t)cout * P);
    // diffusers ResnetBlock2D: temb = time_emb_proj(silu(temb)) before it is added to conv1
    uint32_t ecout, etembDim;
    linearDims(p + ".emb_layers.1.weight", ecout, etembDim);
    BufRef e = linear(silu(temb, etembDim), p + ".emb_layers.1", 1);
    dt("_e", e, cout);
    biasAddCh(h, e, cout, P);
    h = gn(h, p + ".out_layers.0", cout, P, 32, 1e-5f);
    dt("_n2", h, (uint64_t)cout * P);
    h = silu(h, (uint64_t)cout * P);
    h = conv3Raw(h, p + ".out_layers.3.weight", cout, cout, H, Wd);
    dt("_c2", h, (uint64_t)cout * P);
    BufRef sc = x;
    if (cin != cout) {
        sc = conv1(x, p + ".skip_connection", cin, cout, P);
        dt("_sc", sc, (uint64_t)cout * P);
    }
    BufRef r = add2(sc, h, (uint64_t)cout * P);
    copyInto(out, 0, r, (uint64_t)cout * P * 4);
    G.eng.barrier();
    arenaRewind(out.offset + out.size);
    return out;
}

// diffusers GEGLU: y = gelu(secondHalf) * firstHalf, halves of the nout-wide linear output
static BufRef geglu(const BufRef& x, const std::string& p, uint32_t rows) {
    uint32_t nout, nin;
    linearDims(p + ".weight", nout, nin);
    uint32_t F = nout / 2;
    BufRef h = linear(x, p, rows);
    BufRef a = copyNd(h, rows * F, F, rows, 1, 1, F, 0, 1, nout, 0);
    BufRef src{h.buf, h.offset + (uint64_t)F * 4, h.size - (uint64_t)F * 4};
    BufRef b = copyNd(src, rows * F, F, rows, 1, 1, F, 0, 1, nout, 0);
    b = geluErf(b, (uint64_t)rows * F);
    return mul2(a, b, (uint64_t)rows * F);
}

static BufRef spatialTransformer(const BufRef& x, const std::string& p, const BufRef& ctx, uint32_t H, uint32_t Wd,
                                 uint32_t layers, const char* tag = "") {
    uint32_t C, tmp;
    linearDims(p + ".norm.weight", C, tmp);
    uint32_t P = H * Wd;
    uint32_t heads = C / 64;
    bool sel = gTrace && *gDbSel && tag && *tag && strstr(gDbSel, tag) != nullptr;
    auto dt = [&](const char* suf, const BufRef& b, uint64_t floats) {
        if (sel) tr(std::string(tag) + suf, b, floats);
    };
    BufRef outSlot = allocN((uint64_t)C * P);
    BufRef h = gn(x, p + ".norm", C, P, 32, 1e-6f);
    dt("_gn", h, (uint64_t)C * P);
    h = conv1(h, p + ".proj_in", C, C, P);
    dt("_p", h, (uint64_t)C * P);
    BufRef t = transpose2(h, C, P);
    dt("_t", t, (uint64_t)P * C);
    for (uint32_t i = 0; i < layers; i++) {
        std::string tp = p + ".transformer_blocks." + std::to_string(i);
        bool l0 = sel && i == 0;
        BufRef n = layernorm(t, tp + ".norm1", P, C);
        if (l0) dt("_ln1", n, (uint64_t)P * C);
        BufRef a = crossAttention(n, n, tp + ".attn1", P, P, C, heads);
        if (l0) dt("_a1", a, (uint64_t)P * C);
        t = add2(t, a, (uint64_t)P * C);
        n = layernorm(t, tp + ".norm2", P, C);
        if (l0) dt("_ln2", n, (uint64_t)P * C);
        a = crossAttention(n, ctx, tp + ".attn2", P, 77, C, heads);
        if (l0) dt("_a2", a, (uint64_t)P * C);
        t = add2(t, a, (uint64_t)P * C);
        n = layernorm(t, tp + ".norm3", P, C);
        if (l0) dt("_ln3", n, (uint64_t)P * C);
        BufRef f = geglu(n, tp + ".ff.net.0.proj", P);
        if (l0) {
            uint32_t r0, c0;
            linearDims(tp + ".ff.net.0.proj.weight", r0, c0);
            dt("_g", f, (uint64_t)P * (r0 / 2));
        }
        f = linear(f, tp + ".ff.net.2", P);
        if (l0) dt("_f", f, (uint64_t)P * C);
        t = add2(t, f, (uint64_t)P * C);
    }
    BufRef o = transpose2(t, P, C);
    o = conv1(o, p + ".proj_out", C, C, P);
    dt("_po", o, (uint64_t)C * P);
    BufRef r = add2(x, o, (uint64_t)C * P);
    copyInto(outSlot, 0, r, (uint64_t)C * P * 4);
    G.eng.barrier();
    arenaRewind(outSlot.offset + outSlot.size);
    return outSlot;
}

// sinusoidal embedding (diffusers get_timestep_embedding, flip_sin_to_cos=True, shift=0)
static std::vector<float> timeEmbHost(float t, uint32_t dim) {
    std::vector<float> e(dim);
    uint32_t half = dim / 2;
    for (uint32_t i = 0; i < half; i++) {
        float f = std::exp(-std::log(10000.0f) * (float)i / (float)half);
        e[i] = std::cos(t * f);
        e[half + i] = std::sin(t * f);
    }
    return e;
}

static BufRef concatCh(const BufRef& a, uint32_t ca, const BufRef& b, uint32_t cb, uint32_t P) {
    BufRef y = allocN((uint64_t)(ca + cb) * P);
    copyInto(y, 0, a, (uint64_t)ca * P * 4);
    copyInto(y, (uint64_t)ca * P * 4, b, (uint64_t)cb * P * 4);
    return y;
}

static BufRef unetForward(const BufRef& lat, const BufRef& temb, const BufRef& ctx, uint32_t H, uint32_t Wd) {
    // stage tags mirror ref/unet_stages.py so dbg_<tag>.f32 files line up 1:1
    static const char* kDnRes[6] = {"d0_r0", "d0_r1", "d1_r0", "d1_r1", "d2_r0", "d2_r1"};
    static const char* kDnAttn[6] = {"", "", "d1_a0", "d1_a1", "d2_a0", "d2_a1"};
    static const char* kUpRes[9] = {"u0_r0", "u0_r1", "u0_r2", "u1_r0", "u1_r1", "u1_r2", "u2_r0", "u2_r1", "u2_r2"};
    static const char* kUpAttn[9] = {"u0_a0", "u0_a1", "u0_a2", "u1_a0", "u1_a1", "u1_a2", "", "", ""};
    auto dbg = [&](const char* tag, const BufRef& b, uint32_t c, uint32_t hh, uint32_t ww) {
        if (gTrace && tag[0]) tr(tag, b, (uint64_t)c * hh * ww);
    };
    std::vector<BufRef> hs;
    std::vector<uint32_t> hch;
    uint32_t h = H, w = Wd;
    BufRef x = conv3Raw(lat, "input_blocks.0.0.weight", 4, 320, H, Wd);
    uint32_t ch = 320;
    dbg("conv_in", x, ch, h, w);
    hs.push_back(x);
    hch.push_back(ch);

    auto downModule = [&](uint32_t i, const std::string& pre, uint32_t h, uint32_t w, uint32_t tlayers) {
        x = resBlock(x, pre + ".0", temb, h, w, kDnRes[i]);
        ch = wdimT(pre + ".0.out_layers.3.weight");
        dbg(kDnRes[i], x, ch, h, w);
        if (tlayers) x = spatialTransformer(x, pre + ".1", ctx, h, w, tlayers, kDnAttn[i]);
        dbg(kDnAttn[i], x, ch, h, w);
        hs.push_back(x);
        hch.push_back(ch);
    };
    downModule(0, "input_blocks.1", h, w, 0);
    downModule(1, "input_blocks.2", h, w, 0);
    x = conv3s2(x, "input_blocks.3.0.op", h, w);
    ch = wdimT("input_blocks.3.0.op.weight");
    h /= 2;
    w /= 2;
    dbg("d0_ds", x, ch, h, w);
    hs.push_back(x);
    hch.push_back(ch);

    downModule(2, "input_blocks.4", h, w, 2);
    downModule(3, "input_blocks.5", h, w, 2);
    x = conv3s2(x, "input_blocks.6.0.op", h, w);
    ch = wdimT("input_blocks.6.0.op.weight");
    h /= 2;
    w /= 2;
    dbg("d1_ds", x, ch, h, w);
    hs.push_back(x);
    hch.push_back(ch);

    downModule(4, "input_blocks.7", h, w, 10);
    downModule(5, "input_blocks.8", h, w, 10);

    x = resBlock(x, "middle_block.0", temb, h, w, "m_r0");
    ch = wdimT("middle_block.0.out_layers.3.weight");
    dbg("m_r0", x, ch, h, w);
    x = spatialTransformer(x, "middle_block.1", ctx, h, w, 10, "m_a0");
    dbg("m_a0", x, ch, h, w);
    x = resBlock(x, "middle_block.2", temb, h, w, "m_r1");
    ch = wdimT("middle_block.2.out_layers.3.weight");
    dbg("m_r1", x, ch, h, w);

    for (int j = 0; j < 9; j++) {
        std::string pre = "output_blocks." + std::to_string(j);
        uint32_t cs = hch.back();
        BufRef skip = hs.back();
        hs.pop_back();
        hch.pop_back();
        x = concatCh(x, ch, skip, cs, h * w);
        ch = ch + cs;
        uint32_t tlayers = (j < 3) ? 10 : (j < 6) ? 2 : 0;
        x = resBlock(x, pre + ".0", temb, h, w, kUpRes[j]);
        ch = wdimT(pre + ".0.out_layers.3.weight");
        dbg(kUpRes[j], x, ch, h, w);
        if (tlayers) x = spatialTransformer(x, pre + ".1", ctx, h, w, tlayers, kUpAttn[j]);
        dbg(kUpAttn[j], x, ch, h, w);
        std::string upPfx;
        if (hasAnyW(pre + ".2.conv.weight")) upPfx = ".2.conv";
        else if (hasAnyW(pre + ".1.conv.weight")) upPfx = ".1.conv";
        if (!upPfx.empty()) {
            x = up2(x, ch, h, w);
            h *= 2;
            w *= 2;
            x = conv3Raw(x, pre + upPfx + ".weight", ch, ch, h, w);
            dbg(j < 3 ? "u0_up" : "u1_up", x, ch, h, w);
        }
    }

    x = gn(x, "out.0", ch, h * w, 32, 1e-5f);
    dbg("conv_gn", x, ch, h, w);
    x = silu(x, (uint64_t)ch * h * w);
    dbg("conv_act", x, ch, h, w);
    x = conv3Raw(x, "out.2.weight", ch, 4, h, w);
    dbg("conv_out", x, 4, h, w);
    return x;
}

// temb = time_embed(get_timestep_embedding(t)) + add_embedding(concat(pooled, add_time_ids))
static BufRef buildTemb(const BufRef& pool, const BufRef& addTime1536, const BufRef& tEmb320) {
    BufRef t = linear(tEmb320, "time_embed.0", 1);
    t = silu(t, (uint64_t)wdimT("time_embed.0.weight"));
    t = linear(t, "time_embed.2", 1);
    BufRef cat = allocN(2816);
    copyInto(cat, 0, pool, 1280 * 4);
    copyInto(cat, 1280 * 4, addTime1536, 1536 * 4);
    BufRef a = linear(cat, "label_emb.0.0", 1);
    a = silu(a, (uint64_t)wdimT("label_emb.0.0.weight"));
    a = linear(a, "label_emb.0.2", 1);
    return add2(t, a, 1280);
}

// ================================================================ VAE decoder

static BufRef vaeAttn(const BufRef& x, const std::string& p, uint32_t C, uint32_t H, uint32_t Wd) {
    uint32_t P = H * Wd;
    BufRef slot = allocN((uint64_t)C * P);
    BufRef h = gn(x, p + ".group_norm", C, P, 32, 1e-6f);
    BufRef q = conv1(h, p + ".to_q", C, C, P);
    BufRef k = conv1(h, p + ".to_k", C, C, P);
    BufRef v = conv1(h, p + ".to_v", C, C, P);
    BufRef qt = transpose2(q, C, P);
    scaleInPlace(qt, (uint64_t)C * P, 1.0f / std::sqrt((float)C));
    BufRef s = allocN((uint64_t)P * P);
    gemmNn(qt, k, s, P, P, C, 0);
    BufRef pr = softmax(s, P, P);
    BufRef o = allocN((uint64_t)C * P);
    gemmNt(v, pr, o, C, P, P);
    BufRef po = conv1(o, p + ".to_out.0", C, C, P);
    BufRef r = add2(x, po, (uint64_t)C * P);
    copyInto(slot, 0, r, (uint64_t)C * P * 4);
    G.eng.barrier();
    arenaRewind(slot.offset + slot.size);
    return slot;
}

static BufRef vaeResnet(const BufRef& x, const std::string& p, uint32_t cin, uint32_t cout, uint32_t H, uint32_t Wd,
                        bool shortcut) {
    uint32_t P = H * Wd;
    BufRef slot = allocN((uint64_t)cout * P);
    BufRef h = gn(x, p + ".norm1", cin, P, 32, 1e-6f);
    h = silu(h, (uint64_t)cin * P);
    h = conv3Raw(h, p + ".conv1.weight", cin, cout, H, Wd);
    h = gn(h, p + ".norm2", cout, P, 32, 1e-6f);
    h = silu(h, (uint64_t)cout * P);
    h = conv3Raw(h, p + ".conv2.weight", cout, cout, H, Wd);
    BufRef sc = x;
    if (shortcut) sc = conv1(x, p + ".conv_shortcut", cin, cout, P);
    BufRef r = add2(sc, h, (uint64_t)cout * P);
    copyInto(slot, 0, r, (uint64_t)cout * P * 4);
    G.eng.barrier();
    arenaRewind(slot.offset + slot.size);
    return slot;
}

static BufRef vaeUpConv(const BufRef& x, const std::string& p, uint32_t C, uint32_t& H, uint32_t& Wd) {
    BufRef u = up2(x, C, H, Wd);
    H *= 2;
    Wd *= 2;
    return conv3Raw(u, p + ".weight", C, C, H, Wd);
}

static void writePPM(const std::string& path, const std::vector<float>& img, uint32_t H, uint32_t Wd) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::printf("cannot write %s\n", path.c_str());
        return;
    }
    std::fprintf(f, "P6\n%u %u\n255\n", Wd, H);
    std::vector<uint8_t> row((size_t)Wd * 3);
    for (uint32_t y = 0; y < H; y++) {
        for (uint32_t x = 0; x < Wd; x++) {
            for (uint32_t c = 0; c < 3; c++) {
                float v = img[((uint64_t)c * H + y) * Wd + x] * 0.5f + 0.5f;
                v = std::max(0.0f, std::min(1.0f, v));
                row[(size_t)x * 3 + c] = (uint8_t)std::lround(v * 255.0f);
            }
        }
        std::fwrite(row.data(), 1, row.size(), f);
    }
    std::fclose(f);
}

// ================================================================ scheduler (EulerDiscrete, leading)

struct Euler {
    std::vector<float> timesteps;
    std::vector<float> sigmas;
    float initNoiseSigma = 1.0f;
};

// diffusers EulerDiscrete::set_timesteps: "leading" (SD1.5 style, steps_offset=1) and "trailing"
// (what the SDXL repos ship). The two differ in which timesteps get sampled AND in the init scale.
// zsnr = rescale_betas_zero_snr (zero terminal SNR, the recommended schedule for v-pred SDXL):
// sqrt(alpha_bar) is shifted so the terminal entry is 0 and rescaled so the first is unchanged;
// the terminal alpha_bar is then clamped to 2^-24 (sigma_max ~4096). The t fed to the model is
// unchanged - only the sigma levels move.
static Euler buildEuler(int steps, const char* spacing = "leading", int numTrain = 1000, int offset = 1,
                        bool zsnr = false) {
    std::vector<float> betas(numTrain);
    for (int i = 0; i < numTrain; i++) {
        float b0 = 0.00085f, b1 = 0.012f;
        float v = std::sqrt(b0) + (std::sqrt(b1) - std::sqrt(b0)) * (float)i / (float)(numTrain - 1);
        betas[i] = v * v;
    }
    std::vector<double> ab(numTrain);
    double run = 1.0;
    for (int i = 0; i < numTrain; i++) {
        run *= (1.0 - betas[i]);
        ab[i] = run;
    }
    if (zsnr) {
        double a0 = std::sqrt(ab[0]), aT = std::sqrt(ab[numTrain - 1]);
        for (int i = 0; i < numTrain; i++) {
            double s = (std::sqrt(ab[i]) - aT) * (a0 / (a0 - aT));
            ab[i] = s * s;
        }
        ab[numTrain - 1] = std::pow(2.0, -24.0);
    }
    std::vector<float> sigmaOf(numTrain);
    for (int i = 0; i < numTrain; i++) sigmaOf[i] = (float)std::sqrt((1.0 - ab[i]) / ab[i]);
    Euler e;
    bool trailing = std::strcmp(spacing, "trailing") == 0;
    e.timesteps.resize(steps);
    if (trailing) {
        float stepRatio = (float)numTrain / (float)steps;
        for (int i = 0; i < steps; i++)
            e.timesteps[i] = std::floor((float)numTrain - (float)i * stepRatio + 0.5f) - 1.0f;
    } else {
        int stepRatio = numTrain / steps;
        for (int i = 0; i < steps; i++)
            e.timesteps[steps - 1 - i] = (float)std::lround((float)i * stepRatio) + (float)offset;
    }
    e.sigmas.resize(steps + 1);
    for (int i = 0; i < steps; i++) {
        float t = e.timesteps[i];
        if (t <= 0) e.sigmas[i] = sigmaOf[0];
        else if (t >= numTrain - 1) e.sigmas[i] = sigmaOf[numTrain - 1];
        else {
            int i0 = (int)t;
            float fr = t - (float)i0;
            e.sigmas[i] = sigmaOf[i0] * (1.0f - fr) + sigmaOf[i0 + 1] * fr;
        }
    }
    e.sigmas[steps] = 0.0f;
    float mx = 0.0f;
    for (float s : e.sigmas) mx = std::max(mx, s);
    e.initNoiseSigma = trailing ? mx : std::sqrt(mx * mx + 1.0f);
    return e;
}

// ================================================================ main

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string prompt = argc > 1 ? argv[1] : "1girl, silver hair, green eyes, masterpiece, best quality";
    std::string neg = "";
    std::string gguf = "E:/ofc-data/noobai/noobaiXLNAIXL_vPred10Version-Q8_0.gguf";
    std::string te1Path = "E:/ofc-data/noobai/text_encoder__model.safetensors";
    std::string te2Path = "E:/ofc-data/noobai/text_encoder_2__model.safetensors";
    std::string vaePath = "E:/ofc-data/noobai/vae.safetensors";
    std::string tokDir = "E:/ofc-data/noobai/tokenizer";
    std::string tok2Dir = "E:/ofc-data/noobai/tokenizer_2";
    std::string out = "sdvkgen.ppm";
    std::string latentOut = "sdvkgen_latent.bin";
    int steps = 20;
    float cfg = 7.5f;
    uint32_t size = 512;
    uint32_t seed = 1234;
    bool vpred = false;
    float rescale = 0.0f;
    bool dynthresh = false;
    bool textcheck = false;
    bool clipdbg = false;
    bool probe = false;
    bool unetdbg = false;
    std::string dbgStage;
    std::string spacing = "leading";
    std::string probeX;
    std::string latIn;
    float probeT = 999.0f;
    bool noVae = false;
    bool zsnr = false;
    std::string embCondFile, embUncondFile, embPoolFile, embPoolUFile;
    uint32_t stepMbOverride = 0, scratchMbOverride = 0;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--neg" && i + 1 < argc) neg = argv[++i];
        else if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--te1" && i + 1 < argc) te1Path = argv[++i];
        else if (a == "--te2" && i + 1 < argc) te2Path = argv[++i];
        else if (a == "--vae" && i + 1 < argc) vaePath = argv[++i];
        else if (a == "--tok" && i + 1 < argc) tokDir = argv[++i];
        else if (a == "--tok2" && i + 1 < argc) tok2Dir = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--steps" && i + 1 < argc) steps = std::stoi(argv[++i]);
        else if (a == "--cfg" && i + 1 < argc) cfg = std::stof(argv[++i]);
        else if (a == "--rescale" && i + 1 < argc) rescale = std::stof(argv[++i]);
        else if (a == "--size" && i + 1 < argc) size = (uint32_t)std::stoul(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) seed = (uint32_t)std::stoul(argv[++i]);
        else if (a == "--vpred") vpred = true;
        else if (a == "--zsnr") zsnr = true;
        else if (a == "--dynthresh") dynthresh = true;
        else if (a == "--textcheck") textcheck = true;
        else if (a == "--clipdbg") clipdbg = true;
        else if (a == "--unetdbg") unetdbg = true;
        else if (a == "--dbgstage" && i + 1 < argc) dbgStage = argv[++i];
        else if (a == "--spacing" && i + 1 < argc) spacing = argv[++i];
        else if (a == "--probe") probe = true;
        else if (a == "--probe-x" && i + 1 < argc) probeX = argv[++i];
        else if (a == "--probe-t" && i + 1 < argc) probeT = std::stof(argv[++i]);
        else if (a == "--lat-in" && i + 1 < argc) latIn = argv[++i];
        else if (a == "--no-vae") noVae = true;
        else if (a == "--emb-cond" && i + 1 < argc) embCondFile = argv[++i];
        else if (a == "--emb-uncond" && i + 1 < argc) embUncondFile = argv[++i];
        else if (a == "--emb-pool" && i + 1 < argc) embPoolFile = argv[++i];
        else if (a == "--emb-poolu" && i + 1 < argc) embPoolUFile = argv[++i];
        else if (a == "--step-mb" && i + 1 < argc) stepMbOverride = (uint32_t)std::stoul(argv[++i]);
        else if (a == "--scratch-mb" && i + 1 < argc) scratchMbOverride = (uint32_t)std::stoul(argv[++i]);
    }
    if (size % 8 != 0 || size < 64) {
        std::printf("bad --size\n");
        return 1;
    }
    uint32_t latentDim = size / 8;

    static TokSet tok1, tok2;
    buildByteMap(tok1);
    buildByteMap(tok2);
    loadVocab(tok1, tokDir + "/vocab.json");
    loadMerges(tok1, tokDir + "/merges.txt");
    loadVocab(tok2, tok2Dir + "/vocab.json");
    loadMerges(tok2, tok2Dir + "/merges.txt");
    std::printf("tokenizer: vocab %zu/%zu merges %zu/%zu\n", tok1.vocab.size(), tok2.vocab.size(), tok1.merges.size(),
                tok2.merges.size());

    if (!gGG.open(gguf)) {
        std::printf("FAIL: gguf %s\n", gguf.c_str());
        return 1;
    }
    std::printf("gguf: %zu tensors\n", gGG.tensorCount());

    static Safetensors stTE1, stTE2, stVAE;
    if (!stTE1.open(te1Path)) {
        std::printf("FAIL: %s\n", stTE1.error().c_str());
        return 1;
    }
    if (!stTE2.open(te2Path)) {
        std::printf("FAIL: %s\n", stTE2.error().c_str());
        return 1;
    }
    if (!stVAE.open(vaePath)) {
        std::printf("FAIL: %s\n", stVAE.error().c_str());
        return 1;
    }

    // ---------------- phase A: text encoders (param arena reused afterwards)
    // text encoders need 3.3GB; even when TE is skipped, ~2.9GB must remain for the UNet's Q8_0 payload+scales
    G.paramBytes = embCondFile.empty() ? 3300ull << 20 : 3000ull << 20;
    G.allocGradArena = false;
    G.stepBytes = 96ull << 20;
    G.scratchBytes = 96ull << 20;
    gInit("shaders");
    std::printf("device: %s\n", G.eng.deviceName().c_str());

    std::vector<float> embCond, embUncond, poolC, poolU;
    auto encodeBoth = [&](const std::vector<int>& i1, const std::vector<int>& i2, std::vector<float>& emb,
                          std::vector<float>& pool) {
        std::vector<float> eL(77 * 768), eG(77 * 1280), p(1280, 0.0f);
        BufRef idx1 = G.scratchArena.alloc(77 * 4);
        BufRef idx2 = G.scratchArena.alloc(77 * 4);
        BufRef out1 = G.scratchArena.alloc(77ull * 768 * 4);
        BufRef out2 = G.scratchArena.alloc(77ull * 1280 * 4);
        BufRef outP = G.scratchArena.alloc(1280 * 4);
        G.eng.uploadNow(idx1, i1.data(), 77 * 4);
        G.eng.uploadNow(idx2, i2.data(), 77 * 4);
        G.eng.beginStep();
        clipEncode("te1.text_model.", "", idx1, i1, 768, 12, 12, 3, out1, outP);
        clipEncode("te2.text_model.", "te2.text_projection", idx2, i2, 1280, 32, 20, 4, out2, outP);
        G.eng.endStep();
        G.eng.waitStep();
        G.eng.downloadNow(out1, eL.data(), eL.size() * 4);
        G.eng.downloadNow(out2, eG.data(), eG.size() * 4);
        G.eng.downloadNow(outP, p.data(), p.size() * 4);
        emb.assign(77 * 2048, 0.0f);
        for (uint32_t t = 0; t < 77; t++) {
            std::memcpy(&emb[(size_t)t * 2048], &eL[(size_t)t * 768], 768 * 4);
            std::memcpy(&emb[(size_t)t * 2048 + 768], &eG[(size_t)t * 1280], 1280 * 4);
        }
        pool = p;
    };

    if (!embCondFile.empty()) {
        auto loadRaw = [](const std::string& p, size_t n, const char* what) {
            std::vector<float> v(n);
            FILE* f = std::fopen(p.c_str(), "rb");
            if (!f) die(std::string("cannot open ") + what + " " + p);
            if (std::fread(v.data(), 4, n, f) != n) die(std::string("short ") + p);
            std::fclose(f);
            return v;
        };
        embCond = loadRaw(embCondFile, 77ull * 2048, "emb-cond");
        // model_index.json force_zeros_for_empty_prompt=true: empty neg prompt means
        // zero embeddings, not empty-string encoding (empty string still fires EOS;
        // its ctx has rms 0.69 and cos 0.844 with cond, which wrecks CFG at 7.5)
        embUncond = embUncondFile.empty() ? std::vector<float>(77ull * 2048, 0.0f)
                                          : loadRaw(embUncondFile, 77ull * 2048, "emb-uncond");
        poolC = embPoolFile.empty() ? std::vector<float>(1280, 0.0f) : loadRaw(embPoolFile, 1280, "emb-pool");
        poolU = embPoolUFile.empty() ? std::vector<float>(1280, 0.0f) : loadRaw(embPoolUFile, 1280, "emb-poolu");
        clearWeights();
        G.paramArena.reset();
        std::printf("emb from files (te phase skipped)\n");
    } else {
        loadStWeights(stTE1, "te1.");
        loadStWeights(stTE2, "te2.");
        if (clipdbg) {
            gTrace = true;
            std::vector<int> idc = clipTokenize(tok1, prompt);
            BufRef idxD = G.scratchArena.alloc(77 * 4);
            G.eng.uploadNow(idxD, idc.data(), 77 * 4);
            std::vector<int> idr(77);
            G.eng.downloadNow(idxD, idr.data(), 77 * 4);
            std::printf("idx check: host[0..6] =");
            for (int i = 0; i < 7; i++) std::printf(" %d", idc[i]);
            std::printf("  dev[0..6] =");
            for (int i = 0; i < 7; i++) std::printf(" %d", idr[i]);
            std::printf("\n");
            const BufRef& teW = W("te1.text_model.embeddings.token_embedding.weight");
            for (uint32_t row : {16u, 272u, 49406u}) {
                std::vector<float> v(768);
                BufRef r{teW.buf, teW.offset + (uint64_t)row * 768 * 4, 768 * 4};
                G.eng.downloadNow(r, v.data(), 768 * 4);
                std::printf("vram te[%u]: %.6f %.6f %.6f %.6f\n", row, v[0], v[1], v[2], v[3]);
            }
        }
        encodeBoth(clipTokenize(tok1, prompt), clipTokenize(tok2, prompt), embCond, poolC);
        if (clipdbg) {
            flushTrace();
            gTrace = false;
        }
        if (neg.empty()) {
            embUncond.assign(77ull * 2048, 0.0f);
            poolU.assign(1280, 0.0f);
        } else {
            encodeBoth(clipTokenize(tok1, neg), clipTokenize(tok2, neg), embUncond, poolU);
        }
    }
    double ne = 0, nc = 0, nu = 0;
    for (float v : embCond) ne += (double)v * v;
    for (float v : poolC) nc += (double)v * v;
    for (float v : poolU) nu += (double)v * v;
    std::printf("ctx cond rms %.5f  |pooled_c| %.4f |pooled_u| %.4f\n", std::sqrt(ne / embCond.size()), std::sqrt(nc),
                std::sqrt(nu));

    if (textcheck) {
        FILE* f = std::fopen("sdvkgen_ctx_cond.f32", "wb");
        std::fwrite(embCond.data(), 4, embCond.size(), f);
        std::fclose(f);
        f = std::fopen("sdvkgen_ctx_uncond.f32", "wb");
        std::fwrite(embUncond.data(), 4, embUncond.size(), f);
        std::fclose(f);
        f = std::fopen("sdvkgen_pooled_cond.f32", "wb");
        std::fwrite(poolC.data(), 4, poolC.size(), f);
        std::fclose(f);
        f = std::fopen("sdvkgen_pooled_uncond.f32", "wb");
        std::fwrite(poolU.data(), 4, poolU.size(), f);
        std::fclose(f);
        std::printf("textcheck: wrote sdvkgen_ctx_*.f32 / sdvkgen_pooled_*.f32\n");
        gShutdown();
        return 0;
    }

    // ---------------- phase B: UNet
    clearWeights();
    G.paramArena.reset();
    loadGgufUnet();
    uint32_t stepMB = stepMbOverride ? stepMbOverride : (size >= 768 ? 1500u : 900u);
    uint32_t scratchMB = scratchMbOverride ? scratchMbOverride : (size >= 768 ? 900u : 420u);
    // Engine::initArenas only resizes the engine's own arenas; the persistent G.scratchArena
    // (gCol / ctx buffers / hp) lives in gpu_ops and must be resized with it.
    G.eng.initArenas((uint64_t)stepMB << 20, (uint64_t)scratchMB << 20);
    gResizeScratch((uint64_t)scratchMB << 20);
    gCol = G.scratchArena.alloc(kColFloats * 4);
    gSCap = 0;
    std::printf("arenas: step %u MB scratch %u MB\n", stepMB, scratchMB);

    static std::unordered_map<std::string, BufRef> sCache;
    auto scratchBuf = [&](const std::string& key, uint64_t bytes) {
        auto it = sCache.find(key);
        if (it != sCache.end()) return it->second;
        BufRef b = G.scratchArena.alloc((bytes + 255) & ~255ull);
        sCache[key] = b;
        return b;
    };
    BufRef ctxC = scratchBuf("__ctx_cond", 77ull * 2048 * 4);
    BufRef ctxU = scratchBuf("__ctx_uncond", 77ull * 2048 * 4);
    BufRef poolCB = scratchBuf("__pool_cond", 1280 * 4);
    BufRef poolUB = scratchBuf("__pool_uncond", 1280 * 4);
    BufRef addTimeB = scratchBuf("__addtime", 1536 * 4);
    BufRef xLat = scratchBuf("__latent", (uint64_t)4 * latentDim * latentDim * 4);
    BufRef embHost = scratchBuf("__temb", 320 * 4);
    G.eng.uploadNow(ctxC, embCond.data(), embCond.size() * 4);
    G.eng.uploadNow(ctxU, embUncond.data(), embUncond.size() * 4);
    G.eng.uploadNow(poolCB, poolC.data(), poolC.size() * 4);
    G.eng.uploadNow(poolUB, poolU.data(), poolU.size() * 4);

    std::vector<float> addTime(1536);
    float tv[6] = {(float)size, (float)size, 0.0f, 0.0f, (float)size, (float)size};
    for (int i = 0; i < 6; i++) {
        auto e = timeEmbHost(tv[i], 256);
        std::memcpy(&addTime[(size_t)i * 256], e.data(), 256 * 4);
    }
    G.eng.uploadNow(addTimeB, addTime.data(), addTime.size() * 4);

    uint64_t latFloats = (uint64_t)4 * latentDim * latentDim;

    // ---------------- probe mode: one UNet pass on a fixed latent
    if (probe) {
        std::vector<float> lat(latFloats);
        if (!probeX.empty()) {
            FILE* f = std::fopen(probeX.c_str(), "rb");
            if (!f) {
                std::printf("cannot open %s\n", probeX.c_str());
                return 1;
            }
            char magic[4];
            uint32_t dims[3];
            if (std::fread(magic, 1, 4, f) != 4) {
                std::printf("bad probe-x\n");
                return 1;
            }
            std::fread(dims, 4, 3, f);
            if (dims[1] != latentDim) {
                std::printf("probe-x dim %u != %u\n", dims[1], latentDim);
                return 1;
            }
            if (std::fread(lat.data(), 4, lat.size(), f) != lat.size()) {
                std::printf("short probe-x\n");
                return 1;
            }
            std::fclose(f);
        } else {
            std::mt19937 rng(seed);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            for (auto& v : lat) v = nd(rng);
        }
        G.eng.uploadNow(xLat, lat.data(), lat.size() * 4);
        auto te = timeEmbHost(probeT, 320);
        G.eng.uploadNow(embHost, te.data(), te.size() * 4);
        G.eng.beginStep();
        BufRef temb = buildTemb(poolCB, addTimeB, embHost);
        gTrace = unetdbg;
        gDbSel = dbgStage.c_str();
        if (gTrace) tr("temb", temb, 1280);
        BufRef eps = unetForward(xLat, temb, ctxC, latentDim, latentDim);
        G.eng.endStep();
        G.eng.waitStep();
        flushTrace();
        gTrace = false;
        std::vector<float> ep(latFloats), tv3(1280);
        G.eng.downloadNow(eps, ep.data(), ep.size() * 4);
        G.eng.downloadNow(temb, tv3.data(), tv3.size() * 4);
        FILE* fx = std::fopen("probe_x.f32", "wb");
        std::fwrite(lat.data(), 4, lat.size(), fx);
        std::fclose(fx);
        FILE* fe = std::fopen("probe_eps.f32", "wb");
        std::fwrite(ep.data(), 4, ep.size(), fe);
        std::fclose(fe);
        FILE* ft = std::fopen("probe_temb.f32", "wb");
        std::fwrite(tv3.data(), 4, tv3.size(), ft);
        std::fclose(ft);
        double rms = 0;
        uint64_t bad = 0;
        for (float v : ep) {
            if (!std::isfinite(v)) bad++;
            else rms += (double)v * v;
        }
        std::printf("probe: t=%.1f dim=%u eps rms %.5f bad %llu -> probe_x.f32 probe_eps.f32 probe_temb.f32\n", probeT,
                    latentDim, std::sqrt(rms / (double)ep.size()), (unsigned long long)bad);
        gShutdown();
        return 0;
    }

    // ---------------- sampling
    Euler sch = buildEuler(steps, spacing.c_str(), 1000, 1, zsnr);
    std::printf("euler: t0 %.1f  init_sigma %.5f  sigma[0] %.5f\n", sch.timesteps[0], sch.initNoiseSigma,
                sch.sigmas[0]);

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> lat(latFloats);
    for (auto& v : lat) v = nd(rng) * sch.initNoiseSigma;
    // a host-supplied starting latent makes the engine trajectory bit-comparable with the
    // python reference sampler (same lat0 -> same schedule -> same numbers)
    if (!latIn.empty()) {
        FILE* f = std::fopen(latIn.c_str(), "rb");
        if (!f) {
            std::printf("cannot open %s\n", latIn.c_str());
            return 1;
        }
        char magic[4];
        uint32_t dims[3];
        std::fread(magic, 1, 4, f);
        std::fread(dims, 4, 3, f);
        if (dims[1] != latentDim) {
            std::printf("lat-in dim %u != %u\n", dims[1], latentDim);
            return 1;
        }
        if (std::fread(lat.data(), 4, lat.size(), f) != lat.size()) {
            std::printf("short lat-in\n");
            return 1;
        }
        std::fclose(f);
        std::printf("lat from %s\n", latIn.c_str());
    }
    G.eng.uploadNow(xLat, lat.data(), lat.size() * 4);

    auto runPass = [&](const BufRef& pool, const BufRef& ctx, float t, std::vector<float>& epsOut) {
        auto te = timeEmbHost(t, 320);
        G.eng.uploadNow(embHost, te.data(), te.size() * 4);
        G.eng.beginStep();
        BufRef temb = buildTemb(pool, addTimeB, embHost);
        BufRef eps = unetForward(xLat, temb, ctx, latentDim, latentDim);
        G.eng.endStep();
        G.eng.waitStep();
        G.eng.downloadNow(eps, epsOut.data(), epsOut.size() * 4);
    };

    std::vector<float> epsU(latFloats), epsC(latFloats);
    std::vector<float> scaled(latFloats);
    for (int i = 0; i < steps; i++) {
        float t = sch.timesteps[i];
        float sigma = sch.sigmas[i];
        float inv = 1.0f / std::sqrt(sigma * sigma + 1.0f);
        for (uint64_t k = 0; k < latFloats; k++) scaled[k] = lat[k] * inv;
        G.eng.uploadNow(xLat, scaled.data(), scaled.size() * 4);

        auto ta = std::chrono::steady_clock::now();
        runPass(poolUB, ctxU, t, epsU);
        runPass(poolCB, ctxC, t, epsC);
        auto tb = std::chrono::steady_clock::now();

        uint64_t bad = 0;
        for (uint64_t k = 0; k < latFloats; k++)
            if (!std::isfinite(epsC[k]) || !std::isfinite(epsU[k])) bad++;
        if (bad) {
            std::printf("ERROR step %d: eps non-finite %llu/%llu\n", i + 1, (unsigned long long)bad,
                        (unsigned long long)latFloats);
            return 2;
        }

        float dt = sch.sigmas[i + 1] - sigma;
        double sx = 0;
        static std::vector<float> resv;
        resv.resize(latFloats);
        for (uint64_t k = 0; k < latFloats; k++) resv[k] = epsU[k] + cfg * (epsC[k] - epsU[k]);
        // CFG rescale ("mimic scale"): the guided output's std is clamped back toward the
        // uncond output's std before x0 is formed. Zero-uncond / high-cfg v-pred runs
        // otherwise land at |x0| ~2-3x the real latent std and the decode burns.
        double se = 0;
        if (rescale > 0.0f) {
            double m = 0, s2 = 0, u2 = 0;
            for (uint64_t k = 0; k < latFloats; k++) m += resv[k];
            m /= (double)latFloats;
            for (uint64_t k = 0; k < latFloats; k++) {
                double d = resv[k] - m;
                s2 += d * d;
            }
            for (uint64_t k = 0; k < latFloats; k++) u2 += (double)epsU[k] * epsU[k];
            double su = std::sqrt(u2 / (double)latFloats);
            double sg = std::sqrt(s2 / (double)latFloats) + 1e-6;
            for (uint64_t k = 0; k < latFloats; k++) {
                float er = (float)((resv[k] - m) * (su / sg) + m);
                resv[k] = (float)(rescale * er + (1.0 - rescale) * resv[k]);
            }
        }
        for (uint64_t k = 0; k < latFloats; k++) se += (double)resv[k] * resv[k];
        static std::vector<float> x0v;
        x0v.resize(latFloats);
        // Imagen/SD2.x dynamic thresholding: v-prediction checkpoints routinely return x0
        // estimates with ~2x the latent std of real images, which the sampler then feeds back
        // into itself and the decode burns. Clamping to the 99.5 percentile and renormalising
        // keeps every step's x0 inside the distribution the VAE was trained on.
        for (uint64_t k = 0; k < latFloats; k++) {
            float x = lat[k];  // unscaled latent (step() receives the unscaled sample)
            float e = resv[k];
            float x0;
            if (vpred) {
                float s2p1 = sigma * sigma + 1.0f;
                x0 = e * (-sigma / std::sqrt(s2p1)) + x / s2p1;
            } else {
                x0 = x - sigma * e;  // identical to (sample - sigma*eps) in the x0 form
            }
            x0v[k] = x0;
        }
        float thr = 1.0f;
        if (dynthresh) {
            static std::vector<float> ab;
            ab.resize(latFloats);
            for (uint64_t k = 0; k < latFloats; k++) ab[k] = std::abs(x0v[k]);
            size_t q = (size_t)((double)(latFloats - 1) * 0.995);
            std::nth_element(ab.begin(), ab.begin() + q, ab.end());
            thr = std::max(ab[q], 1.0f);
        }
        for (uint64_t k = 0; k < latFloats; k++) {
            float x = lat[k];
            float x0 = x0v[k];
            if (dynthresh) {
                x0 = std::min(std::max(x0, -thr), thr) / thr;
                x0v[k] = x0;
            }
            float prev = x + (x - x0) / sigma * dt;
            lat[k] = prev;
            sx += (double)prev * prev;
        }
        double sx0 = 0;
        for (uint64_t k = 0; k < latFloats; k++) sx0 += (double)x0v[k] * x0v[k];
        std::printf("step %2d  t=%6.1f  sigma %.5f  |x| %.4f  |x0| %.4f  |e| %.4f  thr %.3f  unet %.0f ms\n", i + 1, t,
                    sigma, std::sqrt(sx / latFloats), std::sqrt(sx0 / latFloats), std::sqrt(se / latFloats), thr,
                    std::chrono::duration<double, std::milli>(tb - ta).count());
    }
    G.eng.profileDump();

    {
        FILE* f = std::fopen(latentOut.c_str(), "wb");
        const char magic[4] = {'O', 'F', 'C', 'L'};
        std::fwrite(magic, 1, 4, f);
        uint32_t dd[3] = {4, latentDim, latentDim};
        std::fwrite(dd, 4, 3, f);
        std::fwrite(lat.data(), 4, lat.size(), f);
        std::fclose(f);
        std::printf("latent saved: %s\n", latentOut.c_str());
    }
    if (noVae) {
        gShutdown();
        return 0;
    }

    // ---------------- phase C: VAE decode
    // the UNet param pool (~2.8 GB of Q8 weights) is dead at this point: shrink the arena so the
    // 512x512 decoder gets a step arena that holds its activations
    clearWeights();
    G.paramArena.init(G.eng.device(), G.eng.physDevice(), 640ull << 20, true, false);
    loadStWeights(stVAE, "");
    uint64_t vStepMB = 600 + (uint64_t)latentDim * latentDim * 2600 / 4096;
    G.eng.initArenas(vStepMB << 20, 220ull << 20);
    gResizeScratch(220ull << 20);
    gCol = G.scratchArena.alloc(kColFloats * 4);
    gSCap = 0;
    auto vaeT0 = std::chrono::steady_clock::now();

    G.eng.beginStep();
    uint32_t H = latentDim, Wd = latentDim;
    std::vector<float> latScaled(latFloats);
    for (uint64_t k = 0; k < latFloats; k++) latScaled[k] = lat[k] / 0.13025f;
    BufRef x = allocN(latFloats);
    G.eng.uploadNow(x, latScaled.data(), latScaled.size() * 4);

    x = conv1(x, "post_quant_conv", 4, 4, H * Wd);
    x = conv3Raw(x, "decoder.conv_in.weight", 4, 512, H, Wd);
    x = vaeResnet(x, "decoder.mid_block.resnets.0", 512, 512, H, Wd, false);
    x = vaeAttn(x, "decoder.mid_block.attentions.0", 512, H, Wd);
    x = vaeResnet(x, "decoder.mid_block.resnets.1", 512, 512, H, Wd, false);
    char pfx[96];
    for (int b = 0; b < 4; b++) {
        uint32_t ci = (b <= 1) ? 512 : (b == 2 ? 512 : 256);
        uint32_t co = (b <= 1) ? 512 : (b == 2 ? 256 : 128);
        for (int i = 0; i < 3; i++) {
            std::snprintf(pfx, sizeof pfx, "decoder.up_blocks.%d.resnets.%d", b, i);
            x = vaeResnet(x, pfx, ci, co, H, Wd, i == 0 && ci != co);
        }
        if (b < 3) {
            std::snprintf(pfx, sizeof pfx, "decoder.up_blocks.%d.upsamplers.0.conv", b);
            uint32_t C = (b == 2) ? 256 : 512;
            x = vaeUpConv(x, pfx, C, H, Wd);
        }
    }
    x = gn(x, "decoder.conv_norm_out", 128, H * Wd, 32, 1e-6f);
    x = silu(x, (uint64_t)128 * H * Wd);
    x = conv3Raw(x, "decoder.conv_out.weight", 128, 3, H, Wd);
    G.eng.endStep();
    G.eng.waitStep();
    std::vector<float> img((size_t)3 * H * Wd);
    G.eng.downloadNow(x, img.data(), img.size() * 4);
    std::printf("vae decode %ux%u: %.0f ms (step arena %llu MB)\n", Wd, H,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - vaeT0).count(),
                (unsigned long long)vStepMB);
    writePPM(out, img, H, Wd);
    std::printf("vae decode -> %s (%ux%u)\n", out.c_str(), Wd, H);
    gShutdown();
    return 0;
}
