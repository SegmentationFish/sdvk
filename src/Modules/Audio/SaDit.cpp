// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
#include "SaDit.hpp"

#include "Modules/GPU/gpu_ops.hpp"
#include "Modules/GPU/vk_engine.hpp"
#include "Modules/Model/Gguf.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>
#include <vector>

namespace ofc {
namespace audio {

namespace {
using gpu::BufRef;
using gpu::G;

constexpr uint64_t align256(uint64_t n) { return (n + 255) & ~255ull; }
constexpr int kIn = 128;      // latent channels
constexpr int kCond = 2048;   // condition width
constexpr int kConcat = 2 * kIn + kCond;
constexpr int kDim = 2048;
constexpr int kHeads = 32;
constexpr int kHd = 64;
constexpr int kQkv = 3 * kDim;
constexpr int kRot = 32;
constexpr int kFf = 16384;
constexpr int kFfHalf = 8192;

struct PushGemm {
    uint32_t M, N, K, mode, sA, sB, sC, batch, lda, ldb, ldc, pad;
};

void gemmQ8(const BufRef& a, const BufRef& wq, const BufRef& c, const BufRef& ws, uint32_t M, uint32_t N,
            uint32_t K) {
    PushGemm pc{M, N, K, 0, 0, 0, 0, 1, 0, 0, 0, 0};
    if (M >= 1024)
        G.eng.dispatch("gemm_nt_q8_v_bm128", {a, wq, c, ws}, &pc, (N + 63) / 64, (M + 127) / 128, 1);
    else
        G.eng.dispatch("gemm_nt_q8_v", {a, wq, c, ws}, &pc, (N + 63) / 64, (M + 63) / 64, 1);
}

void gemmF32(const BufRef& a, const BufRef& b, const BufRef& c, uint32_t M, uint32_t N, uint32_t K,
             uint32_t batch = 1, uint32_t sA = 0, uint32_t sB = 0, uint32_t sC = 0, uint32_t lda = 0,
             uint32_t ldb = 0, uint32_t ldc = 0, bool nt = true) {
    PushGemm pc{M, N, K, 0, sA, sB, sC, batch, lda, ldb, ldc, 0};
    G.eng.dispatch(nt ? "gemm_nt" : "gemm_nn", {a, b, c}, &pc, (N + 63) / 64, (M + 63) / 64, batch);
}

void addInPlace(const BufRef& a, const BufRef& b, uint64_t n) {
    struct {
        uint32_t n, pad;
    } pc{(uint32_t)n, 0};
    G.eng.dispatch("add2", {a, a, b}, &pc, (uint32_t)((n + 255) / 256), 1, 1);
}

}  // namespace

struct SaDit::Impl {
    GgufFile gguf;
    int layers = 0;
    uint64_t paramBytes = 0;

    struct Q8 {
        BufRef q, s;
        int rows = 0, cols = 0;
    };
    struct Layer {
        Q8 qkv, out, ffIn, ffOut;
        BufRef norm1g, norm1b, norm2g, norm2b, ffInBias, ffOutBias;
    };
    std::vector<Layer> block;
    Q8 pre, projIn, projOut, post;
    BufRef condProjW, condProjB;  // unused on the GPU (host path keeps its own copy)
    std::vector<float> timeW, timeB1, timeW1, timeW2, timeB2, fourierW;
    std::vector<float> condW, condBias, mixLogits, mixScale;
    int condFrames = 0;

    // host copies used by the (small) conditioning paths
    static std::vector<float> readF(const GgufFile& g, const std::string& n) {
        std::string err;
        std::vector<float> v = g.readFloat(n, &err);
        if (!err.empty()) {
            std::fprintf(stderr, "dit: %s: %s\n", n.c_str(), err.c_str());
            std::abort();
        }
        return v;
    }

    // Loads any 2-D weight as Q8_0: the quantized GGUFs store most matrices as Q4_K/Q6_K (which
    // readQ8 already re-encodes), but a few small ones stay F16 and need a host quantize.
    Q8 loadQ8(const std::string& name) {
        Q8 w;
        std::string err;
        std::vector<int8_t> q;
        std::vector<float> sc;
        const GgufTensor* t = gguf.find(name);
        if (!t) {
            std::fprintf(stderr, "dit: missing tensor %s\n", name.c_str());
            std::abort();
        }
        auto sh = t->shape();
        w.rows = (int)sh.first;
        w.cols = (int)sh.second;
        uint64_t ne = (uint64_t)w.rows * w.cols;
        if (t->type == (GgmlType)8 /* Q8_0 */) {
            if (!gguf.readQ8(name, q, sc, &err)) {
                std::fprintf(stderr, "dit: readQ8 %s: %s\n", name.c_str(), err.c_str());
                std::abort();
            }
        } else {
            std::vector<float> f = readF(gguf, name);
            if (f.size() != ne || ne % 32 != 0) {
                std::fprintf(stderr, "dit: %s has %zu values (%llu expected)\n", name.c_str(), f.size(),
                             (unsigned long long)ne);
                std::abort();
            }
            q.resize(ne);
            sc.resize(ne / 32);
            ofc::quantizeQ8_0(f.data(), q.data(), sc.data(), ne);
        }
        if ((uint64_t)q.size() != (uint64_t)w.rows * w.cols) {
            std::fprintf(stderr, "dit: %s q8 size %zu vs %d*%d\n", name.c_str(), q.size(), w.rows, w.cols);
            std::abort();
        }
        w.q = G.paramArena.alloc(align256(q.size()));
        w.s = G.paramArena.alloc(align256(sc.size() * 4));
        G.eng.uploadNow(w.q, q.data(), q.size());
        G.eng.uploadNow(w.s, sc.data(), sc.size() * 4);
        return w;
    }

    BufRef loadV(const std::string& name) {
        std::vector<float> v = readF(gguf, name);
        BufRef d = G.paramArena.alloc(align256(v.size() * 4));
        G.eng.uploadNow(d, v.data(), v.size() * 4);
        return d;
    }

    // ---- small host ops for the timestep / condition encoders
    std::vector<float> timestepEmbed(float t) const {
        const int F = 128;
        std::vector<float> f(2 * F);
        for (int i = 0; i < F; i++) {
            float a = 2.0f * 3.14159265358979323846f * t * fourierW[(size_t)i];
            f[(size_t)i] = std::cos(a);
            f[(size_t)F + i] = std::sin(a);
        }
        // linear_1 (256 -> 2048) + SiLU
        std::vector<float> h(kDim, 0.0f);
        for (int o = 0; o < kDim; o++) {
            const float* row = &timeW[(size_t)o * 2 * F];
            float acc = timeB1[(size_t)o];
            for (int i = 0; i < 2 * F; i++) acc += row[i] * f[(size_t)i];
            h[(size_t)o] = acc / (1.0f + std::exp(-acc));
        }
        std::vector<float> out(kDim, 0.0f);
        for (int o = 0; o < kDim; o++) {
            const float* row = &timeW2[(size_t)o * kDim];
            float acc = timeB2[(size_t)o];
            for (int i = 0; i < kDim; i++) acc += row[i] * h[(size_t)i];
            out[(size_t)o] = acc;
        }
        return out;
    }
};

// Unit check for the GEMMs the DiT relies on: C[M,N] = A[M,K] * W[N,K]^T in fp32 and Q8.
void saSelfTestGemm(int M, int N, int K, float& e32, float& e8) {
    std::vector<float> a((size_t)M * K), w((size_t)N * K);
    for (size_t i = 0; i < a.size(); i++) a[i] = std::sin((float)i * 0.001f) * 0.5f;
    for (size_t i = 0; i < w.size(); i++) w[i] = std::cos((float)i * 0.0013f) * 0.5f;
    std::vector<float> ref((size_t)M * N, 0.0f);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double acc = 0;
            for (int k = 0; k < K; k++) acc += (double)a[(size_t)m * K + k] * w[(size_t)n * K + k];
            ref[(size_t)m * N + n] = (float)acc;
        }
    G.eng.beginStep();
    BufRef A = G.eng.stepArena().alloc(align256(a.size() * 4));
    BufRef W = G.eng.stepArena().alloc(align256(w.size() * 4));
    BufRef C = G.eng.stepArena().alloc(align256(ref.size() * 4));
    G.eng.uploadNow(A, a.data(), a.size() * 4);
    G.eng.uploadNow(W, w.data(), w.size() * 4);
    gemmF32(A, W, C, (uint32_t)M, (uint32_t)N, (uint32_t)K);
    BufRef Wq = G.eng.stepArena().alloc(align256(w.size()));
    BufRef Ws = G.eng.stepArena().alloc(align256((w.size() / 32) * 4));
    std::vector<int8_t> q(w.size());
    std::vector<float> sc(w.size() / 32);
    ofc::quantizeQ8_0(w.data(), q.data(), sc.data(), w.size());
    BufRef C8 = G.eng.stepArena().alloc(align256(ref.size() * 4));
    G.eng.uploadNow(Wq, q.data(), q.size());
    G.eng.uploadNow(Ws, sc.data(), sc.size() * 4);
    gemmQ8(A, Wq, C8, Ws, (uint32_t)M, (uint32_t)N, (uint32_t)K);
    G.eng.endStep();
    G.eng.waitStep();
    std::vector<float> o32(ref.size()), o8(ref.size());
    G.eng.downloadNow(C, o32.data(), o32.size() * 4);
    G.eng.downloadNow(C8, o8.data(), o8.size() * 4);
    auto rel = [&](const std::vector<float>& o) {
        double num = 0, den = 0;
        for (size_t i = 0; i < ref.size(); i++) {
            num += (double)(o[i] - ref[i]) * (o[i] - ref[i]);
            den += (double)ref[i] * ref[i];
        }
        return (float)std::sqrt(num / std::max(1e-30, den));
    };
    e32 = rel(o32);
    e8 = rel(o8);
}

int SaDit::layers() const { return impl_ ? impl_->layers : 0; }

SaDit::~SaDit() { delete impl_; }

bool SaDit::load(const std::string& ggufPath, const std::string& spvDir, int maxLayers, std::string& err) {
    auto* m = new Impl();
    impl_ = m;
    std::string e;
    if (!m->gguf.open(ggufPath, &e)) {
        err = "cannot open DiT GGUF " + ggufPath + ": " + e;
        return false;
    }
    // layer count from the tensor names
    int nLayers = 0;
    for (const auto& t : m->gguf.tensors()) {
        const std::string key = "diffusion_transformer.transformer.layers.";
        if (t.name.rfind(key, 0) != 0) continue;
        int idx = std::atoi(t.name.c_str() + key.size());
        nLayers = std::max(nLayers, idx + 1);
    }
    if (maxLayers > 0) nLayers = std::min(nLayers, maxLayers);
    m->layers = nLayers;
    if (!nLayers) {
        err = "DiT GGUF has no diffusion_transformer.transformer.layers.* tensors";
        return false;
    }

    // arena: Q8 payload (1 byte + 4/32 scale) per 2-D weight + fp32 vectors, with a safety margin
    uint64_t bytes = 0;
    for (const auto& t : m->gguf.tensors()) {
        auto sh = t.shape();
        uint64_t ne = sh.first * sh.second;
        bool isLayer = t.name.rfind("diffusion_transformer.transformer.layers.", 0) == 0;
        if (isLayer) {
            int idx = std::atoi(t.name.c_str() + 42);
            if (idx >= nLayers) continue;
        }
        if (t.dims.size() >= 2 && ne >= 1024)
            bytes += ne + (ne / 32) * 4;
        else
            bytes += ne * 4;
    }
    m->paramBytes = bytes + (256ull << 20);

    if (!G.eng.alive()) {
        G.allocGradArena = false;
        G.stepBytes = 64ull << 20;
        G.scratchBytes = 64ull << 20;
        G.paramBytes = 64ull << 20;
        ofc::gpu::gInit(spvDir);
    }
    G.eng.initArenas(768ull << 20, 256ull << 20);
    G.paramArena.destroy();
    G.paramArena.init(G.eng.device(), G.eng.physDevice(), m->paramBytes, true, false);

    auto P = [](int i, const char* rest) {
        return "diffusion_transformer.transformer.layers." + std::to_string(i) + "." + rest;
    };
    m->pre = m->loadQ8("diffusion_transformer.preprocess_conv.weight");
    m->post = m->loadQ8("diffusion_transformer.postprocess_conv.weight");
    m->projIn = m->loadQ8("diffusion_transformer.transformer.project_in.weight");
    m->projOut = m->loadQ8("diffusion_transformer.transformer.project_out.weight");

    m->fourierW = Impl::readF(m->gguf, "diffusion_transformer.timestep_features.weight");
    m->timeW = Impl::readF(m->gguf, "diffusion_transformer.to_timestep_embed.0.weight");
    m->timeB1 = Impl::readF(m->gguf, "diffusion_transformer.to_timestep_embed.0.bias");
    m->timeW2 = Impl::readF(m->gguf, "diffusion_transformer.to_timestep_embed.2.weight");
    m->timeB2 = Impl::readF(m->gguf, "diffusion_transformer.to_timestep_embed.2.bias");

    m->condW = Impl::readF(m->gguf, "latent_conditioners.0.weight");
    m->condBias = Impl::readF(m->gguf, "latent_conditioners.0.bias");
    m->mixLogits = Impl::readF(m->gguf, "cond_layer_logits");
    m->mixScale = Impl::readF(m->gguf, "cond_layer_scale");

    m->block.resize((size_t)nLayers);
    for (int i = 0; i < nLayers; i++) {
        auto& b = m->block[(size_t)i];
        b.qkv = m->loadQ8(P(i, "self_attn.to_qkv.weight"));
        b.out = m->loadQ8(P(i, "self_attn.to_out.weight"));
        b.ffIn = m->loadQ8(P(i, "ff.ff.0.proj.weight"));
        b.ffOut = m->loadQ8(P(i, "ff.ff.2.weight"));
        b.norm1g = m->loadV(P(i, "pre_norm.gamma"));
        b.norm1b = m->loadV(P(i, "pre_norm.beta"));
        b.norm2g = m->loadV(P(i, "ff_norm.gamma"));
        b.norm2b = m->loadV(P(i, "ff_norm.beta"));
        b.ffInBias = m->loadV(P(i, "ff.ff.0.proj.bias"));
        b.ffOutBias = m->loadV(P(i, "ff.ff.2.bias"));
    }
    std::printf("dit: %d blocks, arena %.0f MB (q8 weights)\n", nLayers, m->paramBytes / 1048576.0);
    return true;
}

bool SaDit::encodeCond(const std::vector<float>& states, int frames, std::vector<float>& cond,
                       int& latentLen, std::string& err) {
    auto* m = impl_;
    if (!m) {
        err = "dit: not loaded";
        return false;
    }
    const int L = (int)m->mixLogits.size();
    if ((int)states.size() != frames * L * 4096) {
        err = "dit: condition input size mismatch";
        return false;
    }
    // softmax mix over the 8 per-frame hidden states, times the learned scale
    std::vector<float> w((size_t)L);
    float mx = -1e30f;
    for (float v : m->mixLogits) mx = std::max(mx, v);
    float sum = 0;
    for (int l = 0; l < L; l++) {
        w[(size_t)l] = std::exp(m->mixLogits[(size_t)l] - mx);
        sum += w[(size_t)l];
    }
    for (int l = 0; l < L; l++) w[(size_t)l] = w[(size_t)l] / sum * m->mixScale[0];

    std::vector<float> mixed((size_t)frames * 4096, 0.0f);
    for (int t = 0; t < frames; t++) {
        for (int l = 0; l < L; l++) {
            const float* src = &states[(size_t)t * L * 4096 + (size_t)l * 4096];
            float wl = w[(size_t)l];
            float* dst = &mixed[(size_t)t * 4096];
            for (int i = 0; i < 4096; i++) dst[i] += wl * src[i];
        }
    }

    // conv1d 4096 -> 2048, k=3, pad=1 (weights are [out][in][k])
    std::vector<float> y((size_t)frames * 2048, 0.0f);
    for (int t = 0; t < frames; t++) {
        for (int o = 0; o < 2048; o++) {
            float acc = m->condBias[(size_t)o];
            const float* row = &m->condW[(size_t)o * 4096 * 3];
            for (int k = 0; k < 3; k++) {
                int tt = t + k - 1;
                if (tt < 0 || tt >= frames) continue;
                const float* in = &mixed[(size_t)tt * 4096];
                const float* wk = row + (size_t)k;
                for (int i = 0; i < 4096; i++) acc += wk[(size_t)i * 3] * in[i];
            }
            y[(size_t)t * 2048 + o] = acc;
        }
    }

    latentLen = std::max(1, (int)((double)frames * 44100.0 / 24000.0 * 960.0 / 512.0));
    cond.assign((size_t)latentLen * 2048, 0.0f);
    for (int i = 0; i < latentLen; i++) {
        int src = std::min(frames - 1, (int)((double)i * frames / latentLen));
        std::memcpy(&cond[(size_t)i * 2048], &y[(size_t)src * 2048], sizeof(float) * 2048);
    }
    return true;
}

bool SaDit::forward(const std::vector<float>& x, int T, float t, const std::vector<float>& cond,
                    std::vector<float>& vel, std::string& err) {
    auto* m = impl_;
    if (!m) {
        err = "dit: not loaded";
        return false;
    }
    if ((int)x.size() != kIn * T || (int)cond.size() != T * kCond) {
        err = "dit: input size mismatch";
        return false;
    }
    const int S = T + 1;
    const char* dumpDir = std::getenv("OFC_DIT_DUMP");

    G.eng.beginStep();
    auto alloc = [&](uint64_t floats) { return G.eng.stepArena().alloc(align256(floats * 4)); };
    // downloads only work after waitStep, so dumps are queued and written at the end
    std::vector<std::tuple<BufRef, int, int, int>> pending;
    auto dump = [&](const BufRef& b, int rows, int cols, int idx) {
        if (!dumpDir) return;
        // snapshot inside the graph: the activation buffer is reused by later blocks
        BufRef snap = alloc((uint64_t)rows * cols);
        G.eng.copyBuf(snap, BufRef{b.buf, b.offset, (uint64_t)rows * cols * 4});
        pending.emplace_back(snap, rows, cols, idx);
    };

    // ---- concat input [T, 2304] = [latent^T | zeros | cond]
    std::vector<float> inT((size_t)T * kConcat, 0.0f);
    for (int t2 = 0; t2 < T; t2++) {
        float* row = &inT[(size_t)t2 * kConcat];
        for (int c = 0; c < kIn; c++) row[c] = x[(size_t)c * T + t2];
        std::memcpy(row + 2 * kIn, &cond[(size_t)t2 * kCond], sizeof(float) * kCond);
    }
    BufRef inB = alloc((uint64_t)T * kConcat);
    G.eng.uploadNow(inB, inT.data(), inT.size() * 4);

    // ---- preprocess conv (k=1) with residual, then proj_in
    BufRef pre = alloc((uint64_t)T * kConcat);
    gemmQ8(inB, m->pre.q, pre, m->pre.s, (uint32_t)T, (uint32_t)kConcat, (uint32_t)kConcat);
    addInPlace(pre, inB, (uint64_t)T * kConcat);

    BufRef h = alloc((uint64_t)S * kDim);
    {
        BufRef hd = BufRef{h.buf, h.offset + (uint64_t)kDim * 4, h.size - (uint64_t)kDim * 4};  // row 0 reserved
        gemmQ8(pre, m->projIn.q, hd, m->projIn.s, (uint32_t)T, (uint32_t)kDim, (uint32_t)kConcat);
    }
    // debug: replace the whole input path with a dumped [S,2048] state (isolates the blocks)
    if (const char* preFile = std::getenv("OFC_DIT_PRE")) {
        std::vector<float> hv((size_t)S * kDim);
        if (FILE* f = std::fopen(preFile, "rb")) {
            if (std::fread(hv.data(), 4, hv.size(), f) != hv.size()) std::fprintf(stderr, "short pre file\n");
            std::fclose(f);
            G.eng.uploadNow(h, hv.data(), hv.size() * 4);
        }
    }
    // ---- timestep token in row 0
    {
        std::vector<float> te = m->timestepEmbed(t);
        G.eng.uploadNow(BufRef{h.buf, h.offset, (uint64_t)kDim * 4}, te.data(), te.size() * 4);
    }

    // ---- rotary tables [S, kRot/2]
    {
        std::vector<float> cs((size_t)S * (kRot / 2)), sn((size_t)S * (kRot / 2));
        for (int s = 0; s < S; s++) {
            for (int p = 0; p < kRot / 2; p++) {
                float inv = 1.0f / std::pow(10000.0f, (float)(2 * p) / (float)kRot);
                float a = (float)s * inv;
                cs[(size_t)s * (kRot / 2) + p] = std::cos(a);
                sn[(size_t)s * (kRot / 2) + p] = std::sin(a);
            }
        }
        BufRef cb = G.eng.scratchArena().alloc(align256(cs.size() * 4));
        BufRef sb = G.eng.scratchArena().alloc(align256(sn.size() * 4));
        G.eng.uploadNow(cb, cs.data(), cs.size() * 4);
        G.eng.uploadNow(sb, sn.data(), sn.size() * 4);

        BufRef norm = alloc((uint64_t)S * kDim);
        BufRef qkv = alloc((uint64_t)S * kQkv);
        BufRef scores = alloc((uint64_t)kHeads * S * S);
        BufRef probs = alloc((uint64_t)kHeads * S * S);
        BufRef attn = alloc((uint64_t)S * kDim);
        BufRef tmp = alloc((uint64_t)S * kDim);
        BufRef ff = alloc((uint64_t)S * kFf);
        BufRef gate = alloc((uint64_t)S * kFfHalf);

        if (dumpDir) dump(h, S, kDim, 90);  // state entering block 0
        for (int l = 0; l < m->layers; l++) {
            auto& b = m->block[(size_t)l];
            const bool probe = dumpDir && l == 0;
            // norm1
            {
                struct {
                    uint32_t rows, d;
                    float eps;
                    uint32_t affine;
                } pc{(uint32_t)S, (uint32_t)kDim, 1e-5f, 1};
                G.eng.dispatch("layernorm_fwd", {h, b.norm1g, b.norm1b, norm}, &pc, (uint32_t)S, 1, 1);
            }
            if (probe) dump(norm, S, kDim, 93);
            gemmQ8(norm, b.qkv.q, qkv, b.qkv.s, (uint32_t)S, (uint32_t)kQkv, (uint32_t)kDim);

            // rope on q (base 0) and k (base 2048)
            for (int half = 0; half < 2 && !std::getenv("OFC_DIT_NOROPE"); half++) {
                struct {
                    uint32_t seq, heads, hd, rot, rowStride, base;
                } pc{(uint32_t)S, (uint32_t)kHeads, (uint32_t)kHd, (uint32_t)kRot, (uint32_t)kQkv,
                     (uint32_t)(half * kDim)};
                uint32_t n = (uint32_t)(S * kHeads * (kRot / 2));
                G.eng.dispatch("sa_rope", {qkv, cb, sb}, &pc, (n + 255) / 256, 1, 1);
            }
            if (probe) dump(qkv, S, kQkv, 98);  // post-rope qkv
            // scores[h] = q[h] k[h]^T / sqrt(hd)
            {
                BufRef qs = qkv;
                BufRef ks = BufRef{qkv.buf, qkv.offset + (uint64_t)kDim * 4, qkv.size - (uint64_t)kDim * 4};
                gemmF32(qs, ks, scores, (uint32_t)S, (uint32_t)S, (uint32_t)kHd, (uint32_t)kHeads,
                        (uint32_t)kHd, (uint32_t)kHd, (uint32_t)(S * S), (uint32_t)kQkv, (uint32_t)kQkv,
                        (uint32_t)S, true);
            }
            {
                struct {
                    uint32_t n;
                    float s;
                } pc{(uint32_t)(kHeads * S * S), std::getenv("OFC_DIT_SCALE") ? (float)std::atof(std::getenv("OFC_DIT_SCALE")) : 1.0f / std::sqrt((float)kHd)};
                G.eng.dispatch("sa_scale", {scores, probs}, &pc, (uint32_t)((kHeads * S * S + 255) / 256), 1, 1);
                if (probe) dump(probs, kHeads * S, S, 100);
                struct {
                    uint32_t rows, cols;
                } sp{(uint32_t)(kHeads * S), (uint32_t)S};
                G.eng.dispatch("softmax_fwd", {probs, scores}, &sp, (uint32_t)(kHeads * S), 1, 1);
                if (probe) dump(scores, kHeads * S, S, 101);
            }
            // attn[h] = probs[h] v[h]
            {
                BufRef vs = BufRef{qkv.buf, qkv.offset + (uint64_t)(2 * kDim) * 4,
                                   qkv.size - (uint64_t)(2 * kDim) * 4};
                gemmF32(scores, vs, attn, (uint32_t)S, (uint32_t)kHd, (uint32_t)S, (uint32_t)kHeads,
                        (uint32_t)(S * S), (uint32_t)kHd, (uint32_t)kHd, (uint32_t)S, (uint32_t)kQkv,
                        (uint32_t)kDim, false);
            }
            if (probe) dump(attn, S, kDim, 97);
            gemmQ8(attn, b.out.q, tmp, b.out.s, (uint32_t)S, (uint32_t)kDim, (uint32_t)kDim);
            if (probe) dump(tmp, S, kDim, 91);
            addInPlace(h, tmp, (uint64_t)S * kDim);
            // norm2 -> ff_in -> swiglu -> ff_out
            {
                struct {
                    uint32_t rows, d;
                    float eps;
                    uint32_t affine;
                } pc{(uint32_t)S, (uint32_t)kDim, 1e-5f, 1};
                G.eng.dispatch("layernorm_fwd", {h, b.norm2g, b.norm2b, norm}, &pc, (uint32_t)S, 1, 1);
            }
            gemmQ8(norm, b.ffIn.q, ff, b.ffIn.s, (uint32_t)S, (uint32_t)kFf, (uint32_t)kDim);
            {
                struct {
                    uint32_t rows, cols;
                } pc{(uint32_t)S, (uint32_t)kFf};
                G.eng.dispatch("sa_bias_col", {ff, b.ffInBias}, &pc, (uint32_t)(((uint64_t)S * kFf + 255) / 256),
                               1, 1);
            }
            {
                struct {
                    uint32_t rows, half_;
                } pc{(uint32_t)S, (uint32_t)kFfHalf};
                G.eng.dispatch("sa_swiglu", {ff, gate}, &pc, (uint32_t)(((uint64_t)S * kFfHalf + 255) / 256), 1,
                               1);
            }
            gemmQ8(gate, b.ffOut.q, tmp, b.ffOut.s, (uint32_t)S, (uint32_t)kDim, (uint32_t)kFfHalf);
            {
                struct {
                    uint32_t rows, cols;
                } pc{(uint32_t)S, (uint32_t)kDim};
                G.eng.dispatch("sa_bias_col", {tmp, b.ffOutBias}, &pc, (uint32_t)(((uint64_t)S * kDim + 255) / 256),
                               1, 1);
            }
            if (probe) dump(tmp, S, kDim, 92);
            addInPlace(h, tmp, (uint64_t)S * kDim);
            if (dumpDir) dump(h, S, kDim, l);
        }

        // ---- drop the timestep token, proj_out, postprocess conv + residual, transpose
        BufRef body = BufRef{h.buf, h.offset + (uint64_t)kDim * 4, h.size - (uint64_t)kDim * 4};
        BufRef outT = alloc((uint64_t)T * kIn);
        gemmQ8(body, m->projOut.q, outT, m->projOut.s, (uint32_t)T, (uint32_t)kIn, (uint32_t)kDim);
        BufRef post = alloc((uint64_t)T * kIn);
        if (dumpDir) dump(outT, T, kIn, 102);
        gemmQ8(outT, m->post.q, post, m->post.s, (uint32_t)T, (uint32_t)kIn, (uint32_t)kIn);
        if (dumpDir) dump(post, T, kIn, 103);
        addInPlace(post, outT, (uint64_t)T * kIn);
        BufRef outB = alloc((uint64_t)kIn * T);
        {
            struct {
                uint32_t rows, cols;
            } pc{(uint32_t)T, (uint32_t)kIn};
            G.eng.dispatch("transpose", {post, outB}, &pc, (uint32_t)(((uint64_t)T * kIn + 255) / 256), 1, 1);
        }
        G.eng.endStep();
        G.eng.waitStep();
        for (auto& [buf, rows, cols, idx] : pending) {
            std::vector<float> host((size_t)rows * cols);
            G.eng.downloadNow(buf, host.data(), host.size() * 4);
            char path[512];
            std::snprintf(path, sizeof path, "%s_blk%d.f32", dumpDir, idx);
            if (FILE* f = std::fopen(path, "wb")) {
                std::fwrite(host.data(), 4, host.size(), f);
                std::fclose(f);
            }
        }
        vel.assign((size_t)kIn * T, 0.0f);
        G.eng.downloadNow(outB, vel.data(), vel.size() * 4);
    }
    return true;
}

}  // namespace audio
}  // namespace ofc
