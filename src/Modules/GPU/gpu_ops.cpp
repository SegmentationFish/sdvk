// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
#include "gpu_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace ofc {
namespace gpu {

GContext G;
std::unordered_map<std::string, uint64_t> gCopyTags;

// ---------------------------------------------------------------- helpers

static uint32_t CEIL(uint64_t a, uint32_t b) { return (uint32_t)((a + b - 1) / b); }

uint64_t gNumelOf(const Shape& s) {
    uint64_t n = 1;
    for (size_t v : s) n *= v;
    return n;
}
static uint64_t gNumel(const Shape& s) { return gNumelOf(s); }

// last dim first (matches shader convention d0 = last dim), padded with 1s
static void shapeDims4(const Shape& s, uint32_t d[4]) {
    for (int i = 0; i < 4; i++) d[i] = 1;
    size_t r = s.size();
    for (size_t i = 0; i < r && i < 4; i++) d[i] = (uint32_t)s[r - 1 - i];
}

// contiguous strides in elements, last dim first, padded with 0
static void contigStrides4(const Shape& s, uint32_t st[4]) {
    for (int i = 0; i < 4; i++) st[i] = 0;
    size_t r = s.size();
    uint32_t acc = 1;
    for (size_t i = 0; i < r && i < 4; i++) {
        st[i] = acc;
        acc *= (uint32_t)s[r - 1 - i];
    }
}

static void broadcastShape(const Shape& a, const Shape& b, Shape& out) {
    size_t r = std::max(a.size(), b.size());
    out.assign(r, 1);
    for (size_t i = 0; i < r; i++) {
        size_t av = a.size() >= r - i ? a[a.size() - (r - i)] : 1;
        size_t bv = b.size() >= r - i ? b[b.size() - (r - i)] : 1;
        size_t v = std::max(av, bv);
        if (av != v && av != 1) throw std::runtime_error("gpu broadcast mismatch");
        if (bv != v && bv != 1) throw std::runtime_error("gpu broadcast mismatch");
        out[i] = v;
    }
}

// strides (last dim first) for reading `real` inside coordinates of `out`
static void bcastStrides4(const Shape& real, const Shape& out, uint32_t st[4]) {
    uint32_t realC[4];
    Shape padded = out;  // same rank, dims replaced by real where present
    size_t r = out.size();
    Shape rr(r, 1);
    for (size_t i = 0; i < r; i++) {
        size_t ri = real.size() >= r - i ? real[real.size() - (r - i)] : 1;
        rr[i] = ri;
    }
    contigStrides4(rr, realC);
    for (size_t i = 0; i < r && i < 4; i++) {
        uint32_t dim = (uint32_t)rr[r - 1 - i];
        st[i] = (dim == 1) ? 0 : realC[i];
    }
    for (size_t i = r; i < 4; i++) st[i] = 0;
}

static BufRef stepAlloc(uint64_t bytes) {
    if (bytes > (8ull << 20) && std::getenv("OFC_ALLOC_LOG"))
        std::fprintf(stderr, "[alloc] %.2f MB (step used %.0f/%.0f MB)\n", bytes / 1e6, G.eng.stepArena().used() / 1e6,
                     G.eng.stepArena().cap() / 1e6);
    return G.eng.stepArena().alloc(bytes);
}

static void zeroBuf(const BufRef& b) { G.eng.fill(b, 0u); }

static GTPtr makeTensor(const Shape& s, bool rg, bool zeroData) {
    gStepTensorCount++;
    auto t = std::make_shared<GTensor>();
    t->shape = s;
    t->requiresGrad = rg;
    uint64_t bytes = gNumel(s) * 4;
    t->data = stepAlloc(bytes);
    if (zeroData && !G.stepPreZeroed) zeroBuf(t->data);
    if (rg) {
        t->grad = stepAlloc(bytes);
        if (!G.stepPreZeroed) zeroBuf(t->grad);
    }
    return t;
}

static GTPtr reg(GTPtr t) {
    G.tape.push_back(t);
    return t;
}

// push-constant structs (must match the GLSL blocks byte for byte)
struct P_gemm {
    uint32_t M, N, K, mode, sA, sB, sC, batch, lda, ldb, ldc, pad;
};
struct P_nd {
    uint32_t n, d0, d1, d2, d3, os0, os1, os2, os3, ss0, ss1, ss2, ss3, mode;
    float scale;
};
struct P_ew2 {
    uint32_t n, d0, d1, d2, d3, as0, as1, as2, as3, bs0, bs1, bs2, bs3, mode;
};
struct P_mg {
    uint32_t n, d0, d1, d2, d3, ts0, ts1, ts2, ts3, os0, os1, os2, os3;
};
struct P_rc {
    uint32_t rows, cols;
};
struct P_bs {
    uint32_t n;
    float s;
};
struct P_un {
    uint32_t n, mode;
};
struct P_rn {
    uint32_t rows, d;
    float eps;
    uint32_t pad;
};
struct P_ce {
    uint32_t rows, v;
};
struct P_ceb {
    uint32_t rows, v;
    float invRows;
    uint32_t pad;
};
struct P_g {
    uint32_t n, d;
};
struct P_rope {
    uint32_t batch, seq, half, inverse;
    float base;
    uint32_t pad;
};
struct P_mask {
    uint32_t n, period;
};
struct P_topk {
    uint32_t rows, g, k;
};
struct P_ohb {
    uint32_t rows, groups;
};
struct P_adamw {
    uint32_t n;
    float lr, b1, b2, eps, wd, invBc1, invBc2;
};
struct P_nsq {
    uint32_t n;
};
struct P_clipf {
    uint32_t count;
    float maxNorm;
};
struct P_cp {
    uint32_t rows, g, gid;
};
struct P_rb {
    uint32_t batch, s, d;
};

// ---------------------------------------------------------------- init

void gInit(const std::string& spvDir) {
    if (G.eng.alive()) gShutdown();
    G.eng.init();
    G.eng.loadKernels(spvDir);
    G.eng.initArenas(G.stepBytes, G.scratchBytes);
    G.paramArena.init(G.eng.device(), G.eng.physDevice(), G.paramBytes, true, false);
    if (G.allocGradArena) G.paramGradArena.init(G.eng.device(), G.eng.physDevice(), G.paramBytes, true, false);
    G.scratchArena.init(G.eng.device(), G.eng.physDevice(), G.scratchBytes, true, false);
    G.hp = G.scratchArena.alloc(64 * 4);
    float z[64] = {0};
    G.eng.uploadNow(G.hp, z, sizeof z);
}

// Re-create the persistent scratch arena when a phase needs a different size.
// BufRefs previously handed out from it (including G.hp) are invalidated, so hp is rebuilt here.
void gResizeScratch(uint64_t bytes) {
    G.scratchBytes = bytes;
    G.scratchArena.init(G.eng.device(), G.eng.physDevice(), bytes, true, false);
    G.hp = G.scratchArena.alloc(64 * 4);
    float z[64] = {0};
    G.eng.uploadNow(G.hp, z, sizeof z);
}

uint64_t gStepTensorCount = 0;
BufRef gHpBuf() { return G.hp; }

void gUploadHp(const float* hp, int n) { G.eng.uploadNow(G.hp, hp, (uint64_t)n * 4); }

void gShutdown() {
    G.eng.destroy();
    G.paramArena.destroy();
    G.paramGradArena.destroy();
    G.scratchArena.destroy();
    gResetTape();
    gStepTensorCount = 0;
    G.stepPreZeroed = false;
}

void gResetTape() {
    for (auto& t : G.tape) t->backward = nullptr;  // break shared_ptr cycles
    G.tape.clear();
}

BufRef gScratchAlloc(uint64_t bytes) { return G.scratchArena.alloc(bytes); }
void gFillZero(const BufRef& b) { zeroBuf(b); }
void gUploadNow(const BufRef& dst, const void* src, uint64_t bytes) { G.eng.uploadNow(dst, src, bytes); }
void gDownloadNow(const BufRef& src, void* dst, uint64_t bytes) { G.eng.downloadNow(src, dst, bytes); }

// ---------------------------------------------------------------- backward

void gBackward(const GTPtr& loss) {
    if (!loss || !loss->requiresGrad) return;
    G.eng.fill(loss->grad, 0x3F800000u);  // = 1.0f
    for (size_t i = G.tape.size(); i-- > 0;) {
        auto& t = G.tape[i];
        if (t->requiresGrad && t->backward) t->backward();
    }
}

// ---------------------------------------------------------------- tensors

GTPtr gEmpty(const Shape& s, bool rg) { return makeTensor(s, rg, false); }
GTPtr gZeros(const Shape& s, bool rg) { return makeTensor(s, rg, true); }

GTPtr gParam(const Shape& s, const char* name) {
    auto t = std::make_shared<GTensor>();
    t->shape = s;
    t->requiresGrad = true;
    t->persistent = true;
    t->name = name;
    uint64_t bytes = gNumel(s) * 4;
    t->data = G.paramArena.alloc(bytes);
    t->grad = G.paramGradArena.alloc(bytes);
    return t;
}

GTPtr gWrapView(const Shape& s, const BufRef& data, const BufRef& grad, bool rg) {
    auto t = std::make_shared<GTensor>();
    t->shape = s;
    t->data = data;
    t->requiresGrad = rg;
    t->grad = grad;
    return t;
}

GTPtr gViewShape(const GTPtr& x, const Shape& s) {
    if (gNumel(s) != gNumel(x->shape)) throw std::runtime_error("gViewShape numel mismatch");
    auto out = std::make_shared<GTensor>();
    out->shape = s;
    out->data = x->data;
    out->requiresGrad = x->requiresGrad;
    if (out->requiresGrad) {
        out->grad = stepAlloc(gNumel(s) * 4);
        zeroBuf(out->grad);
        auto xp = x;
        out->backward = [xp, out]() {
            uint32_t d[4], st[4];
            shapeDims4(out->shape, d);
            contigStrides4(out->shape, st);
            P_nd pc{};
            pc.n = (uint32_t)gNumel(out->shape);
            pc.d0 = d[0];
            pc.d1 = d[1];
            pc.d2 = d[2];
            pc.d3 = d[3];
            pc.os0 = st[0];
            pc.os1 = st[1];
            pc.os2 = st[2];
            pc.os3 = st[3];
            pc.ss0 = st[0];
            pc.ss1 = st[1];
            pc.ss2 = st[2];
            pc.ss3 = st[3];
            pc.mode = 1;
            pc.scale = 1.0f;
            gCopyTags["bwd_view"]++;
            G.eng.dispatch("copy_nd", {xp->grad, out->grad}, &pc, CEIL(pc.n, 256), 1, 1);
        };
    }
    return reg(out);
}

// ---------------------------------------------------------------- gemm

// batch>1: batched on grid.z, A/B/C stride across batches by sA/sB/sC (elements); sB=0 means broadcast
static void gemmCall(const char* kind, const BufRef& A, const BufRef& B, const BufRef& C, uint32_t M, uint32_t N,
                     uint32_t S, uint32_t mode, uint32_t batch = 1, uint32_t sA = 0, uint32_t sB = 0, uint32_t sC = 0,
                     uint32_t lda = 0, uint32_t ldb = 0, uint32_t ldc = 0) {
    P_gemm pc{M, N, S, mode, sA, sB, sC, batch, lda, ldb, ldc, 0};
    G.eng.dispatch(kind, {A, B, C}, &pc, CEIL(N, 64), CEIL(M, 64), batch);
}

GTPtr gMatmul(const GTPtr& a, const GTPtr& b) {
    size_t ra = a->shape.size(), rb = b->shape.size();
    if (ra < 2 || rb < 2) throw std::runtime_error("gMatmul rank");
    uint64_t M = a->shape[ra - 2], K = a->shape[ra - 1];
    uint64_t K2 = b->shape[rb - 2], N = b->shape[rb - 1];
    if (K != K2) throw std::runtime_error("gMatmul inner dim");
    uint64_t numelA = gNumel(a->shape), numelB = gNumel(b->shape);
    uint64_t batchA = numelA / (M * K);
    uint64_t batchB = numelB / (K * N);
    if (batchB != 1 && batchB != batchA) throw std::runtime_error("gMatmul batch");
    bool rg = a->requiresGrad || b->requiresGrad;
    Shape os = a->shape;
    os[ra - 2] = M;
    os[ra - 1] = N;
    auto out = makeTensor(os, rg, false);
    uint32_t sBf = batchB == 1 ? 0u : (uint32_t)(K * N);
    gemmCall("gemm_nn", a->data, b->data, out->data, (uint32_t)M, (uint32_t)N, (uint32_t)K, 0, (uint32_t)batchA,
             (uint32_t)(M * K), sBf, (uint32_t)(M * N));
    if (rg) {
        out->name = "gMatmul";
        uint64_t M_ = M, K_ = K, N_ = N, bA = batchA, bB = batchB;
        auto ap = a, bp = b, op = out;
        out->backward = [ap, bp, op, M_, K_, N_, bA, bB]() {
            if (bA > 1 && bB > 1) {
                if (ap->requiresGrad)
                    gemmCall("gemm_nt", op->grad, bp->data, ap->grad, (uint32_t)M_, (uint32_t)K_, (uint32_t)N_, 1,
                             (uint32_t)bA, (uint32_t)(M_ * N_), (uint32_t)(K_ * N_), (uint32_t)(M_ * K_));
                if (bp->requiresGrad)
                    gemmCall("gemm_tn", ap->data, op->grad, bp->grad, (uint32_t)K_, (uint32_t)N_, (uint32_t)M_, 1,
                             (uint32_t)bA, (uint32_t)(M_ * K_), (uint32_t)(M_ * N_), (uint32_t)(K_ * N_));
                return;
            }
            for (uint64_t bi = 0; bi < bA; bi++) {
                uint64_t bb = (bB == 1 ? 0 : bi);
                BufRef go{op->grad.buf, op->grad.offset + bi * M_ * N_ * 4, M_ * N_ * 4};
                BufRef pab{ap->data.buf, ap->data.offset + bi * M_ * K_ * 4, M_ * K_ * 4};
                BufRef pb{bp->data.buf, bp->data.offset + bb * K_ * N_ * 4, K_ * N_ * 4};
                if (ap->requiresGrad) {
                    BufRef ga{ap->grad.buf, ap->grad.offset + bi * M_ * K_ * 4, M_ * K_ * 4};
                    gemmCall("gemm_nt", go, pb, ga, (uint32_t)M_, (uint32_t)K_, (uint32_t)N_, 1);
                }
                if (bp->requiresGrad) {
                    BufRef gb{bp->grad.buf, bp->grad.offset + bb * K_ * N_ * 4, K_ * N_ * 4};
                    gemmCall("gemm_tn", pab, go, gb, (uint32_t)K_, (uint32_t)N_, (uint32_t)M_, 1);
                }
            }
        };
    }
    return reg(out);
}

GTPtr gMatmulT(const GTPtr& a, const GTPtr& b) {
    size_t ra = a->shape.size(), rb = b->shape.size();
    if (ra < 2 || rb < 2) throw std::runtime_error("gMatmulT rank");
    uint64_t M = a->shape[ra - 2], K = a->shape[ra - 1];
    uint64_t NB = b->shape[rb - 2], K2 = b->shape[rb - 1];
    if (K != K2) throw std::runtime_error("gMatmulT inner dim");
    uint64_t numelA = gNumel(a->shape), numelB = gNumel(b->shape);
    uint64_t batchA = numelA / (M * K);
    uint64_t batchB = numelB / (NB * K);
    if (batchB != 1 && batchB != batchA) throw std::runtime_error("gMatmulT batch");
    bool rg = a->requiresGrad || b->requiresGrad;
    Shape os = a->shape;
    os[ra - 2] = M;
    os[ra - 1] = NB;
    auto out = makeTensor(os, rg, false);
    uint32_t sBf = batchB == 1 ? 0u : (uint32_t)(NB * K);
    gemmCall("gemm_nt", a->data, b->data, out->data, (uint32_t)M, (uint32_t)NB, (uint32_t)K, 0, (uint32_t)batchA,
             (uint32_t)(M * K), sBf, (uint32_t)(M * NB));
    if (rg) {
        out->name = "gMatmulT";
        uint64_t M_ = M, K_ = K, NB_ = NB, bA = batchA, bB = batchB;
        auto ap = a, bp = b, op = out;
        out->backward = [ap, bp, op, M_, K_, NB_, bA, bB]() {
            if (bA > 1 && bB > 1) {
                if (ap->requiresGrad)
                    gemmCall("gemm_nn", op->grad, bp->data, ap->grad, (uint32_t)M_, (uint32_t)K_, (uint32_t)NB_, 1,
                             (uint32_t)bA, (uint32_t)(M_ * NB_), (uint32_t)(NB_ * K_), (uint32_t)(M_ * K_));
                if (bp->requiresGrad)
                    gemmCall("gemm_tn", op->grad, ap->data, bp->grad, (uint32_t)NB_, (uint32_t)K_, (uint32_t)M_, 1,
                             (uint32_t)bA, (uint32_t)(M_ * NB_), (uint32_t)(M_ * K_), (uint32_t)(NB_ * K_));
                return;
            }
            for (uint64_t bi = 0; bi < bA; bi++) {
                uint64_t bb = (bB == 1 ? 0 : bi);
                BufRef go{op->grad.buf, op->grad.offset + bi * M_ * NB_ * 4, M_ * NB_ * 4};
                BufRef pab{ap->data.buf, ap->data.offset + bi * M_ * K_ * 4, M_ * K_ * 4};
                BufRef pb{bp->data.buf, bp->data.offset + bb * NB_ * K_ * 4, NB_ * K_ * 4};
                if (ap->requiresGrad) {
                    BufRef ga{ap->grad.buf, ap->grad.offset + bi * M_ * K_ * 4, M_ * K_ * 4};
                    // dA = go * B
                    gemmCall("gemm_nn", go, pb, ga, (uint32_t)M_, (uint32_t)K_, (uint32_t)NB_, 1);
                }
                if (bp->requiresGrad) {
                    BufRef gb{bp->grad.buf, bp->grad.offset + bb * NB_ * K_ * 4, NB_ * K_ * 4};
                    // dB[n,k] = sum_m go[m,n]*a[m,k]
                    gemmCall("gemm_tn", go, pab, gb, (uint32_t)NB_, (uint32_t)K_, (uint32_t)M_, 1);
                }
            }
        };
    }
    return reg(out);
}

// dense GroupFFN fully-batched forward/backward: all experts folded into batched GEMMs; merge/bias/GELU take one dispatch each.
// w1 {G,d,fg}  w2 {G,fg,d}  b1 {G,fg}  b2 {G,d}  wT {G,rows} (topk weights)
GTPtr gMoeOut(const GTPtr& flat, const GTPtr& w1, const GTPtr& b1, const GTPtr& w2, const GTPtr& b2,
              const GTPtr& wSparse, size_t nGroups, size_t fg) {
    size_t rows = flat->shape[0];
    size_t d = flat->shape[1];
    size_t hc = nGroups * fg;
    bool rg = flat->requiresGrad || w1->requiresGrad || b1->requiresGrad || w2->requiresGrad || b2->requiresGrad ||
              wSparse->requiresGrad;
    auto pre = makeTensor(Shape{rows, hc}, false, false);  // raw gemm output, becomes pre-activation in place
    auto H = makeTensor(Shape{rows, hc}, false, false);    // gelu output
    auto Y = makeTensor(Shape{nGroups * rows, d}, false, false);
    auto out = makeTensor(Shape{rows, d}, rg, false);

    // Hraw[g] = flat @ w1[g] (A broadcast), layout [rows, nGroups*fg]
    gemmCall("gemm_nn", flat->data, w1->data, pre->data, (uint32_t)rows, (uint32_t)fg, (uint32_t)d, 0,
             (uint32_t)nGroups, 0, (uint32_t)(d * fg), (uint32_t)fg, 0, (uint32_t)fg, (uint32_t)hc);
    {
        struct {
            uint32_t rows, cols;
        } pc{(uint32_t)rows, (uint32_t)hc};
        G.eng.dispatch("bias_gelu_fwd", {pre->data, b1->data, H->data}, &pc, CEIL((uint64_t)rows * hc, 256), 1, 1);
    }
    // Y[g] = H[g] @ w2[g]
    gemmCall("gemm_nn", H->data, w2->data, Y->data, (uint32_t)rows, (uint32_t)d, (uint32_t)fg, 0, (uint32_t)nGroups,
             (uint32_t)fg, (uint32_t)(fg * d), (uint32_t)(rows * d), (uint32_t)hc, (uint32_t)d, (uint32_t)d);
    {
        struct {
            uint32_t rows, d, g;
        } pc{(uint32_t)rows, (uint32_t)d, (uint32_t)nGroups};
        G.eng.dispatch("moe_merge_fwd", {Y->data, b2->data, wSparse->data, out->data}, &pc,
                       CEIL((uint64_t)rows * d, 256), 1, 1);
    }
    if (rg) {
        out->name = "gMoeOut";
        BufRef dY = stepAlloc((uint64_t)nGroups * rows * d * 4);
        BufRef dPre = stepAlloc((uint64_t)rows * hc * 4);
        BufRef dFlatTmp = stepAlloc((uint64_t)nGroups * rows * d * 4);
        auto flatp = flat, w1p = w1, b1p = b1, w2p = w2, b2p = b2, wSp = wSparse, op = out;
        auto Hp = H, Yp = Y, prep = pre;
        out->backward = [=]() {
            {
                struct {
                    uint32_t rows, d, g;
                } pc{(uint32_t)rows, (uint32_t)d, (uint32_t)nGroups};
                G.eng.dispatch("moe_merge_dy", {op->grad, wSp->data, dY}, &pc, CEIL((uint64_t)nGroups * rows * d, 256),
                               1, 1);
            }
            {
                struct {
                    uint32_t rows, d, g;
                } pc{(uint32_t)rows, (uint32_t)d, (uint32_t)nGroups};
                G.eng.dispatch("moe_merge_wg_grad", {op->grad, Yp->data, dY, b2p->grad, wSp->grad}, &pc,
                               (uint32_t)(nGroups * (d + rows)), 1, 1);
            }
            // dPre[g] = dY[g] @ w2[g]^T
            gemmCall("gemm_nt", dY, w2p->data, dPre, (uint32_t)rows, (uint32_t)fg, (uint32_t)d, 0, (uint32_t)nGroups,
                     (uint32_t)(rows * d), (uint32_t)(fg * d), (uint32_t)fg, (uint32_t)d, (uint32_t)d, (uint32_t)hc);
            // dW2[g] = H[g]^T @ dY[g]
            gemmCall("gemm_tn", Hp->data, dY, w2p->grad, (uint32_t)fg, (uint32_t)d, (uint32_t)rows, 1,
                     (uint32_t)nGroups, (uint32_t)fg, (uint32_t)(rows * d), (uint32_t)(fg * d), (uint32_t)hc,
                     (uint32_t)d, (uint32_t)d);
            {
                struct {
                    uint32_t rows, cols;
                } pc{(uint32_t)rows, (uint32_t)hc};
                G.eng.dispatch("gelu_bias_bwd", {dPre, prep->data, b1p->grad}, &pc, (uint32_t)hc, 1, 1);
            }
            // dFlat[g] = dPre[g] @ w1[g]^T, then summed across experts
            gemmCall("gemm_nt", dPre, w1p->data, dFlatTmp, (uint32_t)rows, (uint32_t)d, (uint32_t)fg, 0,
                     (uint32_t)nGroups, (uint32_t)fg, (uint32_t)(d * fg), (uint32_t)(rows * d), (uint32_t)hc,
                     (uint32_t)fg, (uint32_t)d);
            {
                struct {
                    uint32_t n, g;
                } pc{(uint32_t)(rows * d), (uint32_t)nGroups};
                G.eng.dispatch("reduce_g", {dFlatTmp, flatp->grad}, &pc, CEIL((uint64_t)rows * d, 256), 1, 1);
            }
            // dW1[g] = flat^T @ dPre[g]
            gemmCall("gemm_tn", flatp->data, dPre, w1p->grad, (uint32_t)d, (uint32_t)fg, (uint32_t)rows, 1,
                     (uint32_t)nGroups, 0, (uint32_t)fg, (uint32_t)(d * fg), (uint32_t)d, (uint32_t)hc, (uint32_t)fg);
        };
    }
    return reg(out);
}

// ---------------------------------------------------------------- elementwise

void gDumpCopyTags(const char* outPath) {
    std::vector<std::pair<uint64_t, std::string>> v;
    for (auto& kv : gCopyTags) v.push_back({kv.second, kv.first});
    std::sort(v.rbegin(), v.rend());
    FILE* f = std::fopen(outPath, "w");
    for (auto& kv : v) {
        std::fprintf(stderr, "  copy %6llu  %s\n", (unsigned long long)kv.first, kv.second.c_str());
        if (f) std::fprintf(f, "copy %llu %s\n", (unsigned long long)kv.first, kv.second.c_str());
    }
    if (f) std::fclose(f);
}

static void copyNd(const BufRef& dst, const BufRef& src, const Shape& dims, const uint32_t dstSt[4],
                   const uint32_t srcSt[4], uint32_t mode, float scale, const char* tag = "misc") {
    gCopyTags[tag]++;
    uint32_t d[4];
    shapeDims4(dims, d);
    P_nd pc{};
    pc.n = (uint32_t)gNumel(dims);
    pc.d0 = d[0];
    pc.d1 = d[1];
    pc.d2 = d[2];
    pc.d3 = d[3];
    pc.os0 = dstSt[0];
    pc.os1 = dstSt[1];
    pc.os2 = dstSt[2];
    pc.os3 = dstSt[3];
    pc.ss0 = srcSt[0];
    pc.ss1 = srcSt[1];
    pc.ss2 = srcSt[2];
    pc.ss3 = srcSt[3];
    pc.mode = mode;
    pc.scale = scale;
    G.eng.dispatch("copy_nd", {dst, src}, &pc, CEIL(pc.n, 256), 1, 1);
}

static void ew2(const BufRef& c, const BufRef& a, const BufRef& b, const Shape& dims, const uint32_t aSt[4],
                const uint32_t bSt[4], uint32_t mode) {
    uint32_t d[4];
    shapeDims4(dims, d);
    P_ew2 pc{};
    pc.n = (uint32_t)gNumel(dims);
    pc.d0 = d[0];
    pc.d1 = d[1];
    pc.d2 = d[2];
    pc.d3 = d[3];
    pc.as0 = aSt[0];
    pc.as1 = aSt[1];
    pc.as2 = aSt[2];
    pc.as3 = aSt[3];
    pc.bs0 = bSt[0];
    pc.bs1 = bSt[1];
    pc.bs2 = bSt[2];
    pc.bs3 = bSt[3];
    pc.mode = mode;
    G.eng.dispatch("ew2", {c, a, b}, &pc, CEIL(pc.n, 256), 1, 1);
}

static void mulGrad(const BufRef& target, const Shape& outShape, const uint32_t tSt[4], const BufRef& go,
                    const BufRef& other, const uint32_t oSt[4]) {
    uint32_t d[4];
    shapeDims4(outShape, d);
    P_mg pc{};
    pc.n = (uint32_t)gNumel(outShape);
    pc.d0 = d[0];
    pc.d1 = d[1];
    pc.d2 = d[2];
    pc.d3 = d[3];
    pc.ts0 = tSt[0];
    pc.ts1 = tSt[1];
    pc.ts2 = tSt[2];
    pc.ts3 = tSt[3];
    pc.os0 = oSt[0];
    pc.os1 = oSt[1];
    pc.os2 = oSt[2];
    pc.os3 = oSt[3];
    G.eng.dispatch("mul_grad", {target, go, other}, &pc, CEIL(pc.n, 256), 1, 1);
}

// backward recipient for an operand that may be broadcast against out
static void broadcastGrad(const GTPtr& operand, const GTPtr& out, const BufRef& go) {
    if (operand->shape == out->shape) {
        uint32_t st[4];
        contigStrides4(out->shape, st);
        copyNd(operand->grad, go, out->shape, st, st, 1, 1.0f, "bwd_add");
        return;
    }
    if (gNumel(operand->shape) == 1) {
        P_bs pc{(uint32_t)gNumel(out->shape), 1.0f};
        G.eng.dispatch("reduce_all", {go, operand->grad}, &pc, 1, 1, 1);
        return;
    }
    if (operand->shape.size() == 1) {
        uint64_t cols = operand->shape[0];
        if (out->shape.back() != cols) throw std::runtime_error("broadcastGrad: bad 1D");
        P_rc pc{(uint32_t)(gNumel(out->shape) / cols), (uint32_t)cols};
        G.eng.dispatch("colsum_add", {go, operand->grad}, &pc, (uint32_t)cols, 1, 1);
        return;
    }
    // operand broadcast as leading batch dims: out == batch copies of operand
    uint64_t on = gNumel(operand->shape);
    if (out->shape.size() == operand->shape.size() + 1 && out->shape[0] * on == gNumel(out->shape)) {
        P_rb pc{(uint32_t)out->shape[0], (uint32_t)operand->shape[operand->shape.size() - 2],
                (uint32_t)operand->shape.back()};
        G.eng.dispatch("repeat_batch_bwd", {go, operand->grad}, &pc, CEIL((uint64_t)pc.s * pc.d, 256), 1, 1);
        return;
    }
    throw std::runtime_error("broadcastGrad: unsupported pattern");
}

// gradient for one operand of mul (handles broadcast cases)
static void mulGradFor(const GTPtr& operand, const GTPtr& other, const GTPtr& out, const BufRef& go) {
    if (operand->shape == out->shape) {
        uint32_t ost[4], tst[4];
        bcastStrides4(other->shape, out->shape, ost);
        contigStrides4(out->shape, tst);
        mulGrad(operand->grad, out->shape, tst, go, other->data, ost);
        return;
    }
    if (gNumel(operand->shape) == 1) {
        uint64_t n = gNumel(out->shape);
        P_rc pc{(uint32_t)n, 1};
        G.eng.dispatch("colsum_mul_add", {operand->grad, go, other->data}, &pc, 1, 1, 1);
        return;
    }
    if (operand->shape.size() == 1) {
        uint64_t cols = operand->shape[0];
        if (out->shape.back() != cols) throw std::runtime_error("mulGradFor: bad 1D");
        P_rc pc{(uint32_t)(gNumel(out->shape) / cols), (uint32_t)cols};
        G.eng.dispatch("colsum_mul_add", {operand->grad, go, other->data}, &pc, (uint32_t)cols, 1, 1);
        return;
    }
    // trailing-1 broadcast: {..., 1} x {..., cols}
    if (!operand->shape.empty() && operand->shape.back() == 1 && operand->shape.size() == out->shape.size()) {
        bool prefixOk = true;
        for (size_t i = 0; i + 1 < operand->shape.size(); i++)
            if (operand->shape[i] != out->shape[i]) prefixOk = false;
        if (prefixOk) {
            uint64_t cols = out->shape.back();
            uint64_t rows = gNumel(out->shape) / cols;
            P_rc pc{(uint32_t)rows, (uint32_t)cols};
            G.eng.dispatch("rowdot", {go, other->data, operand->grad}, &pc, CEIL(rows, 64), 1, 1);
            return;
        }
    }
    throw std::runtime_error("mulGradFor: unsupported pattern");
}

GTPtr gAdd(const GTPtr& a, const GTPtr& b) {
    Shape os;
    broadcastShape(a->shape, b->shape, os);
    bool rg = a->requiresGrad || b->requiresGrad;
    auto out = makeTensor(os, rg, false);
    uint32_t ast[4], bst[4];
    bcastStrides4(a->shape, os, ast);
    bcastStrides4(b->shape, os, bst);
    ew2(out->data, a->data, b->data, os, ast, bst, 0);
    if (rg) {
        out->name = "gAdd";
        auto ap = a, bp = b, op = out;
        out->backward = [ap, bp, op]() {
            if (ap->requiresGrad && bp->requiresGrad && ap->shape == op->shape && bp->shape == op->shape) {
                struct {
                    uint32_t n, mode;
                } pc{(uint32_t)gNumel(op->shape), 1u};
                G.eng.dispatch("copy2", {ap->grad, bp->grad, op->grad}, &pc, CEIL(pc.n, 256), 1, 1);
                return;
            }
            broadcastGrad(ap, op, op->grad);
            broadcastGrad(bp, op, op->grad);
        };
    }
    return reg(out);
}

GTPtr gMul(const GTPtr& a, const GTPtr& b) {
    Shape os;
    broadcastShape(a->shape, b->shape, os);
    bool rg = a->requiresGrad || b->requiresGrad;
    auto out = makeTensor(os, rg, false);
    uint32_t ast[4], bst[4];
    bcastStrides4(a->shape, os, ast);
    bcastStrides4(b->shape, os, bst);
    ew2(out->data, a->data, b->data, os, ast, bst, 1);
    if (rg) {
        out->name = "gMul";
        auto ap = a, bp = b, op = out;
        out->backward = [ap, bp, op]() {
            if (ap->requiresGrad && bp->requiresGrad && ap->shape == op->shape && bp->shape == op->shape) {
                struct {
                    uint32_t n;
                } pc{(uint32_t)gNumel(op->shape)};
                G.eng.dispatch("mul2", {ap->grad, bp->grad, op->grad, ap->data, bp->data}, &pc, CEIL(pc.n, 256), 1, 1);
                return;
            }
            if (ap->requiresGrad) mulGradFor(ap, bp, op, op->grad);
            if (bp->requiresGrad) mulGradFor(bp, ap, op, op->grad);
        };
    }
    return reg(out);
}

GTPtr gScale(const GTPtr& a, float s, bool track) {
    bool rg = a->requiresGrad && track;
    auto out = makeTensor(a->shape, rg, false);
    uint32_t st[4];
    contigStrides4(a->shape, st);
    copyNd(out->data, a->data, a->shape, st, st, 0, s, "scale_fwd");
    if (rg) {
        out->name = "gScale";
        auto ap = a;
        out->backward = [ap, op = out, s]() {
            uint32_t st2[4];
            contigStrides4(op->shape, st2);
            copyNd(ap->grad, op->grad, op->shape, st2, st2, 1, s, "bwd_scale");
        };
    }
    return reg(out);
}

GTPtr gGelu(const GTPtr& x) {
    bool rg = x->requiresGrad;
    auto out = makeTensor(x->shape, rg, false);
    P_un pc{(uint32_t)gNumel(x->shape), 0};
    G.eng.dispatch("unary_fwd", {x->data, out->data}, &pc, CEIL(pc.n, 256), 1, 1);
    if (rg) {
        out->name = "gGelu";
        auto xp = x, op = out;
        out->backward = [xp, op]() {
            P_un pc2{(uint32_t)gNumel(op->shape), 0};
            G.eng.dispatch("unary_bwd", {xp->data, op->grad, xp->grad}, &pc2, CEIL(pc2.n, 256), 1, 1);
        };
    }
    return reg(out);
}

GTPtr gSigmoid(const GTPtr& x) {
    bool rg = x->requiresGrad;
    auto out = makeTensor(x->shape, rg, false);
    P_un pc{(uint32_t)gNumel(x->shape), 1};
    G.eng.dispatch("unary_fwd", {x->data, out->data}, &pc, CEIL(pc.n, 256), 1, 1);
    if (rg) {
        out->name = "gSigmoid";
        auto xp = x, op = out;
        out->backward = [xp, op]() {
            P_un pc2{(uint32_t)gNumel(op->shape), 1};
            G.eng.dispatch("unary_bwd", {xp->data, op->grad, xp->grad}, &pc2, CEIL(pc2.n, 256), 1, 1);
        };
    }
    return reg(out);
}

// ---------------------------------------------------------------- reductions

GTPtr gSoftmaxLast(const GTPtr& x) {
    size_t r = x->shape.size();
    uint64_t cols = x->shape[r - 1];
    uint64_t rows = gNumel(x->shape) / cols;
    bool rg = x->requiresGrad;
    auto out = makeTensor(x->shape, rg, false);
    P_rc pc{(uint32_t)rows, (uint32_t)cols};
    G.eng.dispatch("softmax_fwd", {x->data, out->data}, &pc, (uint32_t)rows, 1, 1);
    if (rg) {
        out->name = "gSoftmax";
        auto xp = x, op = out;
        out->backward = [xp, op]() {
            P_rc pc2{(uint32_t)(gNumel(op->shape) / op->shape.back()), (uint32_t)op->shape.back()};
            G.eng.dispatch("softmax_bwd", {op->data, op->grad, xp->grad}, &pc2,
                           (uint32_t)(gNumel(op->shape) / op->shape.back()), 1, 1);
        };
    }
    return reg(out);
}

GTPtr gRmsNorm(const GTPtr& x, const GTPtr& w, float eps) {
    size_t r = x->shape.size();
    uint64_t d = x->shape[r - 1];
    uint64_t rows = gNumel(x->shape) / d;
    bool rg = x->requiresGrad || w->requiresGrad;
    auto out = makeTensor(x->shape, rg, false);
    P_rn pc{(uint32_t)rows, (uint32_t)d, eps, 0};
    G.eng.dispatch("rmsnorm_fwd", {x->data, w->data, out->data}, &pc, (uint32_t)rows, 1, 1);
    if (rg) {
        out->name = "gRmsNorm";
        auto xp = x, wp = w, op = out;
        uint64_t d_ = d, rows_ = rows;
        out->backward = [xp, wp, op, d_, rows_]() {
            auto gwRow = std::make_shared<GTensor>();
            gwRow->shape = Shape{rows_, d_};
            gwRow->data = stepAlloc(rows_ * d_ * 4);
            P_rn pc2{(uint32_t)rows_, (uint32_t)d_, 1e-6f, 0};
            if (xp->requiresGrad) {
                G.eng.dispatch("rmsnorm_bwd", {xp->data, wp->data, op->grad, xp->grad, gwRow->data}, &pc2,
                               (uint32_t)rows_, 1, 1);
            } else {
                // still need gwRow: run kernel with a scratch gx (unused path avoided by
                // giving xp a scratch grad? xp has no grad; use gwRow buffer as dummy)
                gwRow->data = stepAlloc(rows_ * d_ * 4);
                BufRef dummy = stepAlloc(rows_ * d_ * 4);
                zeroBuf(dummy);
                G.eng.dispatch("rmsnorm_bwd", {xp->data, wp->data, op->grad, dummy, gwRow->data}, &pc2, (uint32_t)rows_,
                               1, 1);
            }
            if (wp->requiresGrad) {
                P_rc pc3{(uint32_t)rows_, (uint32_t)d_};
                G.eng.dispatch("colsum_add", {gwRow->data, wp->grad}, &pc3, (uint32_t)d_, 1, 1);
            }
        };
    }
    return reg(out);
}

GTPtr gRowSum(const GTPtr& x) {
    size_t r = x->shape.size();
    uint64_t d = x->shape[r - 1];
    uint64_t rows = gNumel(x->shape) / d;
    bool rg = x->requiresGrad;
    Shape os = x->shape;
    os[r - 1] = 1;
    auto out = makeTensor(os, rg, false);
    P_rc pc{(uint32_t)rows, (uint32_t)d};
    G.eng.dispatch("rowsum_fwd", {x->data, out->data}, &pc, (uint32_t)rows, 1, 1);
    if (rg) {
        out->name = "gRowSum";
        auto xp = x, op = out;
        out->backward = [xp, op]() {
            // replicate go over the summed last dim: iterate over x coords
            uint32_t dstSt[4], srcSt[4];
            contigStrides4(xp->shape, dstSt);
            bcastStrides4(op->shape, xp->shape, srcSt);  // last dim -> stride 0
            copyNd(xp->grad, op->grad, xp->shape, dstSt, srcSt, 1, 1.0f, "bwd_rowsum");
        };
    }
    return reg(out);
}

GTPtr gMeanSeq(const GTPtr& x) {
    size_t r = x->shape.size();
    uint64_t d = x->shape[r - 1];
    uint64_t T = x->shape[r - 2];
    uint64_t B = gNumel(x->shape) / (T * d);
    Shape os{x->shape.begin(), x->shape.end() - 2};
    os.push_back(d);
    bool rg = x->requiresGrad;
    auto out = makeTensor(os, rg, false);
    struct P {
        uint32_t B, T, d;
    } pc{(uint32_t)B, (uint32_t)T, (uint32_t)d};
    G.eng.dispatch("mean_seq", {x->data, out->data}, &pc, CEIL(B * d, 256), 1, 1);
    if (rg) {
        out->name = "gMeanSeq";
        auto xp = x, op = out;
        out->backward = [xp, op]() {
            uint32_t dstSt[4], srcSt[4];
            contigStrides4(xp->shape, dstSt);
            bcastStrides4(op->shape, xp->shape, srcSt);
            float invT = 1.0f / (float)xp->shape[xp->shape.size() - 2];
            copyNd(xp->grad, op->grad, xp->shape, dstSt, srcSt, 1, invT, "bwd_meanseq");
        };
    }
    return reg(out);
}

GTPtr gReduceAll(const GTPtr& x) {
    bool rg = x->requiresGrad;
    // reduce_all accumulates (needed in backward), so the output must start zeroed
    auto out = makeTensor(Shape{1}, rg, true);
    P_bs pc{(uint32_t)gNumel(x->shape), 1.0f};
    G.eng.dispatch("reduce_all", {x->data, out->data}, &pc, 1, 1, 1);
    if (rg) {
        out->name = "gReduceAll";
        auto xp = x, op = out;
        out->backward = [xp, op]() {
            P_bs pc2{(uint32_t)gNumel(xp->shape), 1.0f / (float)gNumel(xp->shape)};
            G.eng.dispatch("bcast_scalar", {op->grad, xp->grad}, &pc2, CEIL(pc2.n, 256), 1, 1);
        };
    }
    return reg(out);
}

GTPtr gMeanAll(const GTPtr& x) {
    bool rg = x->requiresGrad;
    auto out = makeTensor(Shape{1}, rg, true);
    P_bs pc{(uint32_t)gNumel(x->shape), 1.0f / (float)gNumel(x->shape)};
    G.eng.dispatch("reduce_all", {x->data, out->data}, &pc, 1, 1, 1);
    if (rg) {
        out->name = "gMeanAll";
        auto xp = x, op = out;
        out->backward = [xp, op]() {
            P_bs pc2{(uint32_t)gNumel(xp->shape), 1.0f / (float)gNumel(xp->shape)};
            G.eng.dispatch("bcast_scalar", {op->grad, xp->grad}, &pc2, CEIL(pc2.n, 256), 1, 1);
        };
    }
    return reg(out);
}

GTPtr gCrossEntropy(const GTPtr& logits, const BufRef& tgt, const BufRef& mask, size_t rows, size_t v,
                    float lossScale) {
    bool rg = logits->requiresGrad;
    auto rowLoss = std::make_shared<GTensor>();
    rowLoss->shape = Shape{rows};
    rowLoss->data = stepAlloc(rows * 4);
    P_ce pc{(uint32_t)rows, (uint32_t)v};
    G.eng.dispatch("crossentropy_fwd", {logits->data, tgt, rowLoss->data, mask}, &pc, (uint32_t)rows, 1, 1);
    float s = lossScale > 0.0f ? lossScale : 1.0f / (float)rows;
    auto out = makeTensor(Shape{1}, rg, true);
    P_bs prd{(uint32_t)rows, s};
    G.eng.dispatch("reduce_all", {rowLoss->data, out->data}, &prd, 1, 1, 1);
    if (rg) {
        out->name = "gCrossEntropy";
        auto lp = logits, op = out;
        out->backward = [lp, op, tgt, mask, rows, v, s]() {
            P_ceb pc2{(uint32_t)rows, (uint32_t)v, s, 0};
            G.eng.dispatch("crossentropy_bwd", {lp->data, tgt, op->grad, lp->grad, mask}, &pc2, (uint32_t)rows, 1, 1);
        };
    }
    return reg(out);
}

// fused attention score post-processing: scale + mask + concat(local,mem) + softmax
GTPtr gAttnSoftmax(const GTPtr& sloc, const GTPtr& smem, const BufRef& mask, size_t T, size_t M, float scale,
                   bool hasMask) {
    size_t rows = gNumel(sloc->shape) / T;
    Shape os = sloc->shape;
    os.back() += M;
    bool rg = sloc->requiresGrad || (smem && smem->requiresGrad);
    auto out = makeTensor(os, rg, false);
    struct P {
        uint32_t rows, T, M, hasMask;
        float scale;
    } pc{(uint32_t)rows, (uint32_t)T, (uint32_t)M, hasMask ? 1u : 0u, scale};
    BufRef smData = smem ? smem->data : sloc->data;
    BufRef maskUse = hasMask ? mask : sloc->data;
    G.eng.dispatch("attn_softmax_fwd", {sloc->data, smData, maskUse, out->data}, &pc, (uint32_t)rows, 1, 1);
    if (rg) {
        out->name = "gAttnSoftmax";
        auto sl = sloc, sm = smem, op = out;
        BufRef mk = maskUse;
        out->backward = [sl, sm, op, mk, pc]() {
            BufRef dsm = sm ? sm->grad : sl->grad;
            G.eng.dispatch("attn_softmax_bwd", {op->data, op->grad, mk, sl->grad, dsm}, &pc, pc.rows, 1, 1);
        };
    }
    return reg(out);
}

// ---------------------------------------------------------------- data movement

GTPtr gEmbedding(const GTPtr& table, const BufRef& ids, size_t n) {
    size_t d = table->shape[1];
    bool rg = table->requiresGrad;
    auto out = makeTensor(Shape{n, d}, rg, false);
    P_g pc{(uint32_t)n, (uint32_t)d};
    G.eng.dispatch("gather_rows", {table->data, ids, out->data}, &pc, CEIL(n * d, 256), 1, 1);
    if (rg) {
        out->name = "gEmbedding";
        auto tp = table, op = out;
        out->backward = [tp, op, ids, n, d]() {
            P_g pc2{(uint32_t)n, (uint32_t)d};
            G.eng.dispatch("scatter_add_rows", {tp->grad, ids, op->grad}, &pc2, CEIL(n * d, 256), 1, 1);
        };
    }
    return reg(out);
}

GTPtr gGatherRows(const GTPtr& src, const BufRef& idx, size_t n) {
    size_t r = src->shape.size();
    size_t d = src->shape[r - 1];
    bool rg = src->requiresGrad;
    auto out = makeTensor(Shape{n, d}, rg, false);
    P_g pc{(uint32_t)n, (uint32_t)d};
    G.eng.dispatch("gather_rows", {src->data, idx, out->data}, &pc, CEIL(n * d, 256), 1, 1);
    if (rg) {
        out->name = "gGatherRows";
        auto sp = src, op = out;
        out->backward = [sp, op, idx, n, d]() {
            P_g pc2{(uint32_t)n, (uint32_t)d};
            G.eng.dispatch("scatter_add_rows", {sp->grad, idx, op->grad}, &pc2, CEIL(n * d, 256), 1, 1);
        };
    }
    return reg(out);
}

GTPtr gScatterAddRowsBase(const GTPtr& base, const BufRef& idx, const GTPtr& src, size_t n) {
    size_t r = base->shape.size();
    size_t d = base->shape[r - 1];
    bool rg = src->requiresGrad || base->requiresGrad;
    auto out = makeTensor(base->shape, rg, false);
    uint32_t st[4];
    contigStrides4(base->shape, st);
    copyNd(out->data, base->data, base->shape, st, st, 0, 1.0f, "scatter_fwd");
    P_g pc{(uint32_t)n, (uint32_t)d};
    G.eng.dispatch("scatter_add_rows", {out->data, idx, src->data}, &pc, CEIL(n * d, 256), 1, 1);
    if (rg) {
        out->name = "gScatterAdd";
        auto bp = base, sp = src, op = out;
        out->backward = [bp, sp, op, idx, n, d]() {
            if (bp->requiresGrad) {
                uint32_t st2[4];
                contigStrides4(bp->shape, st2);
                copyNd(bp->grad, op->grad, bp->shape, st2, st2, 1, 1.0f, "bwd_scatter");
            }
            if (sp->requiresGrad) {
                P_g pc2{(uint32_t)n, (uint32_t)d};
                G.eng.dispatch("gather_rows", {op->grad, idx, sp->grad}, &pc2, CEIL(n * d, 256), 1, 1);
            }
        };
    }
    return reg(out);
}

GTPtr gRope(const GTPtr& x, const BufRef& pos, float base, bool inverse) {
    size_t r = x->shape.size();
    uint64_t dh = x->shape[r - 1];
    uint64_t seq = x->shape[r - 2];
    uint64_t batch = gNumel(x->shape) / (seq * dh);
    uint64_t half = dh / 2;
    bool rg = x->requiresGrad;
    auto out = makeTensor(x->shape, rg, false);
    P_rope pc{(uint32_t)batch, (uint32_t)seq, (uint32_t)half, (uint32_t)(inverse ? 1 : 0), base, 0};
    G.eng.dispatch("rope_fwd", {x->data, pos, out->data}, &pc, CEIL(batch * seq * half, 256), 1, 1);
    if (rg) {
        out->name = "gRope";
        auto xp = x, op = out;
        out->backward = [xp, op, pos, base, inverse]() {
            size_t r2 = op->shape.size();
            uint64_t dh2 = op->shape[r2 - 1], seq2 = op->shape[r2 - 2];
            uint64_t batch2 = gNumel(op->shape) / (seq2 * dh2);
            P_rope pc2{(uint32_t)batch2, (uint32_t)seq2, (uint32_t)(dh2 / 2), (uint32_t)(inverse ? 1 : 0), base, 0};
            G.eng.dispatch("rope_bwd", {op->grad, pos, xp->grad}, &pc2, CEIL(batch2 * seq2 * (dh2 / 2), 256), 1, 1);
        };
    }
    return reg(out);
}

GTPtr gMaskFillLast(const GTPtr& x, const BufRef& mask, size_t period) {
    bool rg = x->requiresGrad;
    auto out = makeTensor(x->shape, rg, false);
    P_mask pc{(uint32_t)gNumel(x->shape), (uint32_t)period};
    G.eng.dispatch("mask_fill_fwd", {x->data, mask, out->data}, &pc, CEIL(pc.n, 256), 1, 1);
    if (rg) {
        out->name = "gMaskFill";
        auto xp = x, op = out;
        out->backward = [xp, op, mask, period]() {
            P_mask pc2{(uint32_t)gNumel(op->shape), (uint32_t)period};
            G.eng.dispatch("mask_fill_bwd", {op->grad, mask, xp->grad}, &pc2, CEIL(pc2.n, 256), 1, 1);
        };
    }
    return reg(out);
}

GTPtr gSliceSeq(const GTPtr& x, size_t start, size_t len) {
    size_t r = x->shape.size();
    uint64_t d = x->shape[r - 1];
    uint64_t seq = x->shape[r - 2];
    uint64_t outer = gNumel(x->shape) / (seq * d);
    Shape os = x->shape;
    os[r - 2] = len;
    bool rg = x->requiresGrad;
    auto out = makeTensor(os, rg, false);
    uint32_t st[4];
    contigStrides4(Shape{len, d}, st);
    Shape dims{len, d};
    for (uint64_t o = 0; o < outer; o++) {
        BufRef src{x->data.buf, x->data.offset + (o * seq + start) * d * 4, len * d * 4};
        BufRef dst{out->data.buf, out->data.offset + o * len * d * 4, len * d * 4};
        copyNd(dst, src, dims, st, st, 0, 1.0f, "slice_fwd");
    }
    if (rg) {
        out->name = "gSliceSeq";
        auto xp = x, op = out;
        out->backward = [xp, op, start, len, seq, d, outer]() {
            uint32_t st2[4];
            contigStrides4(Shape{len, d}, st2);
            Shape dims2{len, d};
            for (uint64_t o = 0; o < outer; o++) {
                BufRef dst{xp->grad.buf, xp->grad.offset + (o * seq + start) * d * 4, len * d * 4};
                BufRef src{op->grad.buf, op->grad.offset + o * len * d * 4, len * d * 4};
                copyNd(dst, src, dims2, st2, st2, 1, 1.0f, "bwd_slice");
            }
        };
    }
    return reg(out);
}

GTPtr gConcatSeq(const GTPtr& a, const GTPtr& b) {
    size_t r = a->shape.size();
    uint64_t d = a->shape[r - 1];
    uint64_t sa = a->shape[r - 2], sb = b->shape[r - 2];
    uint64_t outer = gNumel(a->shape) / (sa * d);
    Shape os = a->shape;
    os[r - 2] = sa + sb;
    bool rg = a->requiresGrad || b->requiresGrad;
    auto out = makeTensor(os, rg, false);
    // one 3D strided copy covers all outer indices (the per-outer loop used to cost dozens of dispatches per layer)
    if (outer > 1) {
        uint32_t dstSt[4], srcSt[4];
        Shape dimsA{outer, sa, d}, dimsB{outer, sb, d};
        contigStrides4(dimsA, dstSt);  // {1, d, (sa+sb)*d}
        dstSt[2] = (uint32_t)((sa + sb) * d);
        contigStrides4(dimsA, srcSt);  // {1, d, sa*d}
        srcSt[2] = (uint32_t)(sa * d);
        copyNd(out->data, a->data, dimsA, dstSt, srcSt, 0, 1.0f, "concatSeq_fwd");
        contigStrides4(dimsB, dstSt);  // {1, d, (sa+sb)*d}
        dstSt[2] = (uint32_t)((sa + sb) * d);
        contigStrides4(dimsB, srcSt);  // {1, d, sb*d}
        srcSt[2] = (uint32_t)(sb * d);
        BufRef dstB{out->data.buf, out->data.offset + (uint64_t)sa * d * 4, out->data.size - (uint64_t)sa * d * 4};
        copyNd(dstB, b->data, dimsB, dstSt, srcSt, 0, 1.0f, "concatSeq_fwd");
    } else {
        uint32_t st[4];
        Shape dimsA{sa, d}, dimsB{sb, d};
        contigStrides4(dimsA, st);
        copyNd(out->data, a->data, dimsA, st, st, 0, 1.0f, "concatSeq_fwd");
        contigStrides4(dimsB, st);
        BufRef dstB{out->data.buf, out->data.offset + (uint64_t)sa * d * 4, out->data.size - (uint64_t)sa * d * 4};
        copyNd(dstB, b->data, dimsB, st, st, 0, 1.0f, "concatSeq_fwd");
    }
    if (rg) {
        out->name = "gConcatSeq";
        auto ap = a, bp = b, op = out;
        out->backward = [ap, bp, op, sa, sb, d, outer]() {
            if (outer > 1) {
                uint32_t dstSt[4], srcSt[4];
                Shape dimsA{outer, sa, d}, dimsB{outer, sb, d};
                if (ap->requiresGrad) {
                    contigStrides4(dimsA, dstSt);
                    dstSt[2] = (uint32_t)(sa * d);
                    contigStrides4(dimsA, srcSt);
                    srcSt[2] = (uint32_t)((sa + sb) * d);
                    copyNd(ap->grad, op->grad, dimsA, dstSt, srcSt, 1, 1.0f, "bwd_concatSeq");
                }
                if (bp->requiresGrad) {
                    contigStrides4(dimsB, dstSt);
                    dstSt[2] = (uint32_t)(sb * d);
                    contigStrides4(dimsB, srcSt);
                    srcSt[2] = (uint32_t)((sa + sb) * d);
                    BufRef srcB{op->grad.buf, op->grad.offset + (uint64_t)sa * d * 4,
                                op->grad.size - (uint64_t)sa * d * 4};
                    copyNd(bp->grad, srcB, dimsB, dstSt, srcSt, 1, 1.0f, "bwd_concatSeq");
                }
                return;
            }
            uint32_t st2[4];
            if (ap->requiresGrad) {
                Shape dimsA{sa, d};
                contigStrides4(dimsA, st2);
                copyNd(ap->grad, op->grad, dimsA, st2, st2, 1, 1.0f, "bwd_concatSeq");
            }
            if (bp->requiresGrad) {
                Shape dimsB{sb, d};
                contigStrides4(dimsB, st2);
                BufRef srcB{op->grad.buf, op->grad.offset + (uint64_t)sa * d * 4, op->grad.size - (uint64_t)sa * d * 4};
                copyNd(bp->grad, srcB, dimsB, st2, st2, 1, 1.0f, "bwd_concatSeq");
            }
        };
    }
    return reg(out);
}

GTPtr gConcatLast(const std::vector<GTPtr>& xs) {
    size_t r = xs[0]->shape.size();
    uint64_t outLast = 0;
    uint64_t rows = gNumel(xs[0]->shape) / xs[0]->shape[r - 1];
    for (auto& x : xs) {
        if (gNumel(x->shape) / x->shape[r - 1] != rows) throw std::runtime_error("gConcatLast row mismatch");
        outLast += x->shape[r - 1];
    }
    Shape os = xs[0]->shape;
    os[r - 1] = outLast;
    bool rg = false;
    for (auto& x : xs) rg = rg || x->requiresGrad;
    auto out = makeTensor(os, rg, false);
    std::vector<uint64_t> offs;
    uint64_t acc = 0;
    for (auto& x : xs) {
        offs.push_back(acc);
        acc += x->shape[r - 1];
    }
    uint32_t ost[4];
    contigStrides4(os, ost);
    for (size_t k = 0; k < xs.size(); k++) {
        uint32_t sst[4];
        contigStrides4(xs[k]->shape, sst);
        BufRef dst{out->data.buf, out->data.offset + offs[k] * 4, out->data.size - offs[k] * 4};
        copyNd(dst, xs[k]->data, xs[k]->shape, ost, sst, 0, 1.0f, "concatLast_fwd");
    }
    if (rg) {
        out->name = "gConcatLast";
        auto op = out;
        out->backward = [xs, op, offs, outLast]() {
            size_t r2 = op->shape.size();
            uint32_t ost2[4];
            contigStrides4(op->shape, ost2);
            for (size_t k = 0; k < xs.size(); k++) {
                if (!xs[k]->requiresGrad) continue;
                uint32_t sst2[4];
                contigStrides4(xs[k]->shape, sst2);
                BufRef src{op->grad.buf, op->grad.offset + offs[k] * 4, op->grad.size - offs[k] * 4};
                copyNd(xs[k]->grad, src, xs[k]->shape, sst2, ost2, 1, 1.0f, "bwd_concatLast");
            }
            (void)r2;
            (void)outLast;
        };
    }
    return reg(out);
}

GTPtr gSplitHeads(const GTPtr& x, size_t heads) {
    size_t r = x->shape.size();
    uint64_t t = x->shape[r - 2], d = x->shape[r - 1];
    uint64_t dh = d / heads;
    Shape os = x->shape;
    os[r - 2] = heads;
    os[r - 1] = t;
    os.push_back(dh);  // becomes {..., heads, t, dh}
    bool rg = x->requiresGrad;
    auto out = makeTensor(os, rg, false);
    // out coords slots: j, ti, hh, b ; x addr = b*(t*d) + ti*d + hh*dh + j
    uint32_t dstSt[4], srcSt[4];
    contigStrides4(os, dstSt);
    srcSt[0] = 1;
    srcSt[1] = (uint32_t)d;
    srcSt[2] = (uint32_t)dh;
    srcSt[3] = (uint32_t)(t * d);
    copyNd(out->data, x->data, os, dstSt, srcSt, 0, 1.0f, "splitHeads_fwd");
    if (rg) {
        out->name = "gSplitHeads";
        auto xp = x, op = out;
        out->backward = [xp, op]() {
            // out coords (b,hh,ti,j), slots: j, ti, hh, b
            // x addr = b*(t*d) + ti*d + hh*dh + j
            uint64_t t2 = xp->shape[xp->shape.size() - 2];
            uint64_t d2 = xp->shape.back();
            uint32_t goSt[4], dstByOut[4];
            contigStrides4(op->shape, goSt);
            dstByOut[0] = 1;
            dstByOut[1] = (uint32_t)d2;
            dstByOut[2] = (uint32_t)op->shape.back();
            dstByOut[3] = (uint32_t)(t2 * d2);
            copyNd(xp->grad, op->grad, op->shape, dstByOut, goSt, 1, 1.0f, "bwd_splitHeads");
        };
    }
    return reg(out);
}

GTPtr gMergeHeads(const GTPtr& x) {
    size_t b = x->shape[0], h = x->shape[1], t = x->shape[2], dh = x->shape[3];
    Shape os{b, t, h * dh};
    bool rg = x->requiresGrad;
    auto out = makeTensor(os, rg, false);
    // iterate over x coords (b,h,ti,j); src contiguous;
    // dst addr = b*(t*d) + ti*d + hh*dh + j  with d = h*dh
    uint32_t srcSt[4], dSt[4];
    srcSt[0] = 1;
    srcSt[1] = dh;
    srcSt[2] = t * dh;
    srcSt[3] = h * t * dh;
    dSt[0] = 1;
    dSt[1] = (uint32_t)(h * dh);
    dSt[2] = dh;
    dSt[3] = (uint32_t)(t * h * dh);
    Shape dims{b, h, t, dh};
    copyNd(out->data, x->data, dims, dSt, srcSt, 0, 1.0f, "mergeHeads_fwd");
    if (rg) {
        out->name = "gMergeHeads";
        auto xp = x, op = out;
        out->backward = [xp, op, b, h, t, dh]() {
            Shape dims2{b, h, t, dh};
            uint32_t xst[4];
            contigStrides4(dims2, xst);  // x grad contiguous over x coords
            uint32_t ost[4];
            ost[0] = 1;
            ost[1] = (uint32_t)(h * dh);
            ost[2] = dh;
            ost[3] = (uint32_t)(t * h * dh);
            copyNd(xp->grad, op->grad, dims2, xst, ost, 1, 1.0f, "bwd_mergeHeads");
        };
    }
    return reg(out);
}

GTPtr gTransposeLast2(const GTPtr& x) {
    size_t r = x->shape.size();
    uint64_t M = x->shape[r - 2], N = x->shape[r - 1];
    uint64_t outer = gNumel(x->shape) / (M * N);
    Shape os = x->shape;
    os[r - 2] = N;
    os[r - 1] = M;
    bool rg = x->requiresGrad;
    auto out = makeTensor(os, rg, false);
    for (uint64_t o = 0; o < outer; o++) {
        Shape dims{N, M};  // coords c0 = i (M index), c1 = j (N index)
        uint32_t dstSt[4], srcSt[4];
        contigStrides4(dims, dstSt);  // addr = i + N*j
        srcSt[0] = (uint32_t)N;       // i -> stride N
        srcSt[1] = 1;                 // j -> stride 1
        srcSt[2] = 0;
        srcSt[3] = 0;
        BufRef dst{out->data.buf, out->data.offset + o * M * N * 4, M * N * 4};
        BufRef src{x->data.buf, x->data.offset + o * M * N * 4, M * N * 4};
        copyNd(dst, src, dims, dstSt, srcSt, 0, 1.0f, "transpose_fwd");
    }
    if (rg) {
        out->name = "gTransposeLast2";
        auto xp = x, op = out;
        out->backward = [xp, op, M, N, outer]() {
            for (uint64_t o = 0; o < outer; o++) {
                Shape dims{N, M};
                uint32_t dstSt2[4], srcSt2[4];
                contigStrides4(dims, srcSt2);  // go is contiguous in the same coords
                dstSt2[0] = (uint32_t)N;
                dstSt2[1] = 1;
                dstSt2[2] = 0;
                dstSt2[3] = 0;
                BufRef dst{xp->grad.buf, xp->grad.offset + o * M * N * 4, M * N * 4};
                BufRef src{op->grad.buf, op->grad.offset + o * M * N * 4, M * N * 4};
                copyNd(dst, src, dims, dstSt2, srcSt2, 1, 1.0f, "bwd_transpose");
            }
        };
    }
    return reg(out);
}

// strides (last dim first) mapping out coords -> x coords with `dim` fixed at `index`
static void selectDimStrides(const Shape& xShape, size_t dim, uint32_t st[4]) {
    size_t r = xShape.size();
    uint32_t xst[4];
    contigStrides4(xShape, xst);
    for (size_t i = 0; i < 4; i++) st[i] = 0;
    for (size_t i = 0; i < r - 1 && i < 4; i++) {
        size_t outDim = r - 2 - i;    // out dim number (from the end) for this coord
        if (outDim >= dim) outDim++;  // skip the removed dim
        if (outDim < r) st[i] = xst[r - 1 - outDim];
    }
}

GTPtr gSelectDim(const GTPtr& x, size_t dim, size_t index) {
    size_t r = x->shape.size();
    Shape os = x->shape;
    os.erase(os.begin() + (long)dim);
    bool rg = x->requiresGrad;
    auto out = makeTensor(os, rg, false);
    uint32_t dstSt[4], srcSt[4];
    contigStrides4(os, dstSt);
    selectDimStrides(x->shape, dim, srcSt);
    uint64_t strideAtDim = 1;
    for (size_t i = dim + 1; i < r; i++) strideAtDim *= x->shape[i];
    uint64_t off = index * strideAtDim;
    BufRef src{x->data.buf, x->data.offset + off * 4, x->data.size - off * 4};
    copyNd(out->data, src, os, dstSt, srcSt, 0, 1.0f, "select_fwd");
    if (rg) {
        out->name = "gSelectDim";
        auto xp = x, op = out;
        out->backward = [xp, op, dim, index, strideAtDim]() {
            uint32_t goSt[4], dstByOut[4];
            contigStrides4(op->shape, goSt);
            selectDimStrides(xp->shape, dim, dstByOut);
            BufRef dst{xp->grad.buf, xp->grad.offset + (index * strideAtDim) * 4,
                       xp->grad.size - index * strideAtDim * 4};
            copyNd(dst, op->grad, op->shape, dstByOut, goSt, 1, 1.0f, "bwd_select");
        };
    }
    return reg(out);
}

GTPtr gUnsqueezeFirst(const GTPtr& x) {
    Shape os;
    os.push_back(1);
    for (auto v : x->shape) os.push_back(v);
    return gViewShape(x, os);
}

GTPtr gRepeatBatch(const GTPtr& x, size_t batch) {
    size_t r = x->shape.size();
    uint64_t s = x->shape[r - 2], d = x->shape[r - 1];
    Shape os = x->shape;
    os.insert(os.begin(), batch);
    bool rg = x->requiresGrad;
    auto out = makeTensor(os, rg, false);
    uint32_t dstSt[4], srcSt[4];
    contigStrides4(os, dstSt);
    // coords (b, s, d): src addr = s*d + d_idx  (batch broadcasts)
    srcSt[0] = 1;
    srcSt[1] = (uint32_t)d;
    srcSt[2] = 0;
    srcSt[3] = 0;
    copyNd(out->data, x->data, os, dstSt, srcSt, 0, 1.0f, "repeat_fwd");
    if (rg) {
        out->name = "gRepeatBatch";
        auto xp = x, op = out;
        out->backward = [xp, op, batch, s, d]() {
            P_rb pc{(uint32_t)batch, (uint32_t)s, (uint32_t)d};
            G.eng.dispatch("repeat_batch_bwd", {op->grad, xp->grad}, &pc, CEIL(s * d, 256), 1, 1);
        };
    }
    return reg(out);
}

// ---------------------------------------------------------------- routing (MoE)

GTPtr gTopkSparse(const GTPtr& probs, size_t k, const BufRef& countsOut, BufRef* idxOut) {
    size_t r2 = probs->shape.size();
    uint64_t g = probs->shape[r2 - 1];
    uint64_t rows = gNumel(probs->shape) / g;
    bool rg = probs->requiresGrad;
    auto sparse = makeTensor(probs->shape, rg, false);
    BufRef idxBuf = stepAlloc(rows * k * 4);
    BufRef srowBuf = stepAlloc(rows * 4);
    zeroBuf(countsOut);
    P_topk pc{(uint32_t)rows, (uint32_t)g, (uint32_t)k};
    G.eng.dispatch("topk_sparse_fwd", {probs->data, sparse->data, idxBuf, srowBuf, countsOut}, &pc, (uint32_t)rows, 1,
                   1);
    if (rg) {
        sparse->name = "gTopkSparse";
        auto pp = probs, sp = sparse;
        sparse->backward = [pp, sp, idxBuf, srowBuf, k, g, rows]() {
            P_topk pc2{(uint32_t)rows, (uint32_t)g, (uint32_t)k};
            G.eng.dispatch("topk_sparse_bwd", {sp->data, sp->grad, idxBuf, srowBuf, pp->grad}, &pc2, (uint32_t)rows, 1,
                           1);
        };
    }
    if (idxOut) *idxOut = idxBuf;
    return reg(sparse);
}

void gCompact(const GTPtr& sparse, size_t k, size_t rows, size_t g, const BufRef& idx, const BufRef& tIdxOut,
              const BufRef& wColOut, const BufRef& countsOut) {
    zeroBuf(countsOut);
    P_topk pc{(uint32_t)rows, (uint32_t)g, (uint32_t)k};
    G.eng.dispatch("compact", {idx, sparse->data, tIdxOut, wColOut, countsOut}, &pc, (uint32_t)rows, 1, 1);
}

GTPtr gCompactWCol(const BufRef& wCol, size_t count, const BufRef& tIdx, const BufRef& counts, size_t g, size_t gid,
                   size_t rows, const GTPtr& sparse) {
    bool rg = sparse->requiresGrad;
    auto t = std::make_shared<GTensor>();
    t->shape = Shape{(size_t)count, 1};
    t->data = wCol;
    t->requiresGrad = rg;
    if (rg) {
        t->grad = stepAlloc(count * 4);
        zeroBuf(t->grad);
        t->name = "gWCol";
        auto sp = sparse, tp = t;
        t->backward = [sp, tp, tIdx, counts, g, gid, rows]() {
            P_cp pc{(uint32_t)rows, (uint32_t)g, (uint32_t)gid};
            G.eng.dispatch("compact_bwd", {tIdx, tp->grad, counts, sp->grad}, &pc, CEIL(rows, 256), 1, 1);
        };
    }
    return reg(t);
}

GTPtr gOneHotBalance(const GTPtr& probs, const BufRef& counts, size_t rows, size_t groups) {
    bool rg = probs->requiresGrad;
    auto out = makeTensor(Shape{1}, rg, false);
    P_ohb pc{(uint32_t)rows, (uint32_t)groups};
    G.eng.dispatch("onehot_balance_fwd", {probs->data, counts, out->data}, &pc, 1, 1, 1);
    if (rg) {
        out->name = "gOneHotBalance";
        auto pp = probs, op = out;
        out->backward = [pp, op, counts, rows, groups]() {
            P_ohb pc2{(uint32_t)rows, (uint32_t)groups};
            G.eng.dispatch("onehot_balance_bwd", {op->grad, counts, pp->grad}, &pc2, CEIL(rows * groups, 256), 1, 1);
        };
    }
    return reg(out);
}

// ---------------------------------------------------------------- trainer helpers

void adamwStep(const GTPtr& p, const BufRef& m, const BufRef& v, const BufRef& clip, const P_adamw& pc) {
    float hp[8] = {pc.lr, pc.b1, pc.b2, pc.eps, pc.wd, pc.invBc1, pc.invBc2, 0.0f};
    gUploadHp(hp, 8);
    struct {
        uint32_t n;
    } core{pc.n};
    G.eng.dispatch("adamw", {p->data, p->grad, m, v, clip, gHpBuf()}, &core, CEIL(pc.n, 256), 1, 1);
}

}  // namespace gpu
}  // namespace ofc
