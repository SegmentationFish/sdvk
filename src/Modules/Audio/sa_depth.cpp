// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
#include "SaDepth.hpp"

#include "Modules/GPU/gpu_ops.hpp"
#include "Modules/GPU/vk_engine.hpp"
#include "Modules/Model/Gguf.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace ofc {
namespace audio {

namespace {
using gpu::BufRef;
using gpu::G;

constexpr uint64_t align256(uint64_t n) { return (n + 255) & ~255ull; }

struct PushGemm {
    uint32_t M, N, K, mode, sA, sB, sC, batch, lda, ldb, ldc, pad;
};

constexpr int H = 4096, FF = 6144, HD = 256, HEADS = 16, SMAX = 8;

void gemmQ8(const BufRef& a, const BufRef& wq, const BufRef& c, const BufRef& ws, uint32_t M,
            uint32_t N, uint32_t K) {
    PushGemm pc{M, N, K, 0, 0, 0, 0, 1, 0, 0, 0, 0};
    G.eng.dispatch("gemm_nt_q8_v", {a, wq, c, ws}, &pc, (N + 63) / 64, (M + 63) / 64, 1);
}

void gemmF32(const BufRef& a, const BufRef& b, const BufRef& c, uint32_t M, uint32_t N, uint32_t K,
             uint32_t batch, uint32_t sA, uint32_t sB, uint32_t sC, uint32_t lda, uint32_t ldb,
             uint32_t ldc, bool nt) {
    PushGemm pc{M, N, K, 0, sA, sB, sC, batch, lda, ldb, ldc, 0};
    G.eng.dispatch(nt ? "gemm_nt" : "gemm_nn", {a, b, c}, &pc, (N + 63) / 64, (M + 63) / 64, batch);
}

}  // namespace

struct SaDepth::Impl {
    GgufFile gguf;
    std::string lastErr;

    struct Q8W {
        BufRef q, s;
        int rows = 0, cols = 0;
    };
    struct Layer {
        Q8W q, k, v, o, gate, up, down;
        BufRef n1, n2;
    };
    std::vector<Layer> layers;
    Q8W projection;
    Q8W audioEmb;                       // [7168, 4096]
    std::vector<Q8W> heads;             // 7 x [1024, 4096]
    BufRef posEmb;                      // [16, 4096] fp32
    BufRef normW;                       // [4096] fp32

    // activations
    BufRef x, xn, qb, kb, vb, attn, ff, gateUp, scores, probs, logits, stateBuf;

    BufRef loadQ8(const std::string& name, bool persist = false) {
        std::vector<int8_t> q;
        std::vector<float> sc;
        if (!gguf.readQ8(name, q, sc, nullptr)) {
            std::fprintf(stderr, "depth: cannot read %s\n", name.c_str());
            std::abort();
        }
        BufRef bq = G.paramArena.alloc(align256(q.size()));
        BufRef bs = G.paramArena.alloc(align256(sc.size() * 4));
        G.eng.uploadNow(bq, q.data(), q.size());
        G.eng.uploadNow(bs, sc.data(), sc.size() * 4);
        (void)persist;
        return bq;
    }

    Q8W q8(const std::string& name) {
        Q8W w;
        const GgufTensor* t = gguf.find(name);
        if (!t) {
            std::fprintf(stderr, "depth: missing %s\n", name.c_str());
            std::abort();
        }
        auto sh = t->shape();
        w.rows = (int)sh.first;
        w.cols = (int)sh.second;
        std::vector<int8_t> q;
        std::vector<float> sc;
        if (!gguf.readQ8(name, q, sc, nullptr)) {
            std::fprintf(stderr, "depth: %s is not Q8_0\n", name.c_str());
            std::abort();
        }
        w.q = G.paramArena.alloc(align256(q.size()));
        w.s = G.paramArena.alloc(align256(sc.size() * 4));
        G.eng.uploadNow(w.q, q.data(), q.size());
        G.eng.uploadNow(w.s, sc.data(), sc.size() * 4);
        return w;
    }

    BufRef f32(const std::string& name, bool persist = false) {
        std::vector<float> v = gguf.readFloat(name, nullptr);
        BufRef b = G.paramArena.alloc(align256(v.size() * 4));
        G.eng.uploadNow(b, v.data(), v.size() * 4);
        (void)persist;
        return b;
    }

    void embedRow(const Q8W& w, int row, std::vector<float>& out) {
        // Q8_0: 34 bytes per 32 values
        const uint64_t rowBytes = (uint64_t)(w.cols / 32) * 34;
        std::vector<uint8_t> buf((size_t)rowBytes);
        if (!gguf.readRange(*gguf.find("audio_embeddings.weight"), (uint64_t)row * rowBytes, buf.data(),
                            rowBytes)) {
            std::fprintf(stderr, "depth: embed row read failed\n");
            std::abort();
        }
        out.resize((size_t)w.cols);
        for (int b = 0; b < w.cols / 32; b++) {
            const uint8_t* blk = buf.data() + b * 34;
            uint16_t h = (uint16_t)(blk[0] | (blk[1] << 8));
            uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, man = h & 0x3FF;
            float d;
            if (exp == 0) d = std::ldexp((float)man, -24);
            else if (exp == 31) d = man ? NAN : INFINITY;
            else d = std::ldexp((float)(man | 0x400), (int)exp - 25);
            if (sign) d = -d;
            for (int i = 0; i < 32; i++)
                out[(size_t)b * 32 + i] = d * (float)(int8_t)blk[2 + i];
        }
    }
};

SaDepth::~SaDepth() { delete impl_; }

bool SaDepth::load(const std::string& ggufPath, const std::string& spvDir, std::string& err) {
    auto* m = new Impl();
    impl_ = m;
    if (!m->gguf.open(ggufPath, &err)) return false;

    if (!G.eng.alive()) {
        G.allocGradArena = false;
        G.stepBytes = 64ull << 20;
        G.scratchBytes = 64ull << 20;
        G.paramBytes = 64ull << 20;
        ofc::gpu::gInit(spvDir);
    }
    G.eng.initArenas(64ull << 20, 16ull << 20);
    G.paramArena.destroy();
    G.paramArena.init(G.eng.device(), G.eng.physDevice(), 900ull << 20, true, false);

    m->projection = m->q8("projection.weight");
    m->audioEmb = m->q8("audio_embeddings.weight");
    m->posEmb = m->f32("pos_embedding.weight");
    m->normW = m->f32("norm.weight");
    m->layers.resize(4);
    for (int i = 0; i < 4; i++) {
        auto& L = m->layers[(size_t)i];
        const std::string p = "layers." + std::to_string(i) + ".";
        L.q = m->q8(p + "attn.to_q.weight");
        L.k = m->q8(p + "attn.to_k.weight");
        L.v = m->q8(p + "attn.to_v.weight");
        L.o = m->q8(p + "attn.to_out.weight");
        L.gate = m->q8(p + "gate_proj.weight");
        L.up = m->q8(p + "up_proj.weight");
        L.down = m->q8(p + "down_proj.weight");
        L.n1 = m->f32(p + "input_layernorm.weight");
        L.n2 = m->f32(p + "post_attention_layernorm.weight");
    }
    m->heads.resize(kDepthCodebooks);
    for (int i = 0; i < kDepthCodebooks; i++)
        m->heads[(size_t)i] = m->q8("audio_heads." + std::to_string(i) + ".weight");

    gpu::Arena& a = G.eng.scratchArena();
    auto alloc = [&](uint64_t floats) { return a.alloc(align256(floats * 4)); };
    m->x = alloc((uint64_t)SMAX * H);
    m->xn = alloc((uint64_t)SMAX * H);
    m->qb = alloc((uint64_t)SMAX * H);
    m->kb = alloc((uint64_t)SMAX * H);
    m->vb = alloc((uint64_t)SMAX * H);
    m->attn = alloc((uint64_t)SMAX * H);
    m->ff = alloc((uint64_t)SMAX * FF);
    m->gateUp = alloc((uint64_t)SMAX * 2 * FF);
    m->scores = alloc((uint64_t)HEADS * SMAX * SMAX);
    m->probs = alloc((uint64_t)HEADS * SMAX * SMAX);
    m->logits = alloc((uint64_t)kDepthVocab);
    m->stateBuf = alloc((uint64_t)kDepthStates * H);
    std::printf("depth: 4 layers, %d codebooks, weights resident (Q8)\n", kDepthCodebooks);
    return true;
}

bool SaDepth::decodeFrame(const std::vector<float>& lmHidden, int semantic, uint32_t seed, float cfg,
                          std::vector<int>& codes, std::vector<float>& states,
                          std::vector<float>& embedding, std::string& err) {
    auto* m = impl_;
    codes.assign(kDepthCodebooks, 0);
    states.assign((size_t)kDepthStates * H, 0.0f);
    embedding.assign(H, 0.0f);
    if ((int)lmHidden.size() != H) {
        err = "depth: lm hidden must be 4096";
        return false;
    }
    std::memcpy(states.data(), lmHidden.data(), sizeof(float) * H);
    std::mt19937 rng(seed);

    // x[0] = projection(lm hidden) + pos[0]; then each chosen code appends embed + pos.
    std::vector<float> vec((size_t)SMAX * H, 0.0f);
    std::vector<float> pos((size_t)kDepthPositions * H);
    G.eng.downloadNow(m->posEmb, pos.data(), pos.size() * 4);
    std::vector<float> h1((size_t)H);
    G.eng.beginStep();
    G.eng.uploadNow(m->x, lmHidden.data(), sizeof(float) * H);
    gemmQ8(m->x, m->projection.q, m->stateBuf, m->projection.s, 1, (uint32_t)H, (uint32_t)H);
    G.eng.endStep();
    G.eng.waitStep();
    G.eng.downloadNow(m->stateBuf, h1.data(), sizeof(float) * H);
    for (int i = 0; i < H; i++) vec[(size_t)i] = h1[(size_t)i] + pos[(size_t)i];

    std::vector<float> logits(kDepthVocab);
    std::vector<float> emb;
    int S = 1;
    for (int step = 0; step < kDepthCodebooks; step++) {
        G.eng.beginStep();
        G.eng.uploadNow(m->x, vec.data(), sizeof(float) * (size_t)S * H);
        struct {
            uint32_t n, pad;
        } noMask{(uint32_t)0, 0};
        (void)noMask;
        for (int l = 0; l < 4; l++) {
            auto& L = m->layers[(size_t)l];
            struct {
                uint32_t rows, d;
                float eps;
            } rp{(uint32_t)S, (uint32_t)H, 1e-6f};
            G.eng.dispatch("sa_rmsnorm", {m->x, L.n1, m->xn}, &rp, (uint32_t)S, 1, 1);
            gemmQ8(m->xn, L.q.q, m->qb, L.q.s, (uint32_t)S, (uint32_t)H, (uint32_t)H);
            gemmQ8(m->xn, L.k.q, m->kb, L.k.s, (uint32_t)S, (uint32_t)H, (uint32_t)H);
            gemmQ8(m->xn, L.v.q, m->vb, L.v.s, (uint32_t)S, (uint32_t)H, (uint32_t)H);
            // 16 heads x 256 dims, causal, scale 1/16
            gemmF32(m->qb, m->kb, m->scores, (uint32_t)S, (uint32_t)S, (uint32_t)HD, (uint32_t)HEADS,
                    (uint32_t)HD, (uint32_t)HD, (uint32_t)(S * S), (uint32_t)H, (uint32_t)H, (uint32_t)S,
                    true);
            {
                struct {
                    uint32_t heads, rows, cols, offset;
                } cm{(uint32_t)HEADS, (uint32_t)S, (uint32_t)S, 0};
                G.eng.dispatch("sa_causal_mask", {m->scores}, &cm,
                               (uint32_t)((HEADS * S * S + 255) / 256), 1, 1);
                struct {
                    uint32_t n;
                    float s;
                } sc{(uint32_t)(HEADS * S * S), 1.0f / 16.0f};
                G.eng.dispatch("sa_scale", {m->scores, m->probs}, &sc,
                               (uint32_t)((HEADS * S * S + 255) / 256), 1, 1);
                struct {
                    uint32_t rows, cols;
                } sp{(uint32_t)(HEADS * S), (uint32_t)S};
                G.eng.dispatch("softmax_fwd", {m->probs, m->scores}, &sp, (uint32_t)(HEADS * S), 1, 1);
            }
            gemmF32(m->scores, m->vb, m->attn, (uint32_t)S, (uint32_t)HD, (uint32_t)S, (uint32_t)HEADS,
                    (uint32_t)(S * S), (uint32_t)HD, (uint32_t)HD, (uint32_t)S, (uint32_t)H, (uint32_t)H,
                    false);
            gemmQ8(m->attn, L.o.q, m->xn, L.o.s, (uint32_t)S, (uint32_t)H, (uint32_t)H);
            {
                struct {
                    uint32_t n, pad;
                } ap{(uint32_t)(S * H), 0};
                G.eng.dispatch("add2", {m->x, m->x, m->xn}, &ap, (uint32_t)(((uint64_t)S * H + 255) / 256),
                               1, 1);
            }
            G.eng.dispatch("sa_rmsnorm", {m->x, L.n2, m->xn}, &rp, (uint32_t)S, 1, 1);
            // rows are [value | gate] interleaved, so both projections write with ldc = 2*FF
            {
                PushGemm pu{(uint32_t)S, (uint32_t)FF, (uint32_t)H, 0, 0, 0, 0, 1, 0, 0,
                            (uint32_t)(2 * FF), 0};
                G.eng.dispatch("gemm_nt_q8_v",
                               {m->xn, L.up.q, BufRef{m->gateUp.buf, m->gateUp.offset, (uint64_t)S * 2 * FF * 4},
                                L.up.s},
                               &pu, ((uint32_t)FF + 63) / 64, ((uint32_t)S + 63) / 64, 1);
                PushGemm pg{(uint32_t)S, (uint32_t)FF, (uint32_t)H, 0, 0, 0, 0, 1, 0, 0,
                            (uint32_t)(2 * FF), 0};
                G.eng.dispatch("gemm_nt_q8_v",
                               {m->xn, L.gate.q,
                                BufRef{m->gateUp.buf, m->gateUp.offset + (uint64_t)FF * 4,
                                       (uint64_t)S * 2 * FF * 4},
                                L.gate.s},
                               &pg, ((uint32_t)FF + 63) / 64, ((uint32_t)S + 63) / 64, 1);
                struct {
                    uint32_t rows, half_;
                } sg{(uint32_t)S, (uint32_t)FF};
                G.eng.dispatch("sa_swiglu",
                               {BufRef{m->gateUp.buf, m->gateUp.offset, (uint64_t)S * 2 * FF * 4},
                                BufRef{m->ff.buf, m->ff.offset, (uint64_t)S * FF * 4}},
                               &sg, (uint32_t)((S * FF + 255) / 256), 1, 1);
            }
            gemmQ8(m->ff, L.down.q, m->xn, L.down.s, (uint32_t)S, (uint32_t)H, (uint32_t)FF);
            {
                struct {
                    uint32_t n, pad;
                } ap{(uint32_t)(S * H), 0};
                G.eng.dispatch("add2", {m->x, m->x, m->xn}, &ap, (uint32_t)(((uint64_t)S * H + 255) / 256),
                               1, 1);
            }
        }
        // head for this codebook reads the last position's normalised hidden state
        {
            struct {
                uint32_t rows, d;
                float eps;
            } rp{(uint32_t)S, (uint32_t)H, 1e-6f};
            G.eng.dispatch("sa_rmsnorm", {m->x, m->normW, m->xn}, &rp, (uint32_t)S, 1, 1);
            BufRef last = BufRef{m->xn.buf, m->xn.offset + (uint64_t)(S - 1) * H * 4, (uint64_t)H * 4};
            gemmQ8(last, m->heads[(size_t)step].q, m->logits, m->heads[(size_t)step].s, 1,
                   (uint32_t)kDepthVocab, (uint32_t)H);
        }
        G.eng.endStep();
        G.eng.waitStep();
        G.eng.downloadNow(m->logits, logits.data(), logits.size() * 4);
        G.eng.downloadNow(BufRef{m->xn.buf, m->xn.offset + (uint64_t)(S - 1) * H * 4, (uint64_t)H * 4},
                          &states[(size_t)(step + 1) * H], sizeof(float) * H);

        // sample the codebook entry
        float mx = *std::max_element(logits.begin(), logits.end());
        {
            float mn = *std::min_element(logits.begin(), logits.end());
            int arg = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
            bool fin = true;
            for (float v : logits)
                if (!std::isfinite(v)) fin = false;
            std::fprintf(stderr, "[d] step %d S %d: finite %d min %.3f max %.3f argmax %d first %d %.3f %.3f\n",
                         step, S, (int)fin, mn, mx, arg, (int)codes[0], logits[0], logits[1]);
        }
        double sum = 0;
        std::vector<float> p((size_t)kDepthVocab);
        for (int i = 0; i < kDepthVocab; i++) {
            p[(size_t)i] = std::exp(logits[(size_t)i] - mx);
            sum += p[(size_t)i];
        }
        std::uniform_real_distribution<double> u(0.0, 1.0);
        double r = u(rng) * sum, acc = 0;
        int pick = 0;
        for (int i = 0; i < kDepthVocab; i++) {
            acc += p[(size_t)i];
            if (acc >= r) {
                pick = i;
                break;
            }
        }
        (void)cfg;
        codes[(size_t)step] = pick;

        // append the chosen embedding for the next step
        m->embedRow(m->audioEmb, step * kDepthVocab + pick, emb);
        for (int i = 0; i < H; i++)
            vec[(size_t)S * H + i] = emb[(size_t)i] + pos[(size_t)(S)*H + i];
        if (S + 1 < SMAX) S++;

        // accumulate the feedback embedding (semantic first, then each acoustic code)
        for (int i = 0; i < H; i++) embedding[(size_t)i] += emb[(size_t)i];
    }
    // `embedding` is the sum of the seven acoustic embeddings; the semantic embedding comes from
    // the LM's own token table, so the caller adds it before scaling by 8^-0.5.
    (void)semantic;
    return true;
}

}  // namespace audio
}  // namespace ofc
