// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
#include "SaLm.hpp"

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

void gemmKq(const BufRef& a, const BufRef& w, const BufRef& c, uint32_t M, uint32_t N, uint32_t K,
            uint32_t type) {
    PushGemm pc{M, N, K, 0, 0, 0, 0, 1, 0, 0, 0, type};
    G.eng.dispatch("gemm_nt_q4k", {a, w, c}, &pc, (N + 63) / 64, (M + 63) / 64, 1);
}

void gemmF32(const BufRef& a, const BufRef& b, const BufRef& c, uint32_t M, uint32_t N, uint32_t K,
             uint32_t batch = 1, uint32_t sA = 0, uint32_t sB = 0, uint32_t sC = 0, uint32_t lda = 0,
             uint32_t ldb = 0, uint32_t ldc = 0, bool nt = true) {
    PushGemm pc{M, N, K, 0, sA, sB, sC, batch, lda, ldb, ldc, 0};
    G.eng.dispatch(nt ? "gemm_nt" : "gemm_nn", {a, b, c}, &pc, (N + 63) / 64, (M + 63) / 64, batch);
}

float halfToFloat(uint16_t h) {
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, man = h & 0x3FF;
    float v;
    if (exp == 0) v = std::ldexp((float)man, -24);
    else if (exp == 31) v = man ? NAN : INFINITY;
    else v = std::ldexp((float)(man | 0x400), (int)exp - 25);
    return sign ? -v : v;
}

uint8_t byteAt(const std::vector<uint8_t>& b, size_t i) { return i < b.size() ? b[i] : 0; }

void getScaleMinK4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Host dequantisation of one quantised row (mirrors the verified Gguf.cpp kernels).
void dequantRow(uint32_t type, const uint8_t* p, size_t k, float* out) {
    if (type == 12) {  // Q4_K
        for (size_t b = 0; b < k / 256; b++) {
            const uint8_t* blk = p + b * 144;
            float d = halfToFloat((uint16_t)(blk[0] | (blk[1] << 8)));
            float dmin = halfToFloat((uint16_t)(blk[2] | (blk[3] << 8)));
            const uint8_t* scales = blk + 4;
            const uint8_t* q = blk + 16;
            float* o = out + b * 256;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc, m;
                getScaleMinK4(is + 0, scales, sc, m);
                float d1 = d * (float)sc, m1 = dmin * (float)m;
                getScaleMinK4(is + 1, scales, sc, m);
                float d2 = d * (float)sc, m2 = dmin * (float)m;
                for (int l = 0; l < 32; l++) o[j + l] = d1 * (float)(q[l] & 0xF) - m1;
                for (int l = 0; l < 32; l++) o[j + 32 + l] = d2 * (float)(q[l] >> 4) - m2;
                q += 32;
                is += 2;
            }
        }
    } else if (type == 14) {  // Q6_K
        for (size_t b = 0; b < k / 256; b++) {
            const uint8_t* blk = p + b * 210;
            const uint8_t* ql = blk;
            const uint8_t* qh = blk + 128;
            const int8_t* scales = (const int8_t*)(blk + 192);
            float d = halfToFloat((uint16_t)(blk[208] | (blk[209] << 8)));
            float* o = out + b * 256;
            for (int n = 0; n < 256; n += 128) {
                const uint8_t* ql2 = ql + n / 2;
                const uint8_t* qh2 = qh + n / 4;
                const int8_t* sc = scales + n / 16;
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    int q1 = (int)((ql2[l] & 0xF) | (((qh2[l] >> 0) & 3) << 4)) - 32;
                    int q2 = (int)((ql2[l + 32] & 0xF) | (((qh2[l] >> 2) & 3) << 4)) - 32;
                    int q3 = (int)((ql2[l] >> 4) | (((qh2[l] >> 4) & 3) << 4)) - 32;
                    int q4 = (int)((ql2[l + 32] >> 4) | (((qh2[l] >> 6) & 3) << 4)) - 32;
                    o[n + l] = d * (float)sc[is + 0] * (float)q1;
                    o[n + l + 32] = d * (float)sc[is + 2] * (float)q2;
                    o[n + l + 64] = d * (float)sc[is + 4] * (float)q3;
                    o[n + l + 96] = d * (float)sc[is + 6] * (float)q4;
                }
            }
        }
    } else {
        std::fprintf(stderr, "lm: unsupported row quant type %u\n", type);
        std::abort();
    }
}

}  // namespace

struct SaLm::Impl {
    GgufFile gguf;
    std::string lastErr;
    int nLayers = 36, dim = 4096, kvDim = 1024, ffn = 12288, vocab = 200000;
    int heads = 32, kvHeads = 8, hd = 128, maxSeq = 160;
    float eps = 1e-6f, theta = 1.0e6f;

    struct Layer {
        const GgufTensor *q, *k, *v, *o, *gate, *up, *down, *inNorm, *postNorm, *qNorm, *kNorm;
    };
    std::vector<Layer> layers;
    const GgufTensor *embed = nullptr, *finalNorm = nullptr, *lmHead = nullptr;

    std::vector<uint8_t> stage;
    BufRef stageDev;
    uint64_t stageBytes = 0;
    // One persistent device slot per matrix of the current layer. uploadNow inside a step writes
    // into a staging arena that is only reset per step, so large streamed weights must be uploaded
    // *between* steps into their own buffers.
    BufRef slot[7];

    BufRef x, xn, qb, kb, vb, attn, tmp, gateUp, swiglu, logits, cosTab, sinTab, scores, probs;
    BufRef kCache, vCache;      // [stream][layer][maxSeq][kvDim]
    BufRef inNormDev[2], postNormDev[2], qnDev, knDev, fnDev;
    std::vector<float> qnH, knH, fnH;

    const GgufTensor* find(const std::string& n) const { return gguf.find(n); }

    BufRef stageLoad(const GgufTensor* t, uint32_t& type) {
        if (!gguf.readRaw(*t, stage)) {
            std::fprintf(stderr, "lm: raw read failed for %s\n", t->name.c_str());
            std::abort();
        }
        if (stage.size() > stageBytes) {
            std::fprintf(stderr, "lm: staging overflow %zu > %llu\n", stage.size(),
                         (unsigned long long)stageBytes);
            std::abort();
        }
        G.eng.uploadNow(BufRef{stageDev.buf, stageDev.offset, stage.size()}, stage.data(), stage.size());
        type = (uint32_t)t->type;
        return stageDev;
    }

    // Stages `rows` rows of a large matrix (rowStart) and returns the device buffer.
    BufRef stageRows(const GgufTensor* t, uint64_t rowStart, uint64_t rows, uint32_t& type) {
        const uint64_t rowBytes = t->nbytes / (uint64_t)t->shape().first;
        const uint64_t bytes = rowBytes * rows;
        stage.resize((size_t)bytes);
        if (!gguf.readRange(*t, rowStart * rowBytes, stage.data(), bytes)) {
            std::fprintf(stderr, "lm: range read failed for %s\n", t->name.c_str());
            std::abort();
        }
        G.eng.uploadNow(BufRef{stageDev.buf, stageDev.offset, bytes}, stage.data(), bytes);
        type = (uint32_t)t->type;
        return stageDev;
    }

    // Uploads the 7 matrices of one layer into their slots. Must run outside beginStep/endStep.
    void uploadLayerWeights(int l) {
        auto& L = layers[(size_t)l];
        const GgufTensor* mats[7] = {L.q, L.k, L.v, L.o, L.gate, L.up, L.down};
        for (int i = 0; i < 7; i++) {
            if (!gguf.readRaw(*mats[i], stage)) {
                std::fprintf(stderr, "lm: raw read failed for %s\n", mats[i]->name.c_str());
                std::abort();
            }
            G.eng.uploadNow(slot[i], stage.data(), stage.size());
        }
    }

    void embedRow(int id, float* out) {
        const uint64_t rowBytes = (uint64_t)(dim / 256) * blockBytes(embed->type);
        std::vector<uint8_t> row((size_t)rowBytes);
        if (!gguf.readRange(*embed, (uint64_t)id * rowBytes, row.data(), rowBytes)) {
            std::fprintf(stderr, "lm: embed row read failed\n");
            std::abort();
        }
        dequantRow((uint32_t)embed->type, row.data(), (size_t)dim, out);
    }

    static uint64_t blockBytes(GgmlType t) {
        switch (t) {
            case GgmlType::Q4_K: return 144;
            case GgmlType::Q5_K: return 176;
            case GgmlType::Q6_K: return 210;
            default: return 0;
        }
    }
};

// Diagnostic: reads and dequantises one embedding row, printing statistics.
void saLmEmbedTest(const std::string& ggufPath, int id, std::string& err) {
    GgufFile g;
    if (!g.open(ggufPath, &err)) return;
    const GgufTensor* t = g.find("model.embed_tokens.weight");
    if (!t) {
        err = "no embed tensor";
        return;
    }
    const uint64_t rows = (uint64_t)t->shape().first;
    const uint64_t rowBytes = t->nbytes / rows;
    std::vector<uint8_t> row((size_t)rowBytes);
    if (!g.readRange(*t, (uint64_t)id * rowBytes, row.data(), rowBytes)) {
        err = "readRange failed";
        return;
    }
    std::vector<float> out(4096);
    dequantRow((uint32_t)t->type, row.data(), 4096, out.data());
    double ss = 0;
    float mn = out[0], mx = out[0];
    for (float v : out) {
        ss += (double)v * v;
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    std::printf("embed[%d] type %u: rms %.5f min %.4f max %.4f first %.5f %.5f\n", id, (unsigned)t->type,
                std::sqrt(ss / 4096.0), mn, mx, out[0], out[1]);
}

SaLm::~SaLm() { delete impl_; }

const std::string& SaLm::error() const { return impl_->lastErr; }

std::vector<int> SaLm::encode(const std::string& text) const {
    (void)text;
    return {};
}

bool SaLm::load(const std::string& ggufPath, const std::string& spvDir, std::string& err) {
    auto* m = new Impl();
    impl_ = m;
    std::string e;
    if (!m->gguf.open(ggufPath, &e)) {
        err = "cannot open LM GGUF " + ggufPath + ": " + e;
        return false;
    }
    if (const GgufMeta* cfg = m->gguf.meta("mm3.config_json")) {
        const std::string& s = cfg->s;
        auto num = [&](const char* key, int def) {
            size_t p = s.find(std::string("\"") + key + "\"");
            if (p == std::string::npos) return def;
            p = s.find(':', p);
            return p == std::string::npos ? def : std::atoi(s.c_str() + p + 1);
        };
        auto fnum = [&](const char* key, float def) {
            size_t p = s.find(std::string("\"") + key + "\"");
            if (p == std::string::npos) return def;
            p = s.find(':', p);
            return p == std::string::npos ? def : (float)std::atof(s.c_str() + p + 1);
        };
        m->dim = num("hidden_size", 4096);
        m->ffn = num("intermediate_size", 12288);
        m->heads = num("num_attention_heads", 32);
        m->kvHeads = num("num_key_value_heads", 8);
        m->nLayers = num("num_hidden_layers", 36);
        m->vocab = num("vocab_size", 200000);
        m->hd = num("head_dim", 128);
        m->theta = fnum("rope_theta", 1.0e6f);
        m->eps = fnum("rms_norm_eps", 1e-6f);
    }
    m->kvDim = m->kvHeads * m->hd;
    auto need = [&](const std::string& n) -> const GgufTensor* {
        const GgufTensor* t = m->gguf.find(n);
        if (!t) err = "LM GGUF is missing " + n;
        return t;
    };
    m->embed = need("model.embed_tokens.weight");
    m->finalNorm = need("model.norm.weight");
    m->lmHead = need("lm_head.weight");
    if (!m->embed || !m->finalNorm || !m->lmHead) return false;
    m->layers.resize((size_t)m->nLayers);
    for (int i = 0; i < m->nLayers; i++) {
        auto& L = m->layers[(size_t)i];
        const std::string p = "model.layers." + std::to_string(i) + ".";
        L.q = need(p + "self_attn.q_proj.weight");
        L.k = need(p + "self_attn.k_proj.weight");
        L.v = need(p + "self_attn.v_proj.weight");
        L.o = need(p + "self_attn.o_proj.weight");
        L.gate = need(p + "mlp.gate_proj.weight");
        L.up = need(p + "mlp.up_proj.weight");
        L.down = need(p + "mlp.down_proj.weight");
        L.inNorm = need(p + "input_layernorm.weight");
        L.postNorm = need(p + "post_attention_layernorm.weight");
        L.qNorm = need(p + "self_attn.q_norm.weight");
        L.kNorm = need(p + "self_attn.k_norm.weight");
        if (!L.q || !L.k || !L.v || !L.o || !L.gate || !L.up || !L.down || !L.inNorm || !L.postNorm ||
            !L.qNorm || !L.kNorm)
            return false;
    }
    // The lm_head/embedding tables are far too large to stage at once (672 MB), so only one
    // 64 MB row block is resident; the vocab projection runs block by block.
    uint64_t big = 64ull << 20;
    for (int i = 0; i < m->nLayers; i++) {
        auto& L = m->layers[(size_t)i];
        for (const GgufTensor* t : {L.q, L.k, L.v, L.o, L.gate, L.up, L.down})
            big = std::max<uint64_t>(big, t->nbytes);
    }
    (void)0;
    m->stageBytes = big;
    m->stage.reserve((size_t)big);
    if (!G.eng.alive()) {
        G.allocGradArena = false;
        G.stepBytes = 64ull << 20;
        G.scratchBytes = 64ull << 20;
        G.paramBytes = 64ull << 20;
        ofc::gpu::gInit(spvDir);
    }
    G.eng.initArenas(64ull << 20, 64ull << 20);
    // KV cache + weight staging live in the persistent scratch arena (gpu_ops owns it)
    ofc::gpu::gResizeScratch(384ull << 20);
    gpu::Arena& pers = G.scratchArena;
    G.paramArena.destroy();
    G.paramArena.init(G.eng.device(), G.eng.physDevice(), 64ull << 20, true, false);

    gpu::Arena& arena = G.eng.stepArena();
    auto alloc = [&](uint64_t floats) { return arena.alloc(align256(floats * 4)); };
    auto persAlloc = [&](uint64_t floats) { return pers.alloc(align256(floats * 4)); };
    const int D = m->dim, F = m->ffn, KV = m->kvDim;
    m->x = persAlloc((uint64_t)2 * D);
    m->xn = persAlloc((uint64_t)2 * D);
    m->qb = persAlloc((uint64_t)2 * D);
    m->kb = persAlloc((uint64_t)2 * KV);
    m->vb = persAlloc((uint64_t)2 * KV);
    m->attn = persAlloc((uint64_t)2 * D);
    m->tmp = persAlloc((uint64_t)2 * D);
    m->gateUp = persAlloc((uint64_t)2 * 2 * F);
    m->swiglu = persAlloc((uint64_t)2 * F);
    m->logits = persAlloc((uint64_t)2 * m->vocab);
    m->scores = persAlloc((uint64_t)m->heads * m->maxSeq);
    m->probs = persAlloc((uint64_t)m->heads * m->maxSeq);
    m->cosTab = persAlloc((uint64_t)m->maxSeq * (m->hd / 2));
    m->sinTab = persAlloc((uint64_t)m->maxSeq * (m->hd / 2));
    m->stageDev = persAlloc(m->stageBytes / 4 + 64);
    {
        auto& L0 = m->layers[0];
        const GgufTensor* mats[7] = {L0.q, L0.k, L0.v, L0.o, L0.gate, L0.up, L0.down};
        for (int i = 0; i < 7; i++) m->slot[i] = persAlloc(mats[i]->nbytes / 4 + 8);
    }
    const uint64_t kvFloats = (uint64_t)2 * m->nLayers * m->maxSeq * KV;
    m->kCache = persAlloc(kvFloats);
    m->vCache = persAlloc(kvFloats);
    m->qnH = m->gguf.readFloat("model.layers.0.self_attn.q_norm.weight");
    m->knH = m->gguf.readFloat("model.layers.0.self_attn.k_norm.weight");
    m->fnH = m->gguf.readFloat("model.norm.weight");
    m->qnDev = persAlloc(m->qnH.size());
    m->knDev = persAlloc(m->knH.size());
    m->fnDev = persAlloc(m->fnH.size());
    G.eng.uploadNow(m->qnDev, m->qnH.data(), m->qnH.size() * 4);
    G.eng.uploadNow(m->knDev, m->knH.data(), m->knH.size() * 4);
    G.eng.uploadNow(m->fnDev, m->fnH.data(), m->fnH.size() * 4);
    m->inNormDev[0] = persAlloc((size_t)D);
    m->inNormDev[1] = persAlloc((size_t)D);
    m->postNormDev[0] = persAlloc((size_t)D);
    m->postNormDev[1] = persAlloc((size_t)D);

    std::fprintf(stderr, "lm: arenas step %.0f MB, scratch %.0f MB, param %.0f MB\n",
                G.eng.stepArena().cap() / 1048576.0, G.scratchArena.cap() / 1048576.0,
                G.paramArena.cap() / 1048576.0);
    std::fprintf(stderr, "lm: %d layers, dim %d, kv %d, ffn %d, vocab %d, theta %.0f, staging %.0f MB\n",
                m->nLayers, m->dim, m->kvDim, m->ffn, m->vocab, m->theta, m->stageBytes / 1048576.0);
    return true;
}

bool SaLm::generate(const std::vector<int>& prompt, int frames, uint32_t seed, float cfg, int topK,
                    std::vector<int>& semantic, std::vector<float>& hidden, std::string& err) {
    auto* m = impl_;
    if (!m) {
        err = "lm: not loaded";
        return false;
    }
    const int D = m->dim, F = m->ffn, KV = m->kvDim, H = m->heads, KHV = m->kvHeads, HD = m->hd;
    const int P = (int)prompt.size();
    if (P <= 0 || P + frames + 2 > m->maxSeq) {
        err = "lm: prompt + frames exceeds the context";
        return false;
    }
    {
        std::vector<float> cs((size_t)m->maxSeq * (HD / 2)), sn((size_t)m->maxSeq * (HD / 2));
        for (int p = 0; p < m->maxSeq; p++)
            for (int i = 0; i < HD / 2; i++) {
                float inv = 1.0f / std::pow(m->theta, (float)(2 * i) / (float)HD);
                float a = (float)p * inv;
                cs[(size_t)p * (HD / 2) + i] = std::cos(a);
                sn[(size_t)p * (HD / 2) + i] = std::sin(a);
            }
        G.eng.uploadNow(m->cosTab, cs.data(), cs.size() * 4);
        G.eng.uploadNow(m->sinTab, sn.data(), sn.size() * 4);
    }

    std::vector<float> emb((size_t)2 * D);
    std::vector<float> logitsHost((size_t)2 * m->vocab);
    std::vector<float> hiddenRow((size_t)D);
    semantic.clear();
    hidden.clear();
    std::mt19937 rng(seed);

    auto softmaxSample = [&](const float* cond, const float* uncond) -> int {
        // mask to the semantic range plus the end-of-audio token
        std::vector<int> idx;
        idx.reserve((size_t)kSemanticVocab);
        for (int i = 0; i < kSemanticVocab; i++) idx.push_back(kTokSemanticBase + i);
        idx.push_back(kTokAudioEnd);
        float cBest = -1e30f;
        for (int v : idx) cBest = std::max(cBest, cond[v]);
        // top-k ranked on the conditional branch
        std::vector<int> keep = idx;
        int k = std::min<int>(topK, (int)keep.size());
        std::partial_sort(keep.begin(), keep.begin() + k, keep.end(),
                          [&](int a, int b) { return cond[a] > cond[b]; });
        keep.resize((size_t)k);
        std::vector<float> g((size_t)k);
        float mx = -1e30f;
        for (int i = 0; i < k; i++) {
            int v = keep[(size_t)i];
            g[(size_t)i] = uncond[v] + (cond[v] - uncond[v]) * cfg;
            mx = std::max(mx, g[(size_t)i]);
        }
        double sum = 0;
        for (int i = 0; i < k; i++) {
            g[(size_t)i] = std::exp(g[(size_t)i] - mx);
            sum += g[(size_t)i];
        }
        std::uniform_real_distribution<double> u(0.0, 1.0);
        double r = u(rng) * sum;
        int pick = 0;
        double acc = 0;
        for (int i = 0; i < k; i++) {
            acc += g[(size_t)i];
            if (acc >= r) {
                pick = i;
                break;
            }
        }
        return keep[(size_t)pick];
    };


    // ---- prefill + sampling
    auto tokenEmb = [&](int id, float* dst, double scale) {
        std::vector<float> e((size_t)D);
        m->embedRow(id, e.data());
        for (int i = 0; i < D; i++) dst[i] = (float)(e[(size_t)i] * scale);
    };
    auto runForward = [&](int pos) {
        // outside the step graph: the one-shot upload buffer is reused per call
        G.eng.uploadNow(m->x, emb.data(), emb.size() * 4);
        struct {
            uint32_t rows, d;
            float eps;
        } rp{(uint32_t)2, (uint32_t)D, m->eps};
        for (int l = 0; l < m->nLayers; l++) {
            auto& L = m->layers[(size_t)l];
            G.eng.endStep();
            G.eng.waitStep();
            m->uploadLayerWeights(l);
            {
                std::vector<float> w0 = m->gguf.readFloat(L.inNorm->name);
                std::vector<float> w1 = m->gguf.readFloat(L.postNorm->name);
                G.eng.uploadNow(m->inNormDev[0], w0.data(), w0.size() * 4);
                G.eng.uploadNow(m->postNormDev[0], w1.data(), w1.size() * 4);
            }
            G.eng.beginStep();
            const uint32_t type = (uint32_t)L.q->type;
            G.eng.dispatch("sa_rmsnorm", {m->x, m->inNormDev[0], m->xn}, &rp, 2, 1, 1);
            gemmKq(m->xn, m->slot[0], m->qb, 2, (uint32_t)D, (uint32_t)D, type);
            gemmKq(m->xn, m->slot[1], m->kb, 2, (uint32_t)KV, (uint32_t)D, type);
            gemmKq(m->xn, m->slot[2], m->vb, 2, (uint32_t)KV, (uint32_t)D, type);
            struct {
                uint32_t rows, d;
                float eps;
            } qp{(uint32_t)(2 * H), (uint32_t)HD, m->eps}, kp{(uint32_t)(2 * KHV), (uint32_t)HD, m->eps};
            G.eng.dispatch("sa_rmsnorm", {m->qb, m->qnDev, m->qb}, &qp, (uint32_t)(2 * H), 1, 1);
            G.eng.dispatch("sa_rmsnorm", {m->kb, m->knDev, m->kb}, &kp, (uint32_t)(2 * KHV), 1, 1);
            for (int which = 0; which < 2; which++) {
                struct {
                    uint32_t seq, heads, hd, rot, rowStride, base;
                } pc{(uint32_t)2, (uint32_t)(which ? KHV : H), (uint32_t)HD, (uint32_t)HD,
                     (uint32_t)(which ? KV : D), 0};
                BufRef buf = which ? m->kb : m->qb;
                BufRef cs = BufRef{m->cosTab.buf, m->cosTab.offset + (uint64_t)pos * (HD / 2) * 4,
                                   m->cosTab.size - (uint64_t)pos * (HD / 2) * 4};
                BufRef sn = BufRef{m->sinTab.buf, m->sinTab.offset + (uint64_t)pos * (HD / 2) * 4,
                                   m->sinTab.size - (uint64_t)pos * (HD / 2) * 4};
                uint32_t n = (uint32_t)(2 * pc.heads * (HD / 2));
                G.eng.dispatch("sa_rope", {buf, cs, sn}, &pc, (n + 255) / 256, 1, 1);
            }
            for (int s = 0; s < 2; s++) {
                uint64_t base = ((uint64_t)s * m->nLayers + l) * m->maxSeq * KV + (uint64_t)pos * KV;
                G.eng.copyBuf(BufRef{m->kCache.buf, m->kCache.offset + base * 4, (uint64_t)KV * 4},
                              BufRef{m->kb.buf, m->kb.offset + (uint64_t)s * KV * 4, (uint64_t)KV * 4});
                G.eng.copyBuf(BufRef{m->vCache.buf, m->vCache.offset + base * 4, (uint64_t)KV * 4},
                              BufRef{m->vb.buf, m->vb.offset + (uint64_t)s * KV * 4, (uint64_t)KV * 4});
            }
            const int LEN = pos + 1;
            for (int s = 0; s < 2; s++) {
                uint64_t kvBase = ((uint64_t)s * m->nLayers + l) * m->maxSeq * KV;
                for (int g = 0; g < KHV; g++) {
                    BufRef qs = BufRef{m->qb.buf, m->qb.offset + ((uint64_t)s * D + (uint64_t)g * 4 * HD) * 4,
                                       m->qb.size - ((uint64_t)s * D + (uint64_t)g * 4 * HD) * 4};
                    BufRef ks = BufRef{m->kCache.buf, m->kCache.offset + (kvBase + (uint64_t)g * HD) * 4,
                                       m->kCache.size - (kvBase + (uint64_t)g * HD) * 4};
                    BufRef sc = BufRef{m->scores.buf, m->scores.offset + (uint64_t)(g * 4) * LEN * 4,
                                       m->scores.size - (uint64_t)(g * 4) * LEN * 4};
                    gemmF32(qs, ks, sc, 1, (uint32_t)LEN, (uint32_t)HD, 4, (uint32_t)HD, 0, (uint32_t)LEN,
                            (uint32_t)D, (uint32_t)KV, (uint32_t)LEN, true);
                }
                struct {
                    uint32_t n;
                    float s;
                } pc{(uint32_t)(H * LEN), 1.0f / std::sqrt((float)HD)};
                G.eng.dispatch("sa_scale", {m->scores, m->probs}, &pc, (uint32_t)((H * LEN + 255) / 256), 1, 1);
                struct {
                    uint32_t rows, cols;
                } sp{(uint32_t)H, (uint32_t)LEN};
                G.eng.dispatch("softmax_fwd", {m->probs, m->scores}, &sp, (uint32_t)H, 1, 1);
                for (int g = 0; g < KHV; g++) {
                    BufRef ps = BufRef{m->scores.buf, m->scores.offset + (uint64_t)(g * 4) * LEN * 4,
                                       m->scores.size - (uint64_t)(g * 4) * LEN * 4};
                    BufRef vs = BufRef{m->vCache.buf, m->vCache.offset + (kvBase + (uint64_t)g * HD) * 4,
                                       m->vCache.size - (kvBase + (uint64_t)g * HD) * 4};
                    BufRef as = BufRef{m->attn.buf,
                                       m->attn.offset + ((uint64_t)s * D + (uint64_t)g * 4 * HD) * 4,
                                       m->attn.size - ((uint64_t)s * D + (uint64_t)g * 4 * HD) * 4};
                    gemmF32(ps, vs, as, 1, (uint32_t)HD, (uint32_t)LEN, 4, (uint32_t)LEN, 0, (uint32_t)HD,
                            (uint32_t)LEN, (uint32_t)KV, (uint32_t)D, false);
                }
            }
            gemmKq(m->attn, m->slot[3], m->tmp, 2, (uint32_t)D, (uint32_t)D, type);
            {
                struct {
                    uint32_t n, pad;
                } ap{(uint32_t)(2 * D), 0};
                G.eng.dispatch("add2", {m->x, m->x, m->tmp}, &ap, (uint32_t)(((uint64_t)2 * D + 255) / 256), 1, 1);
            }
            G.eng.dispatch("sa_rmsnorm", {m->x, m->postNormDev[0], m->xn}, &rp, 2, 1, 1);
            gemmKq(m->xn, m->slot[5],
                   BufRef{m->gateUp.buf, m->gateUp.offset, (uint64_t)2 * F * 4}, 2, (uint32_t)F,
                   (uint32_t)D, type);
            gemmKq(m->xn, m->slot[4],
                   BufRef{m->gateUp.buf, m->gateUp.offset + (uint64_t)2 * F * 4, (uint64_t)2 * 2 * F * 4}, 2,
                   (uint32_t)F, (uint32_t)D, type);
            for (int s = 0; s < 2; s++) {
                struct {
                    uint32_t rows, half_;
                } sg{1u, (uint32_t)F};
                G.eng.dispatch("sa_swiglu",
                               {BufRef{m->gateUp.buf, m->gateUp.offset + (uint64_t)s * 2 * F * 4,
                                       (uint64_t)2 * F * 4},
                                BufRef{m->swiglu.buf, m->swiglu.offset + (uint64_t)s * F * 4,
                                       (uint64_t)F * 4}},
                               &sg, (uint32_t)((F + 255) / 256), 1, 1);
            }
            gemmKq(m->swiglu, m->slot[6], m->tmp, 2, (uint32_t)D, (uint32_t)F, type);
            {
                struct {
                    uint32_t n, pad;
                } ap{(uint32_t)(2 * D), 0};
                G.eng.dispatch("add2", {m->x, m->x, m->tmp}, &ap, (uint32_t)(((uint64_t)2 * D + 255) / 256), 1, 1);
            }
        }
        G.eng.dispatch("sa_rmsnorm", {m->x, m->fnDev, m->xn}, &rp, 2, 1, 1);
        {
            const uint64_t rowBytes = m->lmHead->nbytes / (uint64_t)m->lmHead->shape().first;
            uint64_t rows = std::max<uint64_t>(1, (48ull << 20) / rowBytes);
            for (int v0 = 0; v0 < m->vocab; v0 += (int)rows) {
                int nb = (int)std::min<uint64_t>(rows, (uint64_t)(m->vocab - v0));
                uint32_t t2 = 0;
                BufRef w = m->stageRows(m->lmHead, (uint64_t)v0, (uint64_t)nb, t2);
                BufRef dst = BufRef{m->logits.buf, m->logits.offset + (uint64_t)v0 * 2 * 4,
                                    m->logits.size - (uint64_t)v0 * 2 * 4};
                // rows are 2 tokens wide in the destination, so the GEMM writes strided output
                PushGemm pc{2u, (uint32_t)nb, (uint32_t)D, 0, 0, 0, 0, 1, 0, 0, (uint32_t)(2 * m->vocab), t2};
                G.eng.dispatch("gemm_nt_q4k", {m->xn, w, dst}, &pc, ((uint32_t)nb + 63) / 64, 1, 1);
            }
        }
        G.eng.endStep();
        G.eng.waitStep();
        G.eng.downloadNow(m->logits, logitsHost.data(), logitsHost.size() * 4);
        G.eng.downloadNow(BufRef{m->xn.buf, m->xn.offset, (uint64_t)D * 4}, hiddenRow.data(),
                          hiddenRow.size() * 4);
    };
    for (int i = 0; i < P; i++) {
        if (i == 0)
        tokenEmb(prompt[(size_t)i], &emb[0], 1.0);
        int uncondId = (i == 0 || i >= P - 3) ? prompt[(size_t)i] : kTokAudioCfg;
        tokenEmb(uncondId, &emb[(size_t)D], 1.0);
        runForward(i);

    }
    // sampling loop: one semantic token per frame
    for (int f = 0; f < frames; f++) {
        int tok = softmaxSample(logitsHost.data(), logitsHost.data() + m->vocab);
        if (tok == kTokAudioEnd) break;
        semantic.push_back(tok - kTokSemanticBase);
        hidden.insert(hidden.end(), hiddenRow.begin(), hiddenRow.end());
        tokenEmb(tok, &emb[0], 1.0);
        tokenEmb(tok, &emb[(size_t)D], 1.0);
        runForward(P + f);
    }
    return !semantic.empty();
}
}  // namespace audio
}  // namespace ofc
