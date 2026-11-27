// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
// T5-base encoder for the Stable Audio text conditioner (CPU, fp32).
//
// The encoder is 12 layers of [RMSNorm -> self-attention with the 32-bucket relative
// position bias -> RMSNorm -> ReLU FFN], 64 tokens of context, no biases anywhere. At this
// size it costs a few seconds once per run, so it lives on the host and the GPU is kept for
// the DiT / Oobleck decoder.
#include "SaGen.hpp"
#include "SaTokenizer.hpp"

#include "Modules/Diffusion/Safetensors.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <unordered_map>

namespace ofc {
namespace audio {

namespace {

// c[M,N] = a[M,K] * w[N,K]^T, i.e. a linear layer whose weight is stored [out, in] (torch
// convention). Parallel over the M rows; each output row walks the weight rows in order so the
// reads stay sequential.
void linT(const float* a, const float* w, float* c, int M, int N, int K, int threads) {
    if (threads <= 1) threads = 1;
    auto block = [&](int m0, int m1) {
        for (int i = m0; i < m1; i++) {
            const float* ai = a + (size_t)i * K;
            float* ci = c + (size_t)i * N;
            for (int n = 0; n < N; n++) {
                const float* wn = w + (size_t)n * K;
                float acc = 0.0f;
                for (int k = 0; k < K; k++) acc += ai[k] * wn[k];
                ci[n] = acc;
            }
        }
    };
    int nThreads = std::min(threads, M);
    if (nThreads <= 1) {
        block(0, M);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve((size_t)nThreads);
    int per = (M + nThreads - 1) / nThreads;
    for (int t = 0; t < nThreads; t++) {
        int m0 = t * per, m1 = std::min(M, m0 + per);
        if (m0 >= m1) break;
        pool.emplace_back(block, m0, m1);
    }
    for (auto& th : pool) th.join();
}

void rmsNorm(const float* x, const float* w, float* y, int rows, int d, float eps) {
    for (int r = 0; r < rows; r++) {
        const float* xr = x + (size_t)r * d;
        float ss = 0.0f;
        for (int j = 0; j < d; j++) ss += xr[j] * xr[j];
        float inv = 1.0f / std::sqrt(ss / (float)d + eps);
        float* yr = y + (size_t)r * d;
        for (int j = 0; j < d; j++) yr[j] = xr[j] * inv * w[j];
    }
}

// HF T5 `_relative_position_bucket`, bidirectional: 16 buckets per direction, exact up to 8
// positions, then log-spaced up to max_distance (128).
int relativeBucket(int relPos, int numBuckets, int maxDistance) {
    int buckets = numBuckets / 2;
    int b = 0;
    int n = relPos > 0 ? relPos : -relPos;
    if (relPos > 0) b += buckets;
    const int maxExact = buckets / 2;
    if (n < maxExact) return b + n;
    float v = (float)maxExact + std::log((float)n / (float)maxExact) /
                                    std::log((float)maxDistance / (float)maxExact) * (float)(buckets - maxExact);
    int large = (int)v;
    if (large > buckets - 1) large = buckets - 1;
    return b + large;
}

struct T5Ctx {
    std::unordered_map<std::string, std::vector<float>> w;
    int layers = 12, d = 768, heads = 12, dkv = 64, ffn = 3072, seq = 64, buckets = 32, maxDist = 128;
};

bool loadT5Weights(const std::string& path, T5Ctx& t, std::string& err) {
    Safetensors st;
    if (!st.open(path)) {
        err = "cannot open " + path + ": " + st.error();
        return false;
    }
    static const char* needed[] = {"encoder.final_layer_norm.weight", "shared.weight",
                                   "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"};
    for (const char* n : needed) {
        if (!st.find(n)) {
            err = std::string("T5 checkpoint is missing ") + n;
            return false;
        }
    }
    for (const auto& ts : st.tensors()) {
        // only the encoder tower and the embedding table are needed
        if (ts.name.rfind("encoder.", 0) != 0 && ts.name != "shared.weight") continue;
        t.w[ts.name] = st.readF32(ts.name);
    }
    if (t.w["shared.weight"].size() != (size_t)32128 * 768) {
        err = "unexpected T5 vocabulary size";
        return false;
    }
    return true;
}

const std::vector<float>& W(const T5Ctx& t, const std::string& n) {
    auto it = t.w.find(n);
    if (it == t.w.end()) {
        std::fprintf(stderr, "missing T5 tensor %s\n", n.c_str());
        std::abort();
    }
    return it->second;
}

}  // namespace

bool saTextEncode(const SaParams& p, std::vector<float>& hidden, std::vector<int>& ids, std::string& err) {
    SpTokenizer tok;
    T5Ctx t;
    if (!tok.load(p.modelDir + "/t5-base/spiece.model", err)) return false;
    if (!loadT5Weights(p.modelDir + "/t5-base/model.safetensors", t, err)) return false;

    const int T = t.seq, D = t.d, H = t.heads, DK = t.dkv, F = t.ffn;
    int threads = p.threads > 0 ? p.threads : (int)std::thread::hardware_concurrency();

    ids = tok.encodePadded(p.prompt, T);
    // attention mask: everything up to and including EOS is valid
    std::vector<uint8_t> valid((size_t)T, 0);
    int nValid = 0;
    for (int i = 0; i < T; i++) {
        valid[(size_t)i] = 1;
        nValid = i + 1;
        if (ids[(size_t)i] == tok.eosId()) break;
    }

    // token embedding
    std::vector<float> x((size_t)T * D, 0.0f);
    const std::vector<float>& emb = W(t, "shared.weight");
    for (int i = 0; i < T; i++)
        std::memcpy(&x[(size_t)i * D], &emb[(size_t)ids[(size_t)i] * D], sizeof(float) * (size_t)D);

    // relative position bias per head, computed once (T x T x H), mask folded in
    std::vector<float> bias((size_t)T * T * H, 0.0f);
    {
        const std::vector<float>& rel =
            W(t, "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight");  // [32,12]
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                int b = relativeBucket(j - i, t.buckets, t.maxDist);
                for (int h = 0; h < H; h++) {
                    float v = rel[(size_t)b * H + h];
                    if (!valid[(size_t)j]) v = -1e30f;
                    bias[((size_t)i * T + j) * H + h] = v;
                }
            }
        }
    }

    std::vector<float> xn((size_t)T * D), qkv((size_t)T * 3 * D), q((size_t)T * D), k((size_t)T * D),
        v((size_t)T * D), s((size_t)T * T), probs((size_t)T * T), o((size_t)T * D), h((size_t)T * F);
    std::vector<float> attnOut((size_t)T * D);

    for (int l = 0; l < t.layers; l++) {
        const std::string lp = "encoder.block." + std::to_string(l) + ".layer.";
        rmsNorm(x.data(), W(t, lp + "0.layer_norm.weight").data(), xn.data(), T, D, 1e-6f);
        // q/k/v: the weights are [out, in] so this is x * W^T; transpose once into a K-major copy
        const std::vector<float>& wq = W(t, lp + "0.SelfAttention.q.weight");
        const std::vector<float>& wk = W(t, lp + "0.SelfAttention.k.weight");
        const std::vector<float>& wv = W(t, lp + "0.SelfAttention.v.weight");
        // c[m,n] = sum_k a[m,k] * bT[n,k] with bT = W (already [out, in]) -> pass W directly
        linT(xn.data(), wq.data(), q.data(), T, D, D, threads);
        linT(xn.data(), wk.data(), k.data(), T, D, D, threads);
        linT(xn.data(), wv.data(), v.data(), T, D, D, threads);

        const float scale = 1.0f / std::sqrt((float)DK);
        for (int hh = 0; hh < H; hh++) {
            const int off = hh * DK;
            for (int i = 0; i < T; i++) {
                const float* qi = q.data() + (size_t)i * D + off;
                float* si = s.data() + (size_t)i * T;
                for (int j = 0; j < T; j++) {
                    const float* kj = k.data() + (size_t)j * D + off;
                    float acc = 0.0f;
                    for (int d2 = 0; d2 < DK; d2++) acc += qi[d2] * kj[d2];
                    si[j] = acc * scale + bias[((size_t)i * T + j) * H + hh];
                }
                // softmax over j
                float mx = -1e30f;
                for (int j = 0; j < T; j++) mx = std::max(mx, si[j]);
                float sum = 0.0f;
                for (int j = 0; j < T; j++) {
                    si[j] = std::exp(si[j] - mx);
                    sum += si[j];
                }
                float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
                for (int j = 0; j < T; j++) si[j] *= inv;
                float* oi = o.data() + (size_t)i * D + off;
                for (int d2 = 0; d2 < DK; d2++) oi[d2] = 0.0f;
                for (int j = 0; j < T; j++) {
                    float pj = si[j];
                    if (pj == 0.0f) continue;
                    const float* vj = v.data() + (size_t)j * D + off;
                    for (int d2 = 0; d2 < DK; d2++) oi[d2] += pj * vj[d2];
                }
            }
        }
        if (const char* d = std::getenv("OFC_T5_DUMP")) {
            char path[512];
            std::snprintf(path, sizeof path, "%s_attn%d.f32", d, l);
            if (FILE* f = std::fopen(path, "wb")) { std::fwrite(o.data(), 4, o.size(), f); std::fclose(f); }
        }
        // output projection (W is [out, in] = [768,768]) and residual
        {
            std::vector<float> tmp((size_t)T * D);
            linT(o.data(), W(t, lp + "0.SelfAttention.o.weight").data(), tmp.data(), T, D, D, threads);
            for (size_t i = 0; i < x.size(); i++) x[i] += tmp[i];
        }
        if (const char* d = std::getenv("OFC_T5_DUMP")) {
            char path[512];
            std::snprintf(path, sizeof path, "%s_x%d.f32", d, l);
            if (FILE* f = std::fopen(path, "wb")) { std::fwrite(x.data(), 4, x.size(), f); std::fclose(f); }
        }
        rmsNorm(x.data(), W(t, lp + "1.layer_norm.weight").data(), xn.data(), T, D, 1e-6f);
        {
            std::vector<float> wi((size_t)T * F);
            linT(xn.data(), W(t, lp + "1.DenseReluDense.wi.weight").data(), wi.data(), T, F, D, threads);
            for (size_t i = 0; i < wi.size(); i++) wi[i] = std::max(0.0f, wi[i]);
            std::vector<float> tmp((size_t)T * D);
            linT(wi.data(), W(t, lp + "1.DenseReluDense.wo.weight").data(), tmp.data(), T, D, F, threads);
            for (size_t i = 0; i < x.size(); i++) x[i] += tmp[i];
        }
    }
    hidden.assign((size_t)T * D, 0.0f);
    rmsNorm(x.data(), W(t, "encoder.final_layer_norm.weight").data(), hidden.data(), T, D, 1e-6f);
    if (p.verbose)
        std::printf("t5: %d tokens (%d valid), rms %.4f\n", T, nValid,
                    std::sqrt([&] {
                        double s2 = 0;
                        for (float f : hidden) s2 += (double)f * f;
                        return s2 / (double)hidden.size();
                    }()));
    return true;
}

void saDumpT5(const SaParams& p, const std::string& outDir) {
    std::string err;
    std::vector<float> hidden;
    std::vector<int> ids;
    if (!saTextEncode(p, hidden, ids, err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return;
    }
    std::string dir = outDir;
    auto wr = [&](const std::string& name, const void* data, size_t bytes) {
        std::string path = dir + "/" + name;
        if (FILE* f = std::fopen(path.c_str(), "wb")) {
            std::fwrite(data, 1, bytes, f);
            std::fclose(f);
        }
    };
    wr("eng_ids.i32", ids.data(), ids.size() * sizeof(int));
    wr("eng_t5_last.f32", hidden.data(), hidden.size() * sizeof(float));
    std::printf("t5 dump: %zu ids, %zu floats -> %s\n", ids.size(), hidden.size(), dir.c_str());
}

}  // namespace audio
}  // namespace ofc
