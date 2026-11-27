// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
#include "SaVae.hpp"

#include "Modules/Diffusion/Safetensors.hpp"
#include "Modules/GPU/gpu_ops.hpp"
#include "Modules/GPU/vk_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdlib>
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

void gemmNn(const BufRef& a, const BufRef& b, const BufRef& c, uint32_t M, uint32_t N, uint32_t K,
            uint32_t batch = 1) {
    PushGemm pc{M, N, K, 0, 0, 0, 0, batch, 0, 0, 0, 0};
    G.eng.dispatch("gemm_nn", {a, b, c}, &pc, (N + 63) / 64, (M + 63) / 64, batch);
}

}  // namespace

struct SaVae::Buf {
    struct Conv {
        int cin = 0, cout = 0, k = 0, stride = 1, pad = 0, dil = 1;
        bool transposed = false;
        BufRef w, bias;
    };
    struct ResUnit {
        int ch = 0, dil = 1;
        Conv c1, c2;
        BufRef a1, a2;
    };
    struct Block {
        int cin = 0, cout = 0, stride = 1;
        BufRef snake;
        Conv ct;
        ResUnit res[3];
    };
    Conv decInProj, convIn, convOut;
    BufRef snakeOut;
    Block blocks[4];

    Safetensors st;
    uint64_t paramBytes = 0;
};

namespace {

// Reads a conv weight, folding weight norm (g * v / ||v|| over the in/kernel axes) and
// transposing it into the GEMM layout: conv -> [cout, cin*k], convT -> [cout*k, cin].
BufRef loadConv(SaVae::Buf& m, const std::string& prefix, int cout, int cin, int k, bool transposed,
                BufRef& biasOut) {
    const STTensor* v = m.st.find(prefix + ".weight_v");
    const STTensor* w = m.st.find(prefix + ".weight");
    const STTensor* g = m.st.find(prefix + ".weight_g");
    std::vector<float> src;
    if (v) {
        src = m.st.readF32(v->name);
    } else if (w) {
        src = m.st.readF32(w->name);
    } else {
        std::fprintf(stderr, "vae: missing weight %s\n", prefix.c_str());
        std::abort();
    }
    // A transposed conv stores [cin, cout, k] and its weight_norm g has one entry per *input*
    // channel; a plain conv stores [cout, cin, k] with g per output channel.
    const int rows = transposed ? cin : cout;
    const int rowLen = transposed ? cout * k : cin * k;
    std::vector<float> wv(src.size());
    if (g) {
        std::vector<float> gv = m.st.readF32(g->name);
        if ((int)gv.size() != rows) {
            std::fprintf(stderr, "vae: weight_norm size mismatch at %s\n", prefix.c_str());
            std::abort();
        }
        for (int r = 0; r < rows; r++) {
            const float* row = &src[(size_t)r * rowLen];
            double ss = 0;
            for (int i = 0; i < rowLen; i++) ss += (double)row[i] * row[i];
            float inv = (float)(1.0 / std::max(1e-12, std::sqrt(ss)));
            for (int i = 0; i < rowLen; i++) wv[(size_t)r * rowLen + i] = row[i] * inv * gv[(size_t)r];
        }
    } else {
        wv = src;
    }
    std::vector<float> out((size_t)cout * cin * k);
    if (transposed) {
        for (int ci = 0; ci < cin; ci++)
            for (int co = 0; co < cout; co++)
                for (int ki = 0; ki < k; ki++)
                    out[((size_t)co * k + ki) * cin + ci] = wv[((size_t)ci * cout + co) * k + ki];
    } else {
        out = wv;
    }
    BufRef dev = G.paramArena.alloc(align256(out.size() * 4));
    G.eng.uploadNow(dev, out.data(), out.size() * 4);
    if (const STTensor* b = m.st.find(prefix + ".bias")) {
        std::vector<float> bv = m.st.readF32(b->name);
        biasOut = G.paramArena.alloc(align256(bv.size() * 4));
        G.eng.uploadNow(biasOut, bv.data(), bv.size() * 4);
    } else {
        biasOut = BufRef{};
    }
    return dev;
}

BufRef loadVector(SaVae::Buf& m, const std::string& name) {
    std::vector<float> v = m.st.readF32(name);
    BufRef d = G.paramArena.alloc(align256(v.size() * 4));
    G.eng.uploadNow(d, v.data(), v.size() * 4);
    return d;
}

struct Ctx {
    SaVae::Buf* m;

    BufRef alloc(uint64_t floats) { return G.eng.stepArena().alloc(align256(floats * 4)); }
    uint64_t mark() const { return G.eng.stepArena().used(); }
    const char* dumpDir = nullptr;
    void dump(const char* name, const BufRef& b, int C, int L) {
        if (!dumpDir) return;
        std::vector<float> host((size_t)C * L);
        G.eng.downloadNow(b, host.data(), host.size() * 4);
        char path[512];
        std::snprintf(path, sizeof path, "%s_%s.f32", dumpDir, name);
        if (FILE* f = std::fopen(path, "wb")) {
            std::fwrite(host.data(), 4, host.size(), f);
            std::fclose(f);
        }
    }
    void rewind(uint64_t m) { G.eng.stepArena().rewind(m); }

    BufRef snake(const BufRef& x, int C, int L, const BufRef& alpha) {
        BufRef y = alloc((uint64_t)C * L);
        struct {
            uint32_t C, L;
        } pc{(uint32_t)C, (uint32_t)L};
        G.eng.dispatch("sa_snake", {x, alpha, y}, &pc, (uint32_t)(((uint64_t)C * L + 255) / 256), 1, 1);
        return y;
    }

    void biasAdd(const BufRef& y, const BufRef& b, int C, int L) {
        if (b.buf == VK_NULL_HANDLE) return;
        struct {
            uint32_t C, P;
        } pc{(uint32_t)C, (uint32_t)L};
        G.eng.dispatch("bias_add", {y, b}, &pc, (uint32_t)(((uint64_t)C * L + 255) / 256), 1, 1);
    }

    // out[cout, O], O = (L + 2p - d(k-1) - 1)/s + 1
    // Banded so the im2col block stays around 32 MB: at 44.1 kHz the late blocks would otherwise
    // need C*K*O floats for the whole window (96*7*352768 for a 689-frame one).
    BufRef conv(const BufRef& x, int C, int L, const SaVae::Buf::Conv& c) {
        int O = (L + 2 * c.pad - c.dil * (c.k - 1) - 1) / c.stride + 1;
        if (O <= 0) O = 1;
        BufRef y = alloc((uint64_t)c.cout * O);
        const uint64_t budget = 8ull << 20;  // floats per band (~32 MB)
        int band = (int)std::max<uint64_t>(1, std::min<uint64_t>((uint64_t)O, budget / ((uint64_t)C * c.k)));
        for (int o0 = 0; o0 < O; o0 += band) {
            int nb = std::min(band, O - o0);
            uint64_t m = mark();
            BufRef col = alloc((uint64_t)C * c.k * nb);
            struct {
                uint32_t C, L, K, O, stride;
                int32_t pad;
                uint32_t dil;
                uint32_t o0;
            } pc{(uint32_t)C, (uint32_t)L, (uint32_t)c.k, (uint32_t)nb, (uint32_t)c.stride, c.pad,
                 (uint32_t)c.dil, (uint32_t)o0};
            // the band shifts the output origin: pass a padded shift through pad + o0*stride
            pc.pad = c.pad - o0 * c.stride;
            G.eng.dispatch("sa_im2col1d", {x, col}, &pc, (uint32_t)(((uint64_t)C * c.k * nb + 255) / 256), 1, 1);
            // rows of the band are O_total apart in y, so the GEMM needs an explicit ldc
            BufRef ys = BufRef{y.buf, y.offset + (uint64_t)o0 * 4, y.size - (uint64_t)o0 * 4};
            PushGemm gpc{(uint32_t)c.cout, (uint32_t)nb, (uint32_t)(C * c.k), 0, 0, 0, 0, 1, 0, 0,
                         (uint32_t)O, 0};
            G.eng.dispatch("gemm_nn", {c.w, col, ys}, &gpc, ((uint32_t)nb + 63) / 64,
                           ((uint32_t)c.cout + 63) / 64, 1);
            rewind(m);
        }
        biasAdd(y, c.bias, c.cout, O);
        return y;
    }

    // out[cout, O], O = (L-1)*s - 2p + k
    BufRef convT(const BufRef& x, int C, int L, const SaVae::Buf::Conv& c) {
        BufRef tmp = alloc((uint64_t)c.cout * c.k * L);
        gemmNn(c.w, x, tmp, (uint32_t)(c.cout * c.k), (uint32_t)L, (uint32_t)C);
        int O = (L - 1) * c.stride - 2 * c.pad + c.k;
        BufRef y = alloc((uint64_t)c.cout * O);
        struct {
            uint32_t CO, K, I, O, stride;
            int32_t pad;
            uint32_t pad2;
        } pc{(uint32_t)c.cout, (uint32_t)c.k, (uint32_t)L, (uint32_t)O, (uint32_t)c.stride, c.pad, 0};
        G.eng.dispatch("sa_col2im1d", {tmp, y}, &pc, (uint32_t)(((uint64_t)c.cout * O + 255) / 256), 1, 1);
        biasAdd(y, c.bias, c.cout, O);
        return y;
    }
};

}  // namespace

SaVae::~SaVae() { delete static_cast<Buf*>(impl_); }

bool SaVae::load(const std::string& path, const std::string& spvDir, std::string& err) {
    auto* m = new Buf();
    impl_ = m;
    if (!m->st.open(path)) {
        err = "cannot open VAE " + path + ": " + m->st.error();
        return false;
    }
    uint64_t bytes = 0;
    for (const auto& t : m->st.tensors()) bytes += m->st.numel(t) * 4;
    m->paramBytes = bytes + (64ull << 20);
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

    m->decInProj = {64, 1024, 1, 1, 0, 1, false, BufRef{}, BufRef{}};
    m->decInProj.w = loadConv(*m, "dec_in_proj", 1024, 64, 1, false, m->decInProj.bias);

    m->convIn = {1024, 1536, 7, 1, 3, 1, false, BufRef{}, BufRef{}};
    m->convIn.w = loadConv(*m, "decoder.model.0", 1536, 1024, 7, false, m->convIn.bias);

    static const int strides[4] = {8, 8, 4, 2};
    static const int dims[5] = {1536, 768, 384, 192, 96};
    static const int dils[3] = {1, 3, 9};
    for (int b = 0; b < 4; b++) {
        const std::string bp = "decoder.model." + std::to_string(b + 1) + ".";
        auto& blk = m->blocks[b];
        blk.cin = dims[b];
        blk.cout = dims[b + 1];
        blk.stride = strides[b];
        blk.snake = loadVector(*m, bp + "block.0.alpha");
        blk.ct = {blk.cin, blk.cout, 2 * blk.stride, blk.stride, blk.stride / 2, 1, true, BufRef{}, BufRef{}};
        blk.ct.w = loadConv(*m, bp + "block.1", blk.cout, blk.cin, blk.ct.k, true, blk.ct.bias);
        for (int r = 0; r < 3; r++) {
            auto& ru = blk.res[r];
            const std::string rp = bp + "block." + std::to_string(r + 2) + ".";
            ru.ch = blk.cout;
            ru.dil = dils[r];
            ru.a1 = loadVector(*m, rp + "block.0.alpha");
            ru.a2 = loadVector(*m, rp + "block.2.alpha");
            ru.c1 = {ru.ch, ru.ch, 7, 1, 3 * ru.dil, ru.dil, false, BufRef{}, BufRef{}};
            ru.c1.w = loadConv(*m, rp + "block.1", ru.ch, ru.ch, 7, false, ru.c1.bias);
            ru.c2 = {ru.ch, ru.ch, 1, 1, 0, 1, false, BufRef{}, BufRef{}};
            ru.c2.w = loadConv(*m, rp + "block.3", ru.ch, ru.ch, 1, false, ru.c2.bias);
        }
    }
    m->snakeOut = loadVector(*m, "decoder.model.5.alpha");
    m->convOut = {96, 1, 7, 1, 3, 1, false, BufRef{}, BufRef{}};
    m->convOut.w = loadConv(*m, "decoder.model.6", 1, 96, 7, false, m->convOut.bias);
    std::printf("vae: decoder loaded (%.0f MB, arena %.0f MB)\n", bytes / 1048576.0, m->paramBytes / 1048576.0);
    return true;
}

bool SaVae::decode(const std::vector<float>& latent, int frames, std::vector<float>& stereo,
                   std::string& err) {
    auto* m = static_cast<Buf*>(impl_);
    if (!m) {
        err = "vae: not loaded";
        return false;
    }
    if ((int)latent.size() != 128 * frames) {
        err = "vae: latent size mismatch";
        return false;
    }
    const int T = frames;
    stereo.assign((size_t)2 * frames * kHop, 0.0f);
    for (int side = 0; side < 2; side++) {
        G.eng.beginStep();
        Ctx c{m};
        c.dumpDir = std::getenv("OFC_VAE_DUMP");
        BufRef x = c.alloc((uint64_t)64 * T);
        {
            std::vector<float> xh((size_t)64 * T);
            for (int t = 0; t < T; t++)
                for (int ch = 0; ch < 64; ch++)
                    xh[(size_t)ch * T + t] = latent[(size_t)t * 128 + side * 64 + ch];
            G.eng.uploadNow(x, xh.data(), xh.size() * 4);
        }
        BufRef y = c.conv(x, 64, T, m->decInProj);  // [1024, T]
        c.dump("dec_in_proj", y, 1024, T);
        y = c.conv(y, 1024, T, m->convIn);          // [1536, T]
        c.dump("conv_in", y, 1536, T);
        int C = 1536, L = T;
        for (int b = 0; b < 4; b++) {
            auto& blk = m->blocks[b];
            BufRef s = c.snake(y, C, L, blk.snake);
            y = c.convT(s, C, L, blk.ct);
            C = blk.cout;
            L = (L - 1) * blk.stride - 2 * (blk.stride / 2) + blk.ct.k;
            {
                char nm[32];
                std::snprintf(nm, sizeof nm, "blk%d_ct", b);
                c.dump(nm, y, C, L);
            }
            for (int r = 0; r < 3; r++) {
                auto& ru = blk.res[r];
                uint64_t m = c.mark();
                BufRef a = c.snake(y, C, L, ru.a1);
                BufRef z = c.conv(a, C, L, ru.c1);
                BufRef b2 = c.snake(z, C, L, ru.a2);
                BufRef o = c.conv(b2, C, L, ru.c2);
                struct {
                    uint32_t n, pad;
                } pc{(uint32_t)((uint64_t)C * L), 0};
                G.eng.dispatch("add2", {y, y, o}, &pc, (uint32_t)(((uint64_t)C * L + 255) / 256), 1, 1);
                {
                    char nm[32];
                    std::snprintf(nm, sizeof nm, "blk%d_res%d", b, r);
                    c.dump(nm, y, C, L);
                }
                c.rewind(m);
            }
        }
        {
            char nm[32];
            std::snprintf(nm, sizeof nm, "pre_out");
            c.dump(nm, y, C, L);
        }
        BufRef s = c.snake(y, C, L, m->snakeOut);
        BufRef o = c.conv(s, C, L, m->convOut);  // [1, L]
        c.dump("out_preTanh", o, 1, L);
        G.eng.endStep();
        G.eng.waitStep();
        std::vector<float> host((size_t)L);
        G.eng.downloadNow(o, host.data(), host.size() * 4);
        size_t n = std::min<size_t>((size_t)L, (size_t)frames * kHop);
        for (size_t i = 0; i < n; i++) stereo[i * 2 + side] = std::tanh(host[i]);
    }
    return true;
}

}  // namespace audio
}  // namespace ofc
