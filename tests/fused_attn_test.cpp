// Numeric + latency check for the fused sdvk attention operator.
// Build: cmake --build build --target fused_attn_test
// Run:   ./fused_attn_test [spvDir]

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "Modules/GPU/fused_attn.hpp"
#include "Modules/GPU/vk_engine.hpp"

namespace {

struct Shape {
    const char* name;
    uint32_t Tq, Tk, C, heads;
    bool causal;
    bool check;
};

std::vector<float> reference(const std::vector<float>& q, const std::vector<float>& k,
                             const std::vector<float>& v, uint32_t Tq, uint32_t Tk, uint32_t C,
                             uint32_t heads, bool causal) {
    uint32_t hd = C / heads;
    float scale = 1.0f / std::sqrt((float)hd);
    std::vector<float> out((size_t)Tq * C, 0.0f);
    for (uint32_t tq = 0; tq < Tq; ++tq) {
        for (uint32_t h = 0; h < heads; ++h) {
            std::vector<double> acc(hd, 0.0);
            double m = -INFINITY, l = 0.0;
            for (uint32_t tk = 0; tk < Tk; ++tk) {
                if (causal && tk > tq) continue;
                double dot = 0.0;
                for (uint32_t d = 0; d < hd; ++d) {
                    dot += (double)q[(size_t)tq * C + h * hd + d] *
                           (double)k[(size_t)tk * C + h * hd + d];
                }
                double s = dot * scale;
                double mNew = std::max(m, s);
                double resc = (m == -INFINITY) ? 0.0 : std::exp(m - mNew);
                l *= resc;
                for (uint32_t d = 0; d < hd; ++d) acc[d] *= resc;
                double p = std::exp(s - mNew);
                l += p;
                for (uint32_t d = 0; d < hd; ++d) {
                    acc[d] += p * (double)v[(size_t)tk * C + h * hd + d];
                }
                m = mNew;
            }
            double inv = l > 0.0 ? 1.0 / l : 0.0;
            for (uint32_t d = 0; d < hd; ++d) {
                out[(size_t)tq * C + h * hd + d] = (float)(acc[d] * inv);
            }
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* spvDir = argc > 1 ? argv[1] : "shaders";

    ofc::gpu::Engine eng;
    eng.init();
    eng.loadKernels(spvDir);
    eng.initArenas(1u << 20, 1u << 20);

    const Shape shapes[] = {
        {"self T=32 C=320 h=5 causal", 32, 32, 320, 5, true, true},
        {"cross Tq=32 Tk=77 C=1280 h=20", 32, 77, 1280, 20, false, true},
        {"self T=64 C=320 h=8 causal hd40", 64, 64, 320, 8, true, true},
        {"cross Tq=48 Tk=96 C=512 h=8", 48, 96, 512, 8, false, true},
    };

    int failures = 0;
    std::mt19937 rng(20261101);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (const Shape& s : shapes) {
        std::vector<float> q((size_t)s.Tq * s.C), k((size_t)s.Tk * s.C), v((size_t)s.Tk * s.C);
        for (auto& x : q) x = dist(rng);
        for (auto& x : k) x = dist(rng);
        for (auto& x : v) x = dist(rng);
        std::vector<float> ref = reference(q, k, v, s.Tq, s.Tk, s.C, s.heads, s.causal);

        ofc::gpu::Arena arena;
        arena.init(eng.device(), eng.physDevice(), (uint64_t)(2 * s.Tq + 2 * s.Tk) * s.C * 4 + (4u << 20),
                   true, false);
        ofc::gpu::FusedAttnArgs a{};
        a.q = arena.alloc((uint64_t)s.Tq * s.C * 4);
        a.k = arena.alloc((uint64_t)s.Tk * s.C * 4);
        a.v = arena.alloc((uint64_t)s.Tk * s.C * 4);
        a.out = arena.alloc((uint64_t)s.Tq * s.C * 4);
        a.Tq = s.Tq;
        a.Tk = s.Tk;
        a.C = s.C;
        a.heads = s.heads;
        a.causal = s.causal;

        eng.uploadNow(a.q, q.data(), (uint64_t)q.size() * 4);
        eng.uploadNow(a.k, k.data(), (uint64_t)k.size() * 4);
        eng.uploadNow(a.v, v.data(), (uint64_t)v.size() * 4);

        eng.beginStep();
        ofc::gpu::fusedAttention(eng, a);
        eng.endStep();
        eng.waitStep();

        std::vector<float> got(ref.size());
        eng.downloadNow(a.out, got.data(), (uint64_t)got.size() * 4);

        double maxAbs = 0.0, maxRef = 0.0;
        for (size_t i = 0; i < got.size(); ++i) {
            maxAbs = std::max(maxAbs, std::fabs((double)got[i] - ref[i]));
            maxRef = std::max(maxRef, std::fabs((double)ref[i]));
        }
        double rel = maxAbs / std::max(1e-9, maxRef);
        bool ok = rel <= 2e-5;
        std::printf("%s %-38s maxAbs=%.3g rel=%.3g\n", ok ? "PASS" : "FAIL", s.name, maxAbs, rel);
        if (!ok) ++failures;
        arena.destroy();
    }

    std::printf("\nlatency (fused_attn only):\n");
    const Shape perf[] = {
        {"self 1024^2 C=320 h=5", 1024, 1024, 320, 5, true, false},
        {"self 4096^2 C=320 h=5", 4096, 4096, 320, 5, true, false},
        {"self 9216^2 C=640 h=10", 9216, 9216, 640, 10, true, false},
        {"cross 9216x77 C=1280 h=20", 9216, 77, 1280, 20, false, false},
    };
    for (const Shape& s : perf) {
        ofc::gpu::Arena arena;
        arena.init(eng.device(), eng.physDevice(),
                   (uint64_t)(2 * s.Tq + 2 * s.Tk) * s.C * 4 + (4u << 20), true, false);
        ofc::gpu::FusedAttnArgs a{};
        a.q = arena.alloc((uint64_t)s.Tq * s.C * 4);
        a.k = arena.alloc((uint64_t)s.Tk * s.C * 4);
        a.v = arena.alloc((uint64_t)s.Tk * s.C * 4);
        a.out = arena.alloc((uint64_t)s.Tq * s.C * 4);
        a.Tq = s.Tq; a.Tk = s.Tk; a.C = s.C; a.heads = s.heads; a.causal = s.causal;
        std::vector<float> zero((size_t)std::max(s.Tq, s.Tk) * s.C, 0.0f);
        eng.uploadNow(a.q, zero.data(), (uint64_t)a.q.size);
        eng.uploadNow(a.k, zero.data(), (uint64_t)a.k.size);
        eng.uploadNow(a.v, zero.data(), (uint64_t)a.v.size);
        double best = 1e30;
        for (int it = 0; it < 3; ++it) {
            auto t0 = std::chrono::steady_clock::now();
            eng.beginStep();
            ofc::gpu::fusedAttention(eng, a);
            eng.endStep();
            eng.waitStep();
            auto t1 = std::chrono::steady_clock::now();
            best = std::min(best, std::chrono::duration<double>(t1 - t0).count() * 1e3);
        }
        std::printf("  %-38s %.2f ms\n", s.name, best);
        arena.destroy();
    }

    eng.destroy();
    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS", failures,
                failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
