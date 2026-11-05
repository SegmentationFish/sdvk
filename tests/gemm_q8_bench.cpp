// Microbenchmark / correctness probe for gemm_nt_q8 tile variants.
// Build: cmake --build build --target gemm_q8_bench
// Run:   ./gemm_q8_bench [spvDir]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "Modules/GPU/vk_engine.hpp"

namespace {

struct Pg {
    uint32_t M, N, K, mode, sA, sB, sC, batch, lda, ldb, ldc, pad;
};

struct Variant {
    const char* kernel;
    uint32_t bm, bn;
};

struct Shape {
    uint32_t M, N, K;
};

const Variant kVariants[] = {
    {"gemm_nt_q8", 64, 64},
    {"gemm_nt_q8_v", 64, 64},
    {"gemm_nt_q8_v_bm128", 128, 64},
    {"gemm_nt_q8_v_bk32", 64, 64},
    {"gemm_nt_q8_v_bm128_bk32", 128, 64},
};

uint32_t ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

void uploadZeros(ofc::gpu::Engine& eng, ofc::gpu::BufRef dst) {
    std::vector<uint8_t> zero((size_t)dst.size, 0);
    eng.uploadNow(dst, zero.data(), dst.size);
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* spv = argc > 1 ? argv[1] : "shaders";

    ofc::gpu::Engine eng;
    eng.init();
    eng.loadKernels(spv);
    eng.initArenas(1u << 20, 1u << 20);

    // ---- correctness on a small shape ----
    {
        const uint32_t M = 100, N = 100, K = 96;
        std::mt19937 rng(7);
        std::uniform_int_distribution<int> i8(-127, 127);
        std::vector<float> A((size_t)M * K);
        std::vector<int8_t> B((size_t)N * K);
        std::vector<float> Bs((size_t)(N * K + 31) / 32);
        std::vector<float> Cs((size_t)(N * K + 31) / 32);
        for (auto& x : A) x = (float)(i8(rng)) / 127.0f;
        for (auto& x : B) x = (int8_t)i8(rng);
        for (auto& x : Bs) x = 0.01f;
        for (auto& x : Cs) x = 0.01f;
        std::vector<float> ref((size_t)M * N, 0.0f);
        for (uint32_t m = 0; m < M; ++m)
            for (uint32_t n = 0; n < N; ++n) {
                float acc = 0;
                for (uint32_t k = 0; k < K; ++k) {
                    float w = (float)B[(size_t)n * K + k] * Cs[((size_t)n * K + k) >> 5];
                    acc += A[(size_t)m * K + k] * w;
                }
                ref[(size_t)m * N + n] = acc;
            }

        ofc::gpu::Arena arena;
        arena.init(eng.device(), eng.physDevice(), 8u << 20, true, false);
        auto aB = arena.alloc((uint64_t)M * K * 4);
        auto bB = arena.alloc((uint64_t)(N * K + 3) / 4 * 4);
        auto sB = arena.alloc((uint64_t)Bs.size() * 4);
        auto cB = arena.alloc((uint64_t)M * N * 4);
        eng.uploadNow(aB, A.data(), (uint64_t)A.size() * 4);
        eng.uploadNow(bB, B.data(), (uint64_t)B.size());
        eng.uploadNow(sB, Cs.data(), (uint64_t)Cs.size() * 4);

        for (const auto& v : kVariants) {
            Pg pc{M, N, K, 0, 0, 0, 0, 1, 0, 0, 0, 0};
            std::vector<ofc::gpu::BufRef> bufs = {aB, bB, cB, sB};
            eng.beginStep();
            eng.dispatch(v.kernel, bufs, &pc, ceilDiv(N, v.bn), ceilDiv(M, v.bm));
            eng.endStep();
            eng.waitStep();
            std::vector<float> got((size_t)M * N);
            eng.downloadNow(cB, got.data(), (uint64_t)got.size() * 4);
            double maxAbs = 0, maxRef = 0;
            for (size_t i = 0; i < got.size(); ++i) {
                maxAbs = std::max(maxAbs, std::fabs((double)got[i] - ref[i]));
                maxRef = std::max(maxRef, std::fabs((double)ref[i]));
            }
            std::printf("check %-20s rel=%.3g\n", v.kernel, maxAbs / std::max(1e-9, maxRef));
        }
        arena.destroy();
    }

    const Shape shapes[] = {
        {9216, 1280, 1280},
        {9216, 640, 640},
        {9216, 320, 320},
        {2304, 1280, 1280},
        {576, 1280, 2560},
    };

    std::printf("\n%-22s", "shape MxNxK");
    for (const auto& v : kVariants) std::printf(" %-18s", v.kernel);
    std::printf("\n");

    for (const Shape& s : shapes) {
        ofc::gpu::Arena arena;
        uint64_t aB = (uint64_t)s.M * s.K * 4;
        uint64_t wB = ((uint64_t)s.N * s.K + 3) / 4 * 4;
        uint64_t sB = ((uint64_t)s.N * s.K + 31) / 32 * 4;
        uint64_t cB = (uint64_t)s.M * s.N * 4;
        arena.init(eng.device(), eng.physDevice(), aB + wB + sB + cB + (1u << 20), true, false);
        auto refA = arena.alloc(aB);
        auto refB = arena.alloc(wB);
        auto refS = arena.alloc(sB);
        auto refC = arena.alloc(cB);
        uploadZeros(eng, refA);
        uploadZeros(eng, refB);
        uploadZeros(eng, refS);
        uploadZeros(eng, refC);

        std::printf("%-22s", (std::to_string(s.M) + "x" + std::to_string(s.N) + "x" +
                              std::to_string(s.K)).c_str());
        for (const auto& v : kVariants) {
            Pg pc{s.M, s.N, s.K, 0, 0, 0, 0, 1, 0, 0, 0, 0};
            std::vector<ofc::gpu::BufRef> bufs = {refA, refB, refC, refS};
            double best = 1e30;
            for (int it = 0; it < 3; ++it) {
                auto t0 = std::chrono::steady_clock::now();
                eng.beginStep();
                eng.dispatch(v.kernel, bufs, &pc, ceilDiv(s.N, v.bn), ceilDiv(s.M, v.bm));
                eng.endStep();
                eng.waitStep();
                auto t1 = std::chrono::steady_clock::now();
                best = std::min(best, std::chrono::duration<double>(t1 - t0).count() * 1e3);
            }
            double gflops = 2.0 * s.M * s.N * s.K / (best * 1e6);
            std::printf(" %8.2fms %6.0fG", best, gflops);
        }
        std::printf("\n");
        arena.destroy();
    }

    eng.destroy();
    return 0;
}
