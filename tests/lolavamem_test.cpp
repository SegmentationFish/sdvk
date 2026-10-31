// LolavaMem property test: random alloc/free sequences must never overlap live payloads, must keep
// the physical/free-list invariants, and must return the whole range after everything is freed.
#include "Modules/GPU/lolavamem.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using ofc::gpu::LolavaMem;

struct Live {
    uint64_t off;
    uint64_t bytes;
};

static bool overlaps(const std::vector<Live>& live, uint64_t off, uint64_t bytes) {
    for (const auto& l : live)
        if (off < l.off + l.bytes && l.off < off + bytes) return true;
    return false;
}

int main() {
    const uint64_t kCap = 64ull << 20;  // 64 MiB
    LolavaMem m;
    m.init(kCap, 256);
    std::printf("capacity %.1f MiB, align 256\n", m.capacity() / 1048576.0);
    if (!m.check()) {
        std::printf("FAIL: initial check\n");
        return 1;
    }

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> op(0, 99);
    std::uniform_int_distribution<uint64_t> sz(256, 256 * 1024);
    std::uniform_int_distribution<size_t> pickGen(0, 1000);
    std::vector<Live> live;
    uint64_t allocs = 0, frees = 0, failed = 0;

    const int kIters = std::getenv("LOLA_DEBUG") ? 20 : 60000;
    for (int it = 0; it < kIters; it++) {
        bool doAlloc = live.empty() || op(rng) < 62;
        if (doAlloc) {
            uint64_t bytes = sz(rng);
            uint64_t off = m.alloc(bytes);
            if (off == LolavaMem::kInvalid) {
                failed++;
            } else {
                if (m.used() > m.capacity()) {
                    std::printf("FAIL: used > capacity\n");
                    return 1;
                }
                if (overlaps(live, off, bytes)) {
                    std::printf("FAIL: overlap at iter %d off %llu size %llu\n", it,
                                (unsigned long long)off, (unsigned long long)bytes);
                    return 1;
                }
                live.push_back({off, bytes});
                allocs++;
                if (std::getenv("LOLA_DEBUG"))
                    std::printf("A it=%d off=%llu size=%llu\n", it, (unsigned long long)off, (unsigned long long)bytes);
            }
        } else {
            size_t k = pickGen(rng) % live.size();
            Live l = live[k];
            live[k] = live.back();
            live.pop_back();
            m.free(l.off);
            frees++;
            if (std::getenv("LOLA_DEBUG")) std::printf("F it=%d off=%llu size=%llu\n", it, (unsigned long long)l.off, (unsigned long long)l.bytes);
        }
        if ((it % 4096) == 0 && !m.check()) {
            std::printf("FAIL: check() at iter %d\n", it);
            return 1;
        }
    }
    if (!m.check()) {
        std::printf("FAIL: check() after loop\n");
        return 1;
    }
    if (std::getenv("LOLA_DEBUG"))
        std::printf("live %zu used %llu free %llu largest %llu\n", live.size(),
                    (unsigned long long)m.used(), (unsigned long long)m.freeBytes(),
                    (unsigned long long)m.largestFree());
    std::printf("mid: live %zu, used %.1f MiB, largestFree %.1f MiB, frag %.1f%%\n",
                live.size(), m.used() / 1048576.0, m.largestFree() / 1048576.0,
                m.fragmentation() * 100.0);

    for (auto& l : live) m.free(l.off);
    if (!m.check()) {
        std::printf("FAIL: check() after drain\n");
        return 1;
    }
    if (m.used() != 0 || m.allocCount() != 0) {
        std::printf("FAIL: leak used=%llu live=%llu\n", (unsigned long long)m.used(),
                    (unsigned long long)m.allocCount());
        return 1;
    }
    if (m.freeBytes() != m.capacity() || m.largestFree() != m.capacity()) {
        std::printf("FAIL: not fully coalesced free=%llu largest=%llu cap=%llu\n",
                    (unsigned long long)m.freeBytes(), (unsigned long long)m.largestFree(),
                    (unsigned long long)m.capacity());
        return 1;
    }
    std::printf("OK  allocs %llu frees %llu oom %llu  coalesced back to %.1f MiB\n",
                (unsigned long long)allocs, (unsigned long long)frees, (unsigned long long)failed,
                m.capacity() / 1048576.0);
    return 0;
}
