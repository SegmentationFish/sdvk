// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
#include "fused_attn.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace ofc {
namespace gpu {

namespace {

struct FaPush {
    uint32_t Tq;
    uint32_t Tk;
    uint32_t C;
    uint32_t heads;
    uint32_t hd;
    uint32_t causal;
    uint32_t pad0;
    uint32_t pad1;
    float scale;
};
static_assert(sizeof(FaPush) == 36, "fused_attn push layout drifted");

constexpr uint32_t kQueriesPerWg = 32;

uint32_t ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

}  // namespace

bool fusedAttnSupported(uint32_t C, uint32_t heads) {
    if (!C || !heads || C % heads) return false;
    uint32_t hd = C / heads;
    return hd > 0 && hd <= 64;
}

void fusedAttention(Engine& eng, const FusedAttnArgs& a) {
    if (!fusedAttnSupported(a.C, a.heads)) {
        std::fprintf(stderr, "fusedAttention: unsupported C=%u heads=%u\n", a.C, a.heads);
        std::exit(1);
    }
    uint32_t hd = a.C / a.heads;
    const char* simple = std::getenv("SDVK_ATTN_SIMPLE");
    bool tiled = (hd == 64) && a.Tk >= 256 && !(simple && simple[0] == '1');
    FaPush pc{};
    pc.Tq = a.Tq;
    pc.Tk = a.Tk;
    pc.C = a.C;
    pc.heads = a.heads;
    pc.hd = hd;
    pc.causal = a.causal ? 1u : 0u;
    pc.scale = 1.0f / std::sqrt((float)hd);
    if (tiled) {
        eng.dispatch("fused_attn_tiled", {a.q, a.k, a.v, a.out}, &pc, ceilDiv(a.Tq, 64), a.heads);
    } else {
        eng.dispatch("fused_attn", {a.q, a.k, a.v, a.out}, &pc, ceilDiv(a.Tq, kQueriesPerWg), a.heads);
    }
}

}  // namespace gpu
}  // namespace ofc
