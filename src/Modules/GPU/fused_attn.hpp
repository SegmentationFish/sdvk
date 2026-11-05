// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
#pragma once

#include "vk_engine.hpp"

namespace ofc {
namespace gpu {

// One fused attention operator: [Tq,C] x [Tk,C] x [Tk,C] -> [Tq,C].
// Q/K/V are fp32, head-major inside C (C = heads * headDim, headDim <= 64).
struct FusedAttnArgs {
    BufRef q;
    BufRef k;
    BufRef v;
    BufRef out;
    uint32_t Tq = 0;
    uint32_t Tk = 0;
    uint32_t C = 0;
    uint32_t heads = 1;
    bool causal = false;
};

// Returns false when the shape cannot be served by fused_attn (fall back to the
// GEMM+softmax path): headDim must be 1..64 and divide C by heads.
bool fusedAttnSupported(uint32_t C, uint32_t heads);
void fusedAttention(Engine& eng, const FusedAttnArgs& a);

}  // namespace gpu
}  // namespace ofc
