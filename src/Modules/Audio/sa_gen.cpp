// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
// Stable Audio generation pipeline. Currently wires the conditioning stage; the DiT sampler and
// the Oobleck decoder land in the next step (see HANDOVER.md).
#include "SaGen.hpp"

#include <cstdio>
#include <vector>

namespace ofc {
namespace audio {

bool saGenerate(const SaParams& p, std::string& err) {
    std::vector<float> hidden;
    std::vector<int> ids;
    if (!saTextEncode(p, hidden, ids, err)) return false;
    std::printf("audio: conditioning ready (%zu tokens, %zu floats)\n", ids.size(), hidden.size());
    std::printf("total: dit sampler + oobleck decoder not wired yet\n");
    (void)p.outWav;
    err = "the DiT sampler / Oobleck decoder are not implemented yet (T5 stage verified)";
    return false;
}

}  // namespace audio
}  // namespace ofc
