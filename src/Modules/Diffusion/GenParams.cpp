// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
#include "GenParams.hpp"

#include <cstdio>

namespace ofc {
namespace sdxl {

std::string infoText(const GenParams& p, const std::string& modelName, const std::string& device) {
    char b[2048];
    std::string neg = p.negative.size() > 400 ? p.negative.substr(0, 400) + "..." : p.negative;
    std::snprintf(b, sizeof b,
                  "sdvk (self-built Vulkan compute engine)\n"
                  "model: %s\n"
                  "device: %s\n"
                  "size: %ux%u\n"
                  "steps: %d  cfg: %.2f  seed: %u\n"
                  "sampler: %s%s%s  spacing: %s\n"
                  "prompt: %s\n"
                  "negative: %s",
                  modelName.c_str(), device.c_str(), p.width, p.heightOrWidth(), p.steps, p.cfg, p.seed,
                  samplerName(p.sampler), p.vpred ? " vpred" : "", p.zsnr ? " zsnr" : "", p.spacing.c_str(),
                  p.prompt.c_str(), neg.c_str());
    return b;
}

}  // namespace sdxl
}  // namespace ofc
