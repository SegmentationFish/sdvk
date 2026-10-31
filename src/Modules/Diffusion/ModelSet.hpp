// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
// Model directory resolution: scan a folder, classify the files, fill in what the user left
// blank, and refuse combinations that cannot work. Handles diffusers component folders,
// GGUF checkpoints and single-file safetensors checkpoints.
#pragma once

#include "GenParams.hpp"

#include <string>
#include <vector>

namespace ofc {
namespace sdxl {

struct ModelSet {
    std::vector<std::string> warn;
    std::string unetName;  // short label for the PNG info chunk
};

// Fills p.models from modelDir + whatever explicit paths were given. Returns false and sets err
// when a required component is missing or ambiguous.
bool resolveModels(GenParams& p, ModelSet& out, std::string& err);

// Short human label for a model path (file name without directories / extension).
std::string shortName(const std::string& path);

// Accepts a single-file .safetensors path and reports what it contains, so --unet can point at
// either a diffusers UNet folder file or a full checkpoint (.gguf files are always UNets).
bool inspectUnetFile(const std::string& path, std::string& err);

}  // namespace sdxl
}  // namespace ofc
