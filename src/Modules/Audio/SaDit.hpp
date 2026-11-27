// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
// MiniMax Music 3 synthesis stage: condition encoder + flow-matching DiT (2.4B).
//
// The checkpoint shipped for ComfyUI is a GGUF holding both components:
//   diffusion_transformer.*      preprocess_conv / proj_in / 36 blocks / proj_out / postprocess_conv
//   latent_conditioners.0.*      condition encoder conv1d (4096 -> 2048, k=3)
//   cond_layer_{logits,scale}    the 8-way mix over the autoregressive hidden states
// 2-D weights are loaded as Q8_0 (in-shader dequant, same path as the SDXL UNet); norms and biases
// stay fp32. Everything runs in the "transposed" layout [seq, channels] so a Linear is one GEMM.
#pragma once

#include <string>
#include <vector>

namespace ofc {
namespace audio {

class SaDit {
public:
    ~SaDit();
    // maxLayers = 0 loads every block; a smaller value keeps only the first blocks (used to
    // compare against a small torch reference on a memory-tight box).
    bool load(const std::string& ggufPath, const std::string& spvDir, int maxLayers, std::string& err);

    // states: [frames][8*4096] (layer-major per frame, as the RVQ stage produces).
    // cond: [latentLen][2048], latentLen = int(frames * 44100/24000 * 960/512).
    bool encodeCond(const std::vector<float>& states, int frames, std::vector<float>& cond, int& latentLen,
                    std::string& err);

    // x: [128][T] latents, cond: [T][2048]; vel: [128][T]. Flow time t in [0,1] (0 = noise).
    bool forward(const std::vector<float>& x, int T, float t, const std::vector<float>& cond,
                 std::vector<float>& vel, std::string& err);

    int layers() const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

// Diagnostic: checks the fp32 and Q8 GEMM against a host dot product (used by --gemm-check).
void saSelfTestGemm(int M, int N, int K, float& e32, float& e8);

}  // namespace audio
}  // namespace ofc
