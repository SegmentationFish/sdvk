// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
// MiniMax Music 3 flow-VAE decoder (DAC-style), latent [128, T] -> 44.1 kHz stereo.
//
// The 128 latent channels are two independent 64-channel tracks; each is decoded separately
// through dec_in_proj -> conv_in -> 4 (snake + transposed conv + 3 res units) blocks ->
// snake_out -> conv_out -> tanh, and the two mono tracks are interleaved into stereo.
// Convolutions run as im2col + GEMM (and GEMM + col2im for the transposed ones).
#pragma once

#include <string>
#include <vector>

namespace ofc {
namespace audio {

class SaVae {
public:
    ~SaVae();
    bool load(const std::string& path, const std::string& spvDir, std::string& err);
    // `latent` is [128, frames] frame-major (channel major inside a frame), as produced by the DiT.
    // Output is interleaved stereo at 44100 Hz, `frames * 512` sample pairs.
    bool decode(const std::vector<float>& latent, int frames, std::vector<float>& stereo, std::string& err);
    static constexpr int kSampleRate = 44100;
    static constexpr int kHop = 512;

    struct Buf;  // implementation detail (weights + safetensors handle)
    void* impl_ = nullptr;
};

}  // namespace audio
}  // namespace ofc
