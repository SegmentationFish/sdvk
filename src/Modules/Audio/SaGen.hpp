// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
// Stable Audio Open Small: text conditioning (T5) + DiT + Oobleck decoder.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ofc {
namespace audio {

struct SaParams {
    std::string modelDir;   // .../stable-audio-small (holds model.safetensors, t5-base/, vae/)
    std::string prompt;
    std::string seconds = "6";  // duration as text so the CLI can keep it exact
    float secondsTotal = 6.0f;
    int steps = 8;
    int seed = 0;
    std::string outWav = "sdvk_audio.wav";
    std::string spvDir = "build/shaders";
    int threads = 0;         // 0 = hardware_concurrency
    bool verbose = true;
    // streaming: emit the decoded audio as chunks instead of one buffer
    bool stream = false;
    int chunkFrames = 8;     // latent frames per streamed chunk (8 = 16384 samples)
    int overlapFrames = 4;   // decode context on each side, dropped after crossfade
};

// Tokenizes `prompt`, runs the T5 encoder and returns [64][768] last_hidden_state (row major).
// Implemented on the CPU: the encoder is one-shot and tiny compared to the VAE.
bool saTextEncode(const SaParams& p, std::vector<float>& hidden, std::vector<int>& ids, std::string& err);

// Diagnostics: `--t5check <dumpDir>` writes our ids/hidden next to the torch reference dumps.
void saDumpT5(const SaParams& p, const std::string& outDir);

// Full generation. Writes `p.outWav`; returns false and fills `err` on failure.
bool saGenerate(const SaParams& p, std::string& err);

}  // namespace audio
}  // namespace ofc
