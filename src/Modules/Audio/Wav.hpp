// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
// Zero-dependency WAV reader/writer. Everything downstream works on interleaved float samples in
// [-1,1]; the file layer is the only place that knows about PCM bit depths.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ofc {
namespace audio {

struct Audio {
    uint32_t sampleRate = 0;
    uint32_t channels = 0;
    std::vector<float> samples;  // interleaved, -1..1

    uint64_t frames() const { return channels ? (uint64_t)samples.size() / channels : 0; }
    double seconds() const { return sampleRate ? (double)frames() / (double)sampleRate : 0.0; }
    bool empty() const { return samples.empty() || !channels || !sampleRate; }
};

// Reads PCM 8/16/24/32, IEEE float 32/64 and WAVE_FORMAT_EXTENSIBLE. Extra chunks (LIST, fact,
// bext, id3) are skipped, so files straight out of a DAW load as-is.
bool readWav(const std::string& path, Audio& out, std::string& err);

// bits = 16 (PCM), 24 (PCM) or 32 (IEEE float, no dithering so a roundtrip stays exact).
bool writeWav(const std::string& path, const Audio& a, int bits, std::string& err);

// Drops channels by averaging, keeps the rate.
void toMono(Audio& a);

}  // namespace audio
}  // namespace ofc
