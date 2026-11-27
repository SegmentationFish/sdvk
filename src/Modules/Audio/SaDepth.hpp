// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
// MiniMax Music 3 RVQ depth decoder: an intra-frame causal transformer over codebook positions
// (4 layers, 16 heads x 256, hidden 4096, 7 acoustic codebooks of 1024). For each 25 Hz frame it
// turns the LM's semantic token into 7 acoustic codes and yields the 8 hidden states that the
// condition encoder fuses.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ofc {
namespace audio {

constexpr int kDepthCodebooks = 7;      // acoustic codebooks (semantic comes from the LM)
constexpr int kDepthVocab = 1024;       // entries per codebook
constexpr int kDepthPositions = 16;     // pos_embedding rows
constexpr int kDepthStates = 8;         // LM hidden + 7 depth hiddens
constexpr int kDepthHidden = 4096;

class SaDepth {
public:
    ~SaDepth();
    bool load(const std::string& ggufPath, const std::string& spvDir, std::string& err);

    // Decodes one frame. `lmHidden` is the global LM's last hidden state [4096]; `semantic` is the
    // codebook-0 code. Fills `codes` (7 acoustic codes), `states` ([8][4096], LM hidden first) and
    // `embedding` ([4096], the scaled sum of the 8 codebook embeddings for the LM feedback).
    bool decodeFrame(const std::vector<float>& lmHidden, int semantic, uint32_t seed, float cfg,
                     std::vector<int>& codes, std::vector<float>& states, std::vector<float>& embedding,
                     std::string& err);

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace audio
}  // namespace ofc
