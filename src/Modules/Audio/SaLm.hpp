// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
// MiniMax Music 3 global LM: Qwen3-8B (36 layers, 32/8 GQA, 128 head dim, SwiGLU 12288,
// q/k RMSNorm, RoPE theta 1e6, 200k vocab) that emits 25 Hz semantic audio tokens.
//
// The weights are streamed from the GGUF for every forward instead of staying resident: the
// Q5_K_M file is 5.85 GB and this machine has ~3.5 GB of free RAM, so the only way to run the
// 8B model is to read each layer's matrices once per token and dequantise them in-shader.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ofc {
namespace audio {

// Token ids from the reference prompt assembly (MM3 uses the Qwen2 special tokens).
constexpr int kTokImStart = 151644;
constexpr int kTokImEnd = 151645;
constexpr int kTokAudioStart = 151669;
constexpr int kTokAudioEnd = 151670;
constexpr int kTokCaptionStart = 151671;
constexpr int kTokCaptionEnd = 151672;
constexpr int kTokLyricsStart = 151673;
constexpr int kTokLyricsEnd = 151674;
constexpr int kTokSemanticBase = 151675;   // first semantic codebook entry
constexpr int kSemanticVocab = 16384;
constexpr int kTokAudioCfg = 151654;       // replacement for the cfg stream

class SaLm {
public:
    ~SaLm();
    // Tokenizer tables come from the same GGUF (gpt2-style vocab + merges).
    bool load(const std::string& ggufPath, const std::string& spvDir, std::string& err);

    // Runs `frames` decode steps after prefilling `prompt`; appends one semantic token per frame
    // and the per-frame hidden state ([frames][4096], the value the depth decoder conditions on).
    bool generate(const std::vector<int>& prompt, int frames, uint32_t seed, float cfg, int topK,
                  std::vector<int>& semantic, std::vector<float>& hidden, std::string& err);

    // ---- tokenizer (Qwen2 byte-level BPE)
    std::vector<int> encode(const std::string& text) const;
    const std::string& error() const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace audio
}  // namespace ofc
