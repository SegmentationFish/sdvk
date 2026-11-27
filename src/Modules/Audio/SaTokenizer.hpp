// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
// SentencePiece Unigram tokenizer for the T5 conditioner of Stable Audio Open Small.
//
// The model file (spiece.model) is a protobuf with repeated SentencePiece entries; we parse
// just enough of it (piece string + score + type) and run the same Viterbi segmentation
// sentencepiece does. Text normalization follows T5's "nmt_nfkc" spec: NFKC, collapse
// whitespace, add the dummy prefix and escape spaces as U+2581.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ofc {
namespace audio {

class SpTokenizer {
public:
    bool load(const std::string& spiecePath, std::string& err);
    // Token ids without EOS; `unkId` is substituted for pieces that are not in the vocabulary.
    std::vector<int> encode(const std::string& text) const;
    // T5 convention: append EOS, pad with padId up to `length`.
    std::vector<int> encodePadded(const std::string& text, int length) const;
    int unkId() const { return unkId_; }
    int eosId() const { return eosId_; }
    int padId() const { return padId_; }
    size_t vocabSize() const { return pieces_.size(); }
    // Drops the dummy prefix that encodes the leading space (used only for debugging).
    std::string normalize(const std::string& text) const;

private:
    struct Piece {
        std::string text;
        float score = 0.0f;
        int type = 1;  // 1 normal, 2 unknown, 3 control, 4 user-defined, 6 byte
    };
    std::vector<Piece> pieces_;
    std::unordered_map<std::string, int> ids_;
    int unkId_ = 2, eosId_ = 1, padId_ = 0;
    float minScore_ = 0.0f;
    size_t maxPieceBytes_ = 0;
};

}  // namespace audio
}  // namespace ofc
