// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
#include "SaTokenizer.hpp"

#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace ofc {
namespace audio {

namespace {

// Piece types in the sentencepiece protobuf.
enum : int { kNormal = 1, kUnknown = 2, kControl = 3, kUserDefined = 4, kUnused = 5, kByte = 6 };

struct Reader {
    const uint8_t* p;
    const uint8_t* end;
    bool ok = true;

    uint64_t varint() {
        uint64_t v = 0;
        int shift = 0;
        while (p < end && shift < 64) {
            uint8_t b = *p++;
            v |= (uint64_t)(b & 0x7f) << shift;
            if (!(b & 0x80)) return v;
            shift += 7;
        }
        ok = false;
        return 0;
    }
    const uint8_t* bytes(uint64_t n) {
        if ((uint64_t)(end - p) < n) {
            ok = false;
            return nullptr;
        }
        const uint8_t* r = p;
        p += n;
        return r;
    }
    // Skips one field of wire type `wt` (0 varint, 1 fixed64, 2 length, 5 fixed32).
    void skip(int wt) {
        switch (wt) {
            case 0: varint(); break;
            case 1: bytes(8); break;
            case 2: { uint64_t n = varint(); bytes(n); break; }
            case 5: bytes(4); break;
            default: ok = false; break;
        }
    }
};

float readF32(const uint8_t* p) {
    float f;
    std::memcpy(&f, p, 4);
    return f;
}

// NFKC through the Windows normalizer: T5's "nmt_nfkc" rule is NFKC plus whitespace
// handling, and the precompiled charsmap in spiece.model is exactly that table.
std::string nfkc(const std::string& in) {
    if (in.empty()) return in;
    int wl = MultiByteToWideChar(CP_UTF8, 0, in.data(), (int)in.size(), nullptr, 0);
    if (wl <= 0) return in;
    std::wstring w((size_t)wl, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, in.data(), (int)in.size(), w.data(), wl);

    int need = NormalizeString(NormalizationKC, w.data(), (int)w.size(), nullptr, 0);
    if (need <= 0) return in;
    std::wstring norm((size_t)need, L'\0');
    need = NormalizeString(NormalizationKC, w.data(), (int)w.size(), norm.data(), need);
    if (need <= 0) return in;
    norm.resize((size_t)need);

    int ol = WideCharToMultiByte(CP_UTF8, 0, norm.data(), (int)norm.size(), nullptr, 0, nullptr, nullptr);
    if (ol <= 0) return in;
    std::string out((size_t)ol, '\0');
    WideCharToMultiByte(CP_UTF8, 0, norm.data(), (int)norm.size(), out.data(), ol, nullptr, nullptr);
    return out;
}

bool isSpaceCp(uint32_t cp) {
    // sentencepiece's whitespace set: ASCII space, tab, newline, CR, vertical tab, form feed
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0b || cp == 0x0c || cp == 0x85;
}

uint32_t nextCp(const std::string& s, size_t& i) {
    unsigned char c = (unsigned char)s[i];
    uint32_t cp = c;
    size_t n = 1;
    if (c >= 0xF0) {
        cp = c & 0x07;
        n = 4;
    } else if (c >= 0xE0) {
        cp = c & 0x0F;
        n = 3;
    } else if (c >= 0xC0) {
        cp = c & 0x1F;
        n = 2;
    }
    if (i + n > s.size()) n = 1;
    for (size_t k = 1; k < n; k++) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
    i += n;
    return cp;
}

void appendUtf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out.push_back((char)cp);
    } else if (cp < 0x800) {
        out.push_back((char)(0xC0 | (cp >> 6)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back((char)(0xE0 | (cp >> 12)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char)(0xF0 | (cp >> 18)));
        out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    }
}

const char* kSpace = "\xE2\x96\x81";  // U+2581 LOWER ONE EIGHTH BLOCK

}  // namespace

bool SpTokenizer::load(const std::string& spiecePath, std::string& err) {
    std::ifstream f(spiecePath, std::ios::binary);
    if (!f) {
        err = "cannot open " + spiecePath;
        return false;
    }
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    Reader r{buf.data(), buf.data() + buf.size()};
    while (r.p < r.end && r.ok) {
        uint64_t tag = r.varint();
        int field = (int)(tag >> 3), wt = (int)(tag & 7);
        if (field == 1 && wt == 2) {
            uint64_t len = r.varint();
            const uint8_t* p = r.bytes(len);
            if (!p) break;
            Reader pr{p, p + len};
            Piece pc;
            while (pr.p < pr.end && pr.ok) {
                uint64_t t2 = pr.varint();
                int f2 = (int)(t2 >> 3), w2 = (int)(t2 & 7);
                if (f2 == 1 && w2 == 2) {
                    uint64_t n = pr.varint();
                    const uint8_t* sp = pr.bytes(n);
                    if (!sp) break;
                    pc.text.assign((const char*)sp, (size_t)n);
                } else if (f2 == 2 && w2 == 5) {
                    const uint8_t* sp = pr.bytes(4);
                    if (!sp) break;
                    pc.score = readF32(sp);
                } else if (f2 == 3 && w2 == 0) {
                    pc.type = (int)pr.varint();
                } else {
                    pr.skip(w2);
                }
            }
            pieces_.push_back(std::move(pc));
        } else {
            r.skip(wt);
        }
    }
    if (pieces_.empty()) {
        err = "no pieces in " + spiecePath;
        return false;
    }
    for (size_t i = 0; i < pieces_.size(); i++) {
        const Piece& p = pieces_[i];
        ids_.emplace(p.text, (int)i);
        if (p.type == kUnknown) unkId_ = (int)i;
        if (p.type == kControl) {
            if (p.text == "</s>") eosId_ = (int)i;
            if (p.text == "<pad>") padId_ = (int)i;
        }
        maxPieceBytes_ = std::max(maxPieceBytes_, p.text.size());
        minScore_ = std::min(minScore_, p.score);
    }
    return true;
}

std::string SpTokenizer::normalize(const std::string& text) const {
    std::string t = nfkc(text);
    // collapse whitespace, drop leading/trailing (remove_extra_whitespaces)
    std::string sq;
    bool prevSpace = true;  // leading spaces are dropped
    for (size_t i = 0; i < t.size();) {
        size_t j = i;
        uint32_t cp = nextCp(t, j);
        if (isSpaceCp(cp)) {
            if (!prevSpace) sq.push_back(' ');
            prevSpace = true;
        } else {
            sq.append(t, i, j - i);
            prevSpace = false;
        }
        i = j;
    }
    while (!sq.empty() && sq.back() == ' ') sq.pop_back();
    // add_dummy_prefix: the text is encoded with one leading space
    std::string out = " ";
    out += sq;
    std::string esc;
    for (size_t i = 0; i < out.size();) {
        size_t j = i;
        uint32_t cp = nextCp(out, j);
        if (cp == ' ') {
            esc += kSpace;
        } else {
            esc.append(out, i, j - i);
        }
        i = j;
    }
    return esc;
}

std::vector<int> SpTokenizer::encode(const std::string& text) const {
    const std::string s = normalize(text);
    const int n = (int)s.size();
    std::vector<float> best((size_t)n + 1, -1e30f);
    std::vector<int> bestLen((size_t)n + 1, 0), bestId((size_t)n + 1, -1);
    std::vector<int> bestUnk((size_t)n + 1, 0);
    best[0] = 0.0f;
    const float unkScore = minScore_ - 10.0f;
    for (int i = 0; i < n; i++) {
        if (best[(size_t)i] <= -1e29f) continue;
        // longest match first is not required for correctness; Viterbi over all piece lengths
        for (int len = 1; len <= (int)maxPieceBytes_ && i + len <= n; len++) {
            auto it = ids_.find(s.substr((size_t)i, (size_t)len));
            if (it == ids_.end()) continue;
            const Piece& p = pieces_[(size_t)it->second];
            if (p.type == kControl || p.type == kUnused) continue;
            float sc = best[(size_t)i] + p.score;
            if (sc > best[(size_t)i + len]) {
                best[(size_t)i + len] = sc;
                bestLen[(size_t)i + len] = len;
                bestId[(size_t)i + len] = it->second;
                bestUnk[(size_t)i + len] = 0;
            }
        }
        // unknown byte fallback: consume one UTF-8 code point
        size_t j = (size_t)i;
        nextCp(s, j);
        int len = (int)(j - (size_t)i);
        float sc = best[(size_t)i] + unkScore;
        if (sc > best[(size_t)i + len]) {
            best[(size_t)i + len] = sc;
            bestLen[(size_t)i + len] = len;
            bestId[(size_t)i + len] = unkId_;
            bestUnk[(size_t)i + len] = 1;
        }
    }
    std::vector<int> ids;
    for (int i = n; i > 0;) {
        int len = bestLen[(size_t)i];
        if (len <= 0) {  // unreachable, but never loop forever
            len = 1;
        }
        ids.push_back(bestId[(size_t)i]);
        i -= len;
    }
    std::reverse(ids.begin(), ids.end());
    return ids;
}

std::vector<int> SpTokenizer::encodePadded(const std::string& text, int length) const {
    std::vector<int> ids = encode(text);
    ids.push_back(eosId_);
    if ((int)ids.size() > length) ids.resize((size_t)length);
    ids.resize((size_t)length, padId_);
    return ids;
}

}  // namespace audio
}  // namespace ofc
