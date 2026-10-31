// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
#include "Wav.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace ofc {
namespace audio {
namespace {

uint32_t rd32(const uint8_t* p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
uint16_t rd16(const uint8_t* p) { return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8)); }

void wr32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back((uint8_t)(x & 0xFF));
    v.push_back((uint8_t)((x >> 8) & 0xFF));
    v.push_back((uint8_t)((x >> 16) & 0xFF));
    v.push_back((uint8_t)((x >> 24) & 0xFF));
}
void wr16(std::vector<uint8_t>& v, uint16_t x) {
    v.push_back((uint8_t)(x & 0xFF));
    v.push_back((uint8_t)((x >> 8) & 0xFF));
}
void wrTag(std::vector<uint8_t>& v, const char* t) { v.insert(v.end(), t, t + 4); }

}  // namespace

bool readWav(const std::string& path, Audio& out, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz < 44) {
        std::fclose(f);
        err = path + " is too small to be a WAV file";
        return false;
    }
    std::vector<uint8_t> buf((size_t)sz);
    if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
        std::fclose(f);
        err = "short read on " + path;
        return false;
    }
    std::fclose(f);
    if (std::memcmp(buf.data(), "RIFF", 4) || std::memcmp(buf.data() + 8, "WAVE", 4)) {
        err = path + " is not a RIFF/WAVE file";
        return false;
    }
    uint16_t fmt = 0, ch = 0, bits = 0;
    uint32_t rate = 0;
    bool haveFmt = false;
    size_t dataOff = 0, dataLen = 0;
    size_t pos = 12;
    while (pos + 8 <= buf.size()) {
        uint32_t len = rd32(&buf[pos + 4]);
        const uint8_t* body = &buf[pos + 8];
        if (pos + 8 + len > buf.size()) len = (uint32_t)(buf.size() - (pos + 8));
        if (!std::memcmp(&buf[pos], "fmt ", 4) && len >= 16) {
            fmt = rd16(body);
            ch = rd16(body + 2);
            rate = rd32(body + 4);
            bits = rd16(body + 14);
            // WAVE_FORMAT_EXTENSIBLE hides the real format in the sub-format GUID's first two bytes
            if (fmt == 0xFFFE && len >= 40) fmt = rd16(body + 24);
            haveFmt = true;
        } else if (!std::memcmp(&buf[pos], "data", 4)) {
            dataOff = pos + 8;
            dataLen = len;
        }
        pos += 8 + len + (len & 1);  // chunks are word aligned
    }
    if (!haveFmt || !dataLen) {
        err = path + ": missing fmt or data chunk";
        return false;
    }
    if (!ch || !rate) {
        err = path + ": fmt chunk has no channels or sample rate";
        return false;
    }
    const bool isFloat = fmt == 3;
    if (!(isFloat ? (bits == 32 || bits == 64) : (bits == 8 || bits == 16 || bits == 24 || bits == 32))) {
        err = path + ": unsupported " + (isFloat ? "float" : "PCM") + " width " + std::to_string(bits);
        return false;
    }
    const uint32_t bytes = bits / 8;
    const uint64_t count = dataLen / bytes;
    out.sampleRate = rate;
    out.channels = ch;
    out.samples.resize((size_t)count);
    const uint8_t* p = &buf[dataOff];
    for (uint64_t i = 0; i < count; i++) {
        const uint8_t* q = p + i * bytes;
        float v = 0.0f;
        if (isFloat) {
            if (bits == 32) {
                uint32_t u = rd32(q);
                float t;
                std::memcpy(&t, &u, 4);
                v = t;
            } else {
                uint64_t u = 0;
                for (int k = 7; k >= 0; k--) u = (u << 8) | q[k];
                double t;
                std::memcpy(&t, &u, 8);
                v = (float)t;
            }
        } else if (bits == 8) {
            v = ((int)q[0] - 128) / 128.0f;  // WAV stores 8-bit unsigned
        } else if (bits == 16) {
            v = (int16_t)rd16(q) / 32768.0f;
        } else if (bits == 24) {
            int32_t s = (int32_t)((uint32_t)q[0] << 8 | (uint32_t)q[1] << 16 | (uint32_t)q[2] << 24) >> 8;
            v = s / 8388608.0f;
        } else {
            v = (int32_t)rd32(q) / 2147483648.0f;
        }
        out.samples[(size_t)i] = v;
    }
    return true;
}

bool writeWav(const std::string& path, const Audio& a, int bits, std::string& err) {
    if (a.empty()) {
        err = "nothing to write: the audio buffer is empty";
        return false;
    }
    if (bits != 16 && bits != 24 && bits != 32) {
        err = "writeWav supports 16, 24 or 32 (float) bits";
        return false;
    }
    const uint32_t bytes = (uint32_t)(bits / 8);
    const uint64_t dataLen = (uint64_t)a.samples.size() * bytes;
    if (dataLen + 64 > 0xFFFFFFFFull) {
        err = "audio too long for a RIFF file (>4 GB)";
        return false;
    }
    const bool isFloat = bits == 32;
    std::vector<uint8_t> h;
    wrTag(h, "RIFF");
    wr32(h, (uint32_t)(36 + dataLen));
    wrTag(h, "WAVE");
    wrTag(h, "fmt ");
    wr32(h, 16);
    wr16(h, isFloat ? 3 : 1);
    wr16(h, (uint16_t)a.channels);
    wr32(h, a.sampleRate);
    wr32(h, a.sampleRate * a.channels * bytes);  // byte rate
    wr16(h, (uint16_t)(a.channels * bytes));     // block align
    wr16(h, (uint16_t)bits);
    wrTag(h, "data");
    wr32(h, (uint32_t)dataLen);
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        err = "cannot write " + path;
        return false;
    }
    std::fwrite(h.data(), 1, h.size(), f);
    std::vector<uint8_t> block;
    block.reserve(1 << 16);
    for (size_t i = 0; i < a.samples.size(); i++) {
        float v = std::min(1.0f, std::max(-1.0f, a.samples[i]));
        if (isFloat) {
            uint32_t u;
            std::memcpy(&u, &v, 4);
            wr32(block, u);
        } else if (bits == 16) {
            int32_t s = (int32_t)std::lround(v * 32767.0f);
            wr16(block, (uint16_t)(int16_t)s);
        } else {
            int32_t s = (int32_t)std::lround(v * 8388607.0f);
            block.push_back((uint8_t)(s & 0xFF));
            block.push_back((uint8_t)((s >> 8) & 0xFF));
            block.push_back((uint8_t)((s >> 16) & 0xFF));
        }
        if (block.size() >= (1 << 16)) {
            std::fwrite(block.data(), 1, block.size(), f);
            block.clear();
        }
    }
    if (!block.empty()) std::fwrite(block.data(), 1, block.size(), f);
    std::fclose(f);
    return true;
}

void toMono(Audio& a) {
    if (a.channels <= 1 || a.empty()) return;
    const uint64_t n = a.frames();
    std::vector<float> mono((size_t)n, 0.0f);
    for (uint64_t i = 0; i < n; i++) {
        double acc = 0;
        for (uint32_t c = 0; c < a.channels; c++) acc += a.samples[(size_t)(i * a.channels + c)];
        mono[(size_t)i] = (float)(acc / a.channels);
    }
    a.samples.swap(mono);
    a.channels = 1;
}

}  // namespace audio
}  // namespace ofc
