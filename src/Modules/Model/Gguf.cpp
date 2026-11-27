// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
// GGUF reader (format: magic GGUF / version / counts / KV metadata / tensor table / aligned data section)
#include "Gguf.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>

namespace ofc {

float fp16ToFp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) f = sign << 31;
        else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) {
                man <<= 1;
                exp--;
            }
            man &= 0x3FF;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

uint16_t readU16(const uint8_t* p) { return (uint16_t)p[0] | ((uint16_t)p[1] << 8); }

const char* ggmlTypeName(GgmlType t) {
    switch (t) {
        case GgmlType::F32: return "F32";
        case GgmlType::F16: return "F16";
        case GgmlType::Q4_0: return "Q4_0";
        case GgmlType::Q4_1: return "Q4_1";
        case GgmlType::Q5_0: return "Q5_0";
        case GgmlType::Q5_1: return "Q5_1";
        case GgmlType::Q8_0: return "Q8_0";
        case GgmlType::Q8_1: return "Q8_1";
        case GgmlType::Q2_K: return "Q2_K";
        case GgmlType::Q3_K: return "Q3_K";
        case GgmlType::Q4_K: return "Q4_K";
        case GgmlType::Q5_K: return "Q5_K";
        case GgmlType::Q6_K: return "Q6_K";
        case GgmlType::BF16: return "BF16";
        default: return "?";
    }
}

uint64_t ggmlTypeBlockElems(GgmlType t) {
    switch (t) {
        case GgmlType::F32:
        case GgmlType::F16:
        case GgmlType::BF16: return 1;
        case GgmlType::Q4_0:
        case GgmlType::Q4_1:
        case GgmlType::Q5_0:
        case GgmlType::Q5_1:
        case GgmlType::Q8_0:
        case GgmlType::Q8_1: return 32;
        case GgmlType::Q2_K:
        case GgmlType::Q3_K:
        case GgmlType::Q4_K:
        case GgmlType::Q5_K:
        case GgmlType::Q6_K: return 256;
        default: return 1;
    }
}

uint64_t ggmlTypeBlockBytes(GgmlType t) {
    switch (t) {
        case GgmlType::F32: return 4;
        case GgmlType::F16:
        case GgmlType::BF16: return 2;
        case GgmlType::Q4_0: return 2 + 16;
        case GgmlType::Q4_1: return 2 + 2 + 16;
        case GgmlType::Q5_0: return 2 + 4 + 16;
        case GgmlType::Q5_1: return 2 + 2 + 4 + 16;
        case GgmlType::Q8_0: return 2 + 32;
        case GgmlType::Q8_1: return 2 + 2 + 32;
        // K-quants pack 256 values per super-block (llama.cpp block_q*_K layouts).
        case GgmlType::Q2_K: return 2 + 2 + 16 + 64;
        case GgmlType::Q3_K: return 2 + 12 + 32 + 64;
        case GgmlType::Q4_K: return 2 + 2 + 12 + 128;
        case GgmlType::Q5_K: return 2 + 2 + 12 + 32 + 128;
        case GgmlType::Q6_K: return 2 + 16 + 64 + 128;
        default: return 1;
    }
}

uint64_t GgufTensor::nElems() const {
    uint64_t n = 1;
    for (uint64_t d : dims) n *= d;
    return n;
}

std::pair<uint64_t, uint64_t> GgufTensor::shape() const {
    if (dims.size() == 1) return {1, dims[0]};
    if (dims.size() == 2) return {dims[1], dims[0]};
    uint64_t rows = 1;
    for (size_t i = 1; i < dims.size(); i++) rows *= dims[i];
    return {rows, dims[0]};
}

bool GgufFile::readExact(void* dst, size_t n) const {
    f_.read((char*)dst, (std::streamsize)n);
    return (bool)f_;
}

bool GgufFile::readString(std::string& s) const {
    uint64_t len = 0;
    if (!readExact(&len, 8)) return false;
    if (len > (1ull << 30)) return false;
    s.resize((size_t)len);
    if (len) return readExact(s.data(), (size_t)len);
    return true;
}

bool GgufFile::readValue(uint32_t type, GgufMeta& m) const {
    m.type = type;
    switch (type) {
        case GGUF_UINT8: {
            uint8_t v;
            if (!readExact(&v, 1)) return false;
            m.u = v;
            return true;
        }
        case GGUF_INT8: {
            int8_t v;
            if (!readExact(&v, 1)) return false;
            m.u = (uint64_t)(int64_t)v;
            return true;
        }
        case GGUF_UINT16: {
            uint16_t v;
            if (!readExact(&v, 2)) return false;
            m.u = v;
            return true;
        }
        case GGUF_INT16: {
            int16_t v;
            if (!readExact(&v, 2)) return false;
            m.u = (uint64_t)(int64_t)v;
            return true;
        }
        case GGUF_UINT32: {
            uint32_t v;
            if (!readExact(&v, 4)) return false;
            m.u = v;
            return true;
        }
        case GGUF_INT32: {
            int32_t v;
            if (!readExact(&v, 4)) return false;
            m.u = (uint64_t)(int64_t)v;
            return true;
        }
        case GGUF_FLOAT32: {
            float v;
            if (!readExact(&v, 4)) return false;
            m.f = v;
            return true;
        }
        case GGUF_BOOL: {
            uint8_t v;
            if (!readExact(&v, 1)) return false;
            m.b = (v != 0);
            return true;
        }
        case GGUF_STRING: return readString(m.s);
        case GGUF_UINT64: {
            uint64_t v;
            if (!readExact(&v, 8)) return false;
            m.u = v;
            return true;
        }
        case GGUF_INT64: {
            int64_t v;
            if (!readExact(&v, 8)) return false;
            m.u = (uint64_t)v;
            return true;
        }
        case GGUF_FLOAT64: {
            double v;
            if (!readExact(&v, 8)) return false;
            m.f = v;
            return true;
        }
        default: return false;  // ARRAY is handled by the caller
    }
}

bool GgufFile::readArrayItems(uint32_t etype, uint64_t count, GgufMeta& m) const {
    m.arrType = etype;
    if (count > (1ull << 32)) return false;
    switch (etype) {
        case GGUF_STRING: {
            m.strs.resize((size_t)count);
            for (uint64_t i = 0; i < count; i++)
                if (!readString(m.strs[(size_t)i])) return false;
            return true;
        }
        case GGUF_UINT8:
        case GGUF_INT8: {
            m.u64s.resize((size_t)count);
            for (uint64_t i = 0; i < count; i++) {
                if (etype == GGUF_UINT8) {
                    uint8_t v;
                    if (!readExact(&v, 1)) return false;
                    m.u64s[(size_t)i] = v;
                } else {
                    int8_t v;
                    if (!readExact(&v, 1)) return false;
                    m.u64s[(size_t)i] = (uint64_t)(int64_t)v;
                }
            }
            return true;
        }
        case GGUF_UINT16:
        case GGUF_INT16: {
            m.u64s.resize((size_t)count);
            for (uint64_t i = 0; i < count; i++) {
                if (etype == GGUF_UINT16) {
                    uint16_t v;
                    if (!readExact(&v, 2)) return false;
                    m.u64s[(size_t)i] = v;
                } else {
                    int16_t v;
                    if (!readExact(&v, 2)) return false;
                    m.u64s[(size_t)i] = (uint64_t)(int64_t)v;
                }
            }
            return true;
        }
        case GGUF_UINT32:
        case GGUF_INT32: {
            m.u64s.resize((size_t)count);
            for (uint64_t i = 0; i < count; i++) {
                if (etype == GGUF_UINT32) {
                    uint32_t v;
                    if (!readExact(&v, 4)) return false;
                    m.u64s[(size_t)i] = v;
                } else {
                    int32_t v;
                    if (!readExact(&v, 4)) return false;
                    m.u64s[(size_t)i] = (uint64_t)(int64_t)v;
                }
            }
            return true;
        }
        case GGUF_UINT64:
        case GGUF_INT64: {
            m.u64s.resize((size_t)count);
            for (uint64_t i = 0; i < count; i++) {
                uint64_t v;
                if (!readExact(&v, 8)) return false;
                m.u64s[(size_t)i] = v;
            }
            return true;
        }
        case GGUF_FLOAT32: {
            m.fs.resize((size_t)count);
            for (uint64_t i = 0; i < count; i++) {
                float v;
                if (!readExact(&v, 4)) return false;
                m.fs[(size_t)i] = v;
            }
            return true;
        }
        case GGUF_FLOAT64: {
            m.fs.resize((size_t)count);
            for (uint64_t i = 0; i < count; i++) {
                double v;
                if (!readExact(&v, 8)) return false;
                m.fs[(size_t)i] = v;
            }
            return true;
        }
        case GGUF_BOOL: {
            m.bools.resize((size_t)count);
            for (uint64_t i = 0; i < count; i++) {
                uint8_t v;
                if (!readExact(&v, 1)) return false;
                m.bools[(size_t)i] = (v != 0);
            }
            return true;
        }
        default: return false;
    }
}

bool GgufFile::open(const std::string& path, std::string* err, const std::string& prefix) {
    path_ = path;
    if (f_.is_open()) f_.close();
    meta_.clear();
    tensors_.clear();
    byName_.clear();
    f_.open(path, std::ios::binary);
    if (!f_) {
        if (err) *err = "cannot open " + path;
        return false;
    }
    f_.seekg(0, std::ios::end);
    fileSize_ = (uint64_t)f_.tellg();
    f_.seekg(0, std::ios::beg);

    char magic[4];
    uint32_t version = 0;
    uint64_t nTensors = 0, nKv = 0;
    if (!readExact(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
        if (err) *err = "bad magic (not a GGUF file)";
        return false;
    }
    if (!readExact(&version, 4) || (version != 2 && version != 3)) {
        if (err) *err = "unsupported GGUF version";
        return false;
    }
    version_ = version;
    if (!readExact(&nTensors, 8) || !readExact(&nKv, 8)) {
        if (err) *err = "truncated header";
        return false;
    }
    if (nTensors > (1ull << 24) || nKv > (1ull << 24)) {
        if (err) *err = "implausible counts";
        return false;
    }

    for (uint64_t i = 0; i < nKv; i++) {
        std::string key;
        if (!readString(key)) {
            if (err) *err = "truncated kv key";
            return false;
        }
        uint32_t type = 0;
        if (!readExact(&type, 4)) {
            if (err) *err = "truncated kv type";
            return false;
        }
        GgufMeta m;
        if (type == GGUF_ARRAY) {
            uint32_t etype = 0;
            uint64_t count = 0;
            if (!readExact(&etype, 4) || !readExact(&count, 8)) {
                if (err) *err = "truncated array header";
                return false;
            }
            m.type = GGUF_ARRAY;
            if (!readArrayItems(etype, count, m)) {
                if (err) *err = "truncated array items for " + key;
                return false;
            }
        } else {
            if (!readValue(type, m)) {
                if (err) *err = "truncated kv value for " + key;
                return false;
            }
        }
        meta_[key] = std::move(m);
    }

    tensors_.reserve((size_t)nTensors);
    for (uint64_t i = 0; i < nTensors; i++) {
        GgufTensor t;
        if (!readString(t.name)) {
            if (err) *err = "truncated tensor name";
            return false;
        }
        uint32_t nd = 0;
        if (!readExact(&nd, 4) || nd == 0 || nd > 4) {
            if (err) *err = "bad tensor ndims";
            return false;
        }
        t.dims.resize(nd);
        for (uint32_t d = 0; d < nd; d++)
            if (!readExact(&t.dims[d], 8)) {
                if (err) *err = "truncated tensor dims";
                return false;
            }
        uint32_t ty = 0;
        if (!readExact(&ty, 4)) {
            if (err) *err = "truncated tensor type";
            return false;
        }
        t.type = (GgmlType)ty;
        if (!readExact(&t.offset, 8)) {
            if (err) *err = "truncated tensor offset";
            return false;
        }
        if (!prefix.empty()) {
            if (t.name.rfind(prefix, 0) != 0) continue;
            t.name.erase(0, prefix.size());
        }
        uint64_t be = ggmlTypeBlockElems(t.type), bb = ggmlTypeBlockBytes(t.type);
        uint64_t ne = t.nElems();
        t.nbytes = (ne / be) * bb + ((ne % be) ? bb : 0);  // GGUF guarantees divisibility; round up defensively
        if (byName_.count(t.name)) {
            if (err) *err = "duplicate tensor " + t.name;
            return false;
        }
        byName_[t.name] = tensors_.size();
        tensors_.push_back(std::move(t));
    }

    uint32_t align = 32;
    if (auto it = meta_.find("general.alignment"); it != meta_.end() && it->second.isNum())
        align = (uint32_t)it->second.u;
    if (align == 0 || (align & (align - 1))) {
        if (err) *err = "bad alignment";
        return false;
    }
    uint64_t pos = (uint64_t)f_.tellg();
    dataOffset_ = (pos + align - 1) / align * align;
    if (dataOffset_ > fileSize_) {
        if (err) *err = "data offset beyond EOF";
        return false;
    }
    return true;
}

const GgufTensor* GgufFile::find(const std::string& name) const {
    auto it = byName_.find(name);
    return it == byName_.end() ? nullptr : &tensors_[it->second];
}

const GgufMeta* GgufFile::meta(const std::string& key) const {
    auto it = meta_.find(key);
    return it == meta_.end() ? nullptr : &it->second;
}

uint64_t GgufFile::getU64(const std::string& key, uint64_t def) const {
    const GgufMeta* m = meta(key);
    if (!m) return def;
    if (m->isNum()) return m->u;
    return def;
}

int64_t GgufFile::getI64(const std::string& key, int64_t def) const {
    const GgufMeta* m = meta(key);
    if (!m) return def;
    if (m->isNum()) return (int64_t)m->u;
    return def;
}

double GgufFile::getF64(const std::string& key, double def) const {
    const GgufMeta* m = meta(key);
    if (!m) return def;
    if (m->isNum()) return (m->type == GGUF_FLOAT32 || m->type == GGUF_FLOAT64) ? m->f : (double)m->u;
    return def;
}

float GgufFile::getF32(const std::string& key, float def) const { return (float)getF64(key, def); }

std::string GgufFile::getStr(const std::string& key, const std::string& def) const {
    const GgufMeta* m = meta(key);
    if (!m || !m->isStr()) return def;
    return m->s;
}

uint64_t GgufFile::archU64(const std::string& arch, const std::string& key, uint64_t def) const {
    return getU64(arch + "." + key, def);
}

float GgufFile::archF32(const std::string& arch, const std::string& key, float def) const {
    return getF32(arch + "." + key, def);
}

bool GgufFile::readRange(const GgufTensor& t, uint64_t byteOff, void* dst, uint64_t bytes) const {
    if (byteOff + bytes > t.nbytes) return false;
    f_.clear();
    f_.seekg((std::streamoff)(dataOffset_ + t.offset + byteOff), std::ios::beg);
    if (!f_) return false;
    f_.read((char*)dst, (std::streamsize)bytes);
    return (uint64_t)f_.gcount() == bytes;
}

bool GgufFile::readRaw(const GgufTensor& t, std::vector<uint8_t>& out) const {
    out.resize((size_t)t.nbytes);
    f_.clear();
    f_.seekg((std::streamoff)(dataOffset_ + t.offset));
    if (!readExact(out.data(), (size_t)t.nbytes)) return false;
    return true;
}

// ---------------------------------------------------------------- K-quants
// Super-block layouts and the 6-bit scale/min unpacking mirror llama.cpp's
// ggml-quants.c (block_q*_K). The engine does not run K-quant GEMMs: these
// tensors are dequantized on load and requantized to the Q8_0 path.

static inline void getScaleMinK4(int j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (uint8_t)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

static void dequantQ2K(const uint8_t* p, float* y, uint64_t ne) {
    for (uint64_t b = 0; b < ne / 256; b++) {
        const uint8_t* blk = p + b * 84;
        const uint8_t* scales = blk;
        const uint8_t* q = blk + 16;
        float d = fp16ToFp32(readU16(blk + 80));
        float dmin = fp16ToFp32(readU16(blk + 82));
        float* out = y + b * 256;
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc = scales[is++];
                float dl = d * (float)(sc & 0x0F), ml = dmin * (float)(sc >> 4);
                for (int l = 0; l < 16; l++)
                    out[n + j * 32 + l] = dl * (float)((q[l] >> shift) & 3) - ml;
                sc = scales[is++];
                dl = d * (float)(sc & 0x0F);
                ml = dmin * (float)(sc >> 4);
                for (int l = 0; l < 16; l++)
                    out[n + j * 32 + 16 + l] = dl * (float)((q[l + 16] >> shift) & 3) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

static void dequantQ3K(const uint8_t* p, float* y, uint64_t ne) {
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    for (uint64_t b = 0; b < ne / 256; b++) {
        const uint8_t* blk = p + b * 110;
        const uint8_t* hm = blk;
        const uint8_t* q = blk + 32;
        const uint8_t* scalesRaw = blk + 96;
        float d_all = fp16ToFp32(readU16(blk + 108));
        uint32_t aux[4];
        std::memcpy(aux, scalesRaw, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = (const int8_t*)aux;
        uint8_t m = 1;
        int is = 0;
        float* out = y + b * 256;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d_all * (float)(scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    out[n + j * 32 + l] = dl * (float)((int)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                dl = d_all * (float)(scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    out[n + j * 32 + 16 + l] = dl * (float)((int)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

static void dequantQ4K(const uint8_t* p, float* y, uint64_t ne) {
    for (uint64_t b = 0; b < ne / 256; b++) {
        const uint8_t* blk = p + b * 144;
        float d = fp16ToFp32(readU16(blk));
        float dmin = fp16ToFp32(readU16(blk + 2));
        const uint8_t* scales = blk + 4;
        const uint8_t* q = blk + 16;
        float* out = y + b * 256;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            getScaleMinK4(is + 0, scales, sc, m);
            float d1 = d * (float)sc, m1 = dmin * (float)m;
            getScaleMinK4(is + 1, scales, sc, m);
            float d2 = d * (float)sc, m2 = dmin * (float)m;
            for (int l = 0; l < 32; l++) out[j + l] = d1 * (float)(q[l] & 0x0F) - m1;
            for (int l = 0; l < 32; l++) out[j + 32 + l] = d2 * (float)(q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

static void dequantQ5K(const uint8_t* p, float* y, uint64_t ne) {
    for (uint64_t b = 0; b < ne / 256; b++) {
        const uint8_t* blk = p + b * 176;
        float d = fp16ToFp32(readU16(blk));
        float dmin = fp16ToFp32(readU16(blk + 2));
        const uint8_t* scales = blk + 4;
        const uint8_t* qh = blk + 16;
        const uint8_t* ql = blk + 48;
        float* out = y + b * 256;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            getScaleMinK4(is + 0, scales, sc, m);
            float d1 = d * (float)sc, m1 = dmin * (float)m;
            getScaleMinK4(is + 1, scales, sc, m);
            float d2 = d * (float)sc, m2 = dmin * (float)m;
            for (int l = 0; l < 32; l++)
                out[j + l] = d1 * (float)((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32; l++)
                out[j + 32 + l] = d2 * (float)((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

static void dequantQ6K(const uint8_t* p, float* y, uint64_t ne) {
    for (uint64_t b = 0; b < ne / 256; b++) {
        const uint8_t* blk = p + b * 210;
        const uint8_t* ql = blk;
        const uint8_t* qh = blk + 128;
        const int8_t* scales = (const int8_t*)(blk + 192);
        float d = fp16ToFp32(readU16(blk + 208));
        float* out = y + b * 256;
        for (int n = 0; n < 256; n += 128) {
            const uint8_t* ql2 = ql + n / 2;
            const uint8_t* qh2 = qh + n / 4;
            const int8_t* sc = scales + n / 16;
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (int)((ql2[l] & 0x0F) | (((qh2[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql2[l + 32] & 0x0F) | (((qh2[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql2[l] >> 4) | (((qh2[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql2[l + 32] >> 4) | (((qh2[l] >> 6) & 3) << 4)) - 32;
                out[n + l] = d * (float)sc[is + 0] * (float)q1;
                out[n + l + 32] = d * (float)sc[is + 2] * (float)q2;
                out[n + l + 64] = d * (float)sc[is + 4] * (float)q3;
                out[n + l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
        }
    }
}

bool ggmlTypeIsKQuant(GgmlType t) {
    return t == GgmlType::Q2_K || t == GgmlType::Q3_K || t == GgmlType::Q4_K ||
           t == GgmlType::Q5_K || t == GgmlType::Q6_K;
}

void quantizeQ8_0(const float* x, int8_t* q, float* scales, uint64_t n) {
    for (uint64_t b = 0; b < n / 32; b++) {
        const float* xb = x + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) amax = std::max(amax, std::fabs(xb[i]));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        scales[b] = d;
        for (int i = 0; i < 32; i++) {
            int v = (int)std::lrint(xb[i] * id);
            q[b * 32 + i] = (int8_t)std::max(-127, std::min(127, v));
        }
    }
}

std::vector<float> GgufFile::readFloat(const std::string& name, std::string* err) const {
    const GgufTensor* tp = find(name);
    if (!tp) {
        if (err) *err = "tensor not found: " + name;
        return {};
    }
    const GgufTensor& t = *tp;
    uint64_t ne = t.nElems();
    std::vector<uint8_t> raw;
    if (!readRaw(t, raw)) {
        if (err) *err = "read failed: " + name;
        return {};
    }
    std::vector<float> out((size_t)ne);
    const uint8_t* p = raw.data();
    switch (t.type) {
        case GgmlType::F32: std::memcpy(out.data(), p, (size_t)ne * 4); return out;
        case GgmlType::F16:
            for (uint64_t i = 0; i < ne; i++) out[(size_t)i] = fp16ToFp32(readU16(p + i * 2));
            return out;
        case GgmlType::BF16:
            for (uint64_t i = 0; i < ne; i++) {
                uint32_t bits = (uint32_t)readU16(p + i * 2) << 16;
                std::memcpy(&out[(size_t)i], &bits, 4);
            }
            return out;
        case GgmlType::Q4_0: {
            uint64_t nb = ne / 32;
            for (uint64_t b = 0; b < nb; b++) {
                const uint8_t* blk = p + b * 18;
                float d = fp16ToFp32(readU16(blk));
                const uint8_t* qs = blk + 2;
                for (uint64_t i = 0; i < 16; i++) {
                    int lo = (qs[i] & 0x0F) - 8;
                    int hi = (qs[i] >> 4) - 8;
                    out[(size_t)(b * 32 + i * 2)] = d * (float)lo;
                    out[(size_t)(b * 32 + i * 2 + 1)] = d * (float)hi;
                }
            }
            return out;
        }
        case GgmlType::Q8_0: {
            uint64_t nb = ne / 32;
            for (uint64_t b = 0; b < nb; b++) {
                const uint8_t* blk = p + b * 34;
                float d = fp16ToFp32(readU16(blk));
                const int8_t* qs = (const int8_t*)(blk + 2);
                for (uint64_t i = 0; i < 32; i++) out[(size_t)(b * 32 + i)] = d * (float)qs[i];
            }
            return out;
        }
        case GgmlType::Q2_K:
        case GgmlType::Q3_K:
        case GgmlType::Q4_K:
        case GgmlType::Q5_K:
        case GgmlType::Q6_K: {
            if (ne % 256) {
                if (err) *err = std::string("K-quant tensor with non-256-multiple length: ") + name;
                return {};
            }
            switch (t.type) {
                case GgmlType::Q2_K: dequantQ2K(p, out.data(), ne); break;
                case GgmlType::Q3_K: dequantQ3K(p, out.data(), ne); break;
                case GgmlType::Q4_K: dequantQ4K(p, out.data(), ne); break;
                case GgmlType::Q5_K: dequantQ5K(p, out.data(), ne); break;
                default: dequantQ6K(p, out.data(), ne); break;
            }
            return out;
        }
        default:
            if (err) *err = std::string("unsupported quant type: ") + ggmlTypeName(t.type);
            return {};
    }
}

bool GgufFile::readQ8(const std::string& name, std::vector<int8_t>& q, std::vector<float>& scales,
                      std::string* err) const {
    const GgufTensor* tp = find(name);
    if (!tp) {
        if (err) *err = "tensor not found: " + name;
        return false;
    }
    const GgufTensor& t = *tp;
    if (t.type != GgmlType::Q8_0) {
        if (err) *err = "tensor " + name + " is " + ggmlTypeName(t.type) + ", not Q8_0";
        return false;
    }
    uint64_t ne = t.nElems();
    std::vector<uint8_t> raw;
    if (!readRaw(t, raw)) {
        if (err) *err = "read failed: " + name;
        return false;
    }
    uint64_t nb = ne / 32;
    q.resize((size_t)ne);
    scales.resize((size_t)nb);
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t* blk = raw.data() + b * 34;
        scales[(size_t)b] = fp16ToFp32(readU16(blk));
        std::memcpy(q.data() + (size_t)b * 32, blk + 2, 32);
    }
    return true;
}

bool GgufFile::readQ8Into(const std::string& name, int8_t* qDst, float* scaleDst, std::string* err) const {
    const GgufTensor* tp = find(name);
    if (!tp) {
        if (err) *err = "tensor not found: " + name;
        return false;
    }
    const GgufTensor& t = *tp;
    if (t.type != GgmlType::Q8_0) {
        if (err) *err = "tensor " + name + " is " + ggmlTypeName(t.type) + ", not Q8_0";
        return false;
    }
    uint64_t nb = t.nElems() / 32;
    f_.clear();
    f_.seekg((std::streamoff)(dataOffset_ + t.offset));
    uint8_t block[34];
    for (uint64_t b = 0; b < nb; b++) {
        if (!readExact(block, 34)) {
            if (err) *err = "read failed: " + name;
            return false;
        }
        scaleDst[b] = fp16ToFp32(readU16(block));
        std::memcpy(qDst + b * 32, block + 2, 32);
    }
    return true;
}

}  // namespace ofc
