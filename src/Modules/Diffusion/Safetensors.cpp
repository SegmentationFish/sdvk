// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
#include "Safetensors.hpp"

#include <cstring>
#include <stdexcept>

namespace ofc {

namespace {

struct J {
    enum Type { OBJECT, ARRAY, STRING, NUMBER, BOOL, NUL } t = NUL;
    std::string s;
    double num = 0;
    bool b = false;
    std::vector<std::pair<std::string, J>> obj;
    std::vector<J> arr;
};

struct Parser {
    const char* p;
    const char* e;
    void ws() {
        while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
    }
    std::string str() {
        std::string out;
        if (p >= e || *p != '"') throw std::runtime_error("json: expected string");
        p++;
        while (p < e && *p != '"') {
            if (*p == '\\' && p + 1 < e) {
                p++;
                switch (*p) {
                    case 'n': out.push_back('\n'); break;
                    case 't': out.push_back('\t'); break;
                    case 'r': out.push_back('\r'); break;
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    default: out.push_back(*p); break;
                }
                p++;
            } else out.push_back(*p++);
        }
        if (p >= e) throw std::runtime_error("json: unterminated string");
        p++;
        return out;
    }
    J val() {
        ws();
        J v;
        if (p >= e) throw std::runtime_error("json: eof");
        if (*p == '{') {
            v.t = J::OBJECT;
            p++;
            ws();
            if (p < e && *p == '}') {
                p++;
                return v;
            }
            while (true) {
                ws();
                std::string k = str();
                ws();
                if (p >= e || *p != ':') throw std::runtime_error("json: expected ':'");
                p++;
                J x = val();
                v.obj.push_back({std::move(k), std::move(x)});
                ws();
                if (p < e && *p == ',') {
                    p++;
                    continue;
                }
                if (p < e && *p == '}') {
                    p++;
                    break;
                }
                throw std::runtime_error("json: expected ',' or '}'");
            }
            return v;
        }
        if (*p == '[') {
            v.t = J::ARRAY;
            p++;
            ws();
            if (p < e && *p == ']') {
                p++;
                return v;
            }
            while (true) {
                v.arr.push_back(val());
                ws();
                if (p < e && *p == ',') {
                    p++;
                    continue;
                }
                if (p < e && *p == ']') {
                    p++;
                    break;
                }
                throw std::runtime_error("json: expected ',' or ']'");
            }
            return v;
        }
        if (*p == '"') {
            v.t = J::STRING;
            v.s = str();
            return v;
        }
        if (std::strncmp(p, "true", 4) == 0) {
            v.t = J::BOOL;
            v.b = true;
            p += 4;
            return v;
        }
        if (std::strncmp(p, "false", 5) == 0) {
            v.t = J::BOOL;
            v.b = false;
            p += 5;
            return v;
        }
        if (std::strncmp(p, "null", 4) == 0) {
            v.t = J::NUL;
            p += 4;
            return v;
        }
        char* endp = nullptr;
        v.num = std::strtod(p, &endp);
        if (endp == p) throw std::runtime_error("json: bad number");
        v.t = J::NUMBER;
        p = endp;
        return v;
    }
};

const J* field(const J& o, const char* name) {
    for (auto& kv : o.obj)
        if (kv.first == name) return &kv.second;
    return nullptr;
}

float fp16ToFp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FF;
    uint32_t out;
    if (exp == 0) {
        if (man == 0) {
            out = sign;
        } else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) {
                man <<= 1;
                exp--;
            }
            man &= 0x3FF;
            out = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1F) {
        out = sign | 0x7F800000u | (man << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

}  // namespace

bool Safetensors::open(const std::string& path) {
    err_.clear();
    if (f_.is_open()) f_.close();
    ts_.clear();
    idx_.clear();
    path_ = path;
    f_.open(path, std::ios::binary);
    if (!f_) {
        err_ = "cannot open " + path;
        return false;
    }
    uint64_t hlen = 0;
    f_.read((char*)&hlen, 8);
    if (!f_ || hlen == 0 || hlen > (64ull << 20)) {
        err_ = "bad header length";
        return false;
    }
    std::string hdr(hlen, '\0');
    f_.read(hdr.data(), (std::streamsize)hlen);
    if (!f_) {
        err_ = "truncated header";
        return false;
    }
    dataStart_ = 8 + hlen;
    try {
        Parser ps{hdr.data(), hdr.data() + hdr.size()};
        J root = ps.val();
        ts_.clear();
        idx_.clear();
        for (auto& kv : root.obj) {
            if (kv.first == "__metadata__") continue;
            const J& o = kv.second;
            STTensor t;
            t.name = kv.first;
            if (auto* d = field(o, "dtype")) t.dtype = d->s;
            if (auto* sh = field(o, "shape"))
                for (auto& x : sh->arr) t.shape.push_back((uint64_t)x.num);
            if (auto* of = field(o, "data_offsets")) {
                t.begin = (uint64_t)of->arr[0].num;
                t.end = (uint64_t)of->arr[1].num;
            }
            idx_[t.name] = ts_.size();
            ts_.push_back(std::move(t));
        }
    } catch (const std::exception& ex) {
        err_ = std::string("header parse: ") + ex.what();
        return false;
    }
    return true;
}

bool Safetensors::ensureOpen(const std::string& path) {
    if (isOpen() && path_ == path) return true;
    return open(path);
}

const STTensor* Safetensors::find(const std::string& name) const {
    auto it = idx_.find(name);
    return it == idx_.end() ? nullptr : &ts_[it->second];
}

bool Safetensors::read(const STTensor& t, void* dst, uint64_t bytes) const {
    if (bytes > t.end - t.begin) {
        err_ = "read beyond tensor";
        return false;
    }
    f_.clear();
    f_.seekg((std::streamoff)(dataStart_ + t.begin), std::ios::beg);
    f_.read((char*)dst, (std::streamsize)bytes);
    if (!f_) {
        err_ = "short read";
        return false;
    }
    return true;
}

std::vector<float> Safetensors::readF32(const std::string& name) const {
    const STTensor* t = find(name);
    if (!t) throw std::runtime_error("tensor not found: " + name);
    if (t->dtype != "F32" && t->dtype != "F16" && t->dtype != "BF16")
        throw std::runtime_error("tensor " + name + " is " + t->dtype + ", not F32/F16/BF16");
    std::vector<float> v((size_t)numel(*t));
    if (t->dtype == "F32") {
        if (!read(*t, v.data(), v.size() * 4)) throw std::runtime_error(err_);
    } else if (t->dtype == "F16") {
        std::vector<uint16_t> h(v.size());
        if (!read(*t, h.data(), h.size() * 2)) throw std::runtime_error(err_);
        for (size_t i = 0; i < h.size(); i++) v[i] = fp16ToFp32(h[i]);
    } else {
        std::vector<uint16_t> h(v.size());
        if (!read(*t, h.data(), h.size() * 2)) throw std::runtime_error(err_);
        for (size_t i = 0; i < h.size(); i++) {
            uint32_t bits = (uint32_t)h[i] << 16;
            std::memcpy(&v[i], &bits, 4);
        }
    }
    return v;
}

uint64_t Safetensors::numel(const STTensor& t) const {
    uint64_t n = 1;
    for (auto d : t.shape) n *= d;
    return n;
}

}  // namespace ofc
