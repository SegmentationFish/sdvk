// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Segmented Fish
// sdvk - see LICENSE at the repository root. No warranty.
// Minimal safetensors reader (F32/F16/I64/I32 headers; tensor data read on demand).
#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace ofc {

struct STTensor {
    std::string name;
    std::string dtype;
    std::vector<uint64_t> shape;
    uint64_t begin = 0, end = 0;  // byte offsets within the data section
};

class Safetensors {
public:
    bool open(const std::string& path);
    // parse only if this file has not been opened yet (lets several views share one handle)
    bool ensureOpen(const std::string& path);
    bool isOpen() const { return !ts_.empty(); }
    const std::string& path() const { return path_; }

    const std::vector<STTensor>& tensors() const { return ts_; }
    const STTensor* find(const std::string& name) const;
    bool read(const STTensor& t, void* dst, uint64_t bytes) const;
    std::vector<float> readF32(const std::string& name) const;
    uint64_t numel(const STTensor& t) const;
    uint64_t dataStart() const { return dataStart_; }
    const std::string& error() const { return err_; }

private:
    mutable std::ifstream f_;
    std::string path_;
    std::vector<STTensor> ts_;
    std::unordered_map<std::string, size_t> idx_;
    uint64_t dataStart_ = 0;
    mutable std::string err_;
};

}  // namespace ofc
