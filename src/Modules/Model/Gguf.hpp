// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
#pragma once
// GGUF reader: the llama.cpp-ecosystem model container (input format for the compat layer)
// supports: v2/v3 headers, all metadata KV types, tensor table, F32/F16/BF16/Q4_0/Q8_0 dequantization
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <fstream>

namespace ofc {

enum class GgmlType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    BF16 = 30,
};

const char* ggmlTypeName(GgmlType t);
uint64_t ggmlTypeBlockElems(GgmlType t);  // elements per block (scalars = 1)
uint64_t ggmlTypeBlockBytes(GgmlType t);  // bytes per block (scalars = element size)

struct GgufTensor {
    std::string name;
    std::vector<uint64_t> dims;  // original order: dims[0] = fastest-varying dim (row length, ggml ne[0])
    GgmlType type = GgmlType::F32;
    uint64_t offset = 0;  // offset relative to the data section start
    uint64_t nbytes = 0;

    uint64_t nElems() const;
    // logical shape (row-major convention): (rows, cols) = (dims[1], dims[0]); rows = 1 for 1-D
    std::pair<uint64_t, uint64_t> shape() const;
};

// metadata value (only the aggregated forms we need)
struct GgufMeta {
    uint32_t type = 0;  // GGUF_TYPE_*
    uint64_t u = 0;     // all integer scalars
    double f = 0;       // FLOAT32/FLOAT64
    bool b = false;     // BOOL
    std::string s;      // STRING
    // arrays:
    uint32_t arrType = 0;
    std::vector<std::string> strs;
    std::vector<uint64_t> u64s;
    std::vector<double> fs;
    std::vector<bool> bools;

    bool isArray() const { return type == 9; }
    bool isStr() const { return type == 8; }
    bool isNum() const { return type != 8 && type != 9; }
};

enum GgufMetaType : uint32_t {
    GGUF_UINT8 = 0,
    GGUF_INT8 = 1,
    GGUF_UINT16 = 2,
    GGUF_INT16 = 3,
    GGUF_UINT32 = 4,
    GGUF_INT32 = 5,
    GGUF_FLOAT32 = 6,
    GGUF_BOOL = 7,
    GGUF_STRING = 8,
    GGUF_ARRAY = 9,
    GGUF_UINT64 = 10,
    GGUF_INT64 = 11,
    GGUF_FLOAT64 = 12,
};

class GgufFile {
public:
    bool open(const std::string& path, std::string* err = nullptr);

    uint32_t version() const { return version_; }
    size_t tensorCount() const { return tensors_.size(); }
    const std::vector<GgufTensor>& tensors() const { return tensors_; }
    const GgufTensor* find(const std::string& name) const;
    uint64_t dataOffset() const { return dataOffset_; }
    uint64_t fileSize() const { return fileSize_; }

    // metadata
    bool has(const std::string& key) const { return meta_.count(key) != 0; }
    const GgufMeta* meta(const std::string& key) const;
    uint64_t getU64(const std::string& key, uint64_t def = 0) const;
    int64_t getI64(const std::string& key, int64_t def = 0) const;
    double getF64(const std::string& key, double def = 0) const;
    float getF32(const std::string& key, float def = 0) const;
    std::string getStr(const std::string& key, const std::string& def = "") const;
    // common tensor hyperparameters (fetched by arch prefix)
    uint64_t archU64(const std::string& arch, const std::string& key, uint64_t def = 0) const;
    float archF32(const std::string& arch, const std::string& key, float def = 0) const;

    // tensor data
    bool readRaw(const GgufTensor& t, std::vector<uint8_t>& out) const;
    // dequantize to f32 (row-major (rows, cols), matching shape())
    std::vector<float> readFloat(const std::string& name, std::string* err = nullptr) const;
    // raw Q8_0 export: int8 payload + fp32 scale (32 elements per block), file-flat order preserved
    // (PyTorch row-major [N,K], N = out channels, K = in*ksize) for in-shader dequantization
    bool readQ8(const std::string& name, std::vector<int8_t>& q, std::vector<float>& scales,
                std::string* err = nullptr) const;
    // Same data, straight into the staging map: no intermediate copy, which keeps a 2.5 GB
    // upload from tripling the process commit (see readQ8 for the layout).
    bool readQ8Into(const std::string& name, int8_t* qDst, float* scaleDst, std::string* err = nullptr) const;

    const std::map<std::string, GgufMeta>& allMeta() const { return meta_; }

private:
    std::string path_;
    mutable std::ifstream f_;
    uint32_t version_ = 0;
    uint64_t dataOffset_ = 0;
    uint64_t fileSize_ = 0;
    std::map<std::string, GgufMeta> meta_;
    std::vector<GgufTensor> tensors_;
    std::map<std::string, size_t> byName_;

    bool readExact(void* dst, size_t n) const;
    bool readString(std::string& s) const;
    bool readValue(uint32_t type, GgufMeta& m) const;  // scalars (non-array)
    bool readArrayItems(uint32_t etype, uint64_t count, GgufMeta& m) const;
};

float fp16ToFp32(uint16_t h);
uint16_t readU16(const uint8_t* p);

}  // namespace ofc
