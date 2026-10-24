#pragma once
// GGUF 读取器：llama.cpp 生态的模型容器（兼容层的输入格式）
// 支持：v2/v3 header、全类型元数据 KV、张量表、F32/F16/BF16/Q4_0/Q8_0 反量化
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
uint64_t ggmlTypeBlockElems(GgmlType t);  // 每块元素数（标量=1）
uint64_t ggmlTypeBlockBytes(GgmlType t);  // 每块字节数（标量=元素大小）

struct GgufTensor {
    std::string name;
    std::vector<uint64_t> dims;  // 原始顺序：dims[0]=最快变化维（行内长度，ggml ne[0]）
    GgmlType type = GgmlType::F32;
    uint64_t offset = 0;  // 相对数据段起点
    uint64_t nbytes = 0;

    uint64_t nElems() const;
    // 逻辑形状（行优先习惯）：(rows, cols) = (dims[1], dims[0])；一维时 rows=1
    std::pair<uint64_t, uint64_t> shape() const;
};

// 元数据值（只保留我们需要的聚合形式）
struct GgufMeta {
    uint32_t type = 0;  // GGUF_TYPE_*
    uint64_t u = 0;     // 所有整型标量
    double f = 0;       // FLOAT32/FLOAT64
    bool b = false;     // BOOL
    std::string s;      // STRING
    // 数组：
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

    // 元数据
    bool has(const std::string& key) const { return meta_.count(key) != 0; }
    const GgufMeta* meta(const std::string& key) const;
    uint64_t getU64(const std::string& key, uint64_t def = 0) const;
    int64_t getI64(const std::string& key, int64_t def = 0) const;
    double getF64(const std::string& key, double def = 0) const;
    float getF32(const std::string& key, float def = 0) const;
    std::string getStr(const std::string& key, const std::string& def = "") const;
    // 常见张量超参（按 arch 前缀取）
    uint64_t archU64(const std::string& arch, const std::string& key, uint64_t def = 0) const;
    float archF32(const std::string& arch, const std::string& key, float def = 0) const;

    // 张量数据
    bool readRaw(const GgufTensor& t, std::vector<uint8_t>& out) const;
    // 反量化为 f32（行优先 (rows, cols)，与 shape() 对应）
    std::vector<float> readFloat(const std::string& name, std::string* err = nullptr) const;
    // Q8_0 原始导出：int8 载荷 + fp32 scale（每 32 元素一块），保持文件平坦顺序
    // （PyTorch 行优先 [N,K]，N=out 通道, K=in*ksize），供 GPU shader 内反量化
    bool readQ8(const std::string& name, std::vector<int8_t>& q, std::vector<float>& scales,
                std::string* err = nullptr) const;

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
    bool readValue(uint32_t type, GgufMeta& m) const;  // 标量（非数组）
    bool readArrayItems(uint32_t etype, uint64_t count, GgufMeta& m) const;
};

float fp16ToFp32(uint16_t h);
uint16_t readU16(const uint8_t* p);

}  // namespace ofc
