// GPU tensor ops with host-side autograd tape (mirrors Modules/Core/Ops.hpp).
#pragma once

#include "vk_engine.hpp"
#include "Modules/Core/Tensor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace ofc {
namespace gpu {

struct GTensor {
    Shape shape;
    BufRef data;
    bool requiresGrad = false;
    BufRef grad;
    std::string name;
    std::function<void()> backward;
    bool persistent = false;
};
using GTPtr = std::shared_ptr<GTensor>;

struct GContext {
    Engine eng;
    Arena paramArena;      // persistent weights
    Arena paramGradArena;  // persistent weight grads
    Arena scratchArena;    // persistent small buffers (positions, masks, partials...)
    std::vector<GTPtr> tape;
    BufRef hp;                   // 设备侧超参（lr/b1/b2/eps/wd/invBc1/invBc2/b2t）——重放前提
    bool stepPreZeroed = false;  // true: 训练驱动已整块清零 step arena（跳过逐张量清零）
    uint64_t paramBytes = 0;
    bool allocGradArena = true;  // 纯推理置 false，省下一份 paramBytes 显存
    uint64_t stepBytes = 2048ull << 20;
    uint64_t scratchBytes = 256ull << 20;
};
extern GContext G;

void gInit(const std::string& spvDir);
void gResizeScratch(uint64_t bytes);
void gShutdown();
uint64_t gNumelOf(const Shape& s);

// 设备侧超参缓冲
extern uint64_t gStepTensorCount;
BufRef gHpBuf();
void gUploadHp(const float* hp, int n);  // n ≤ 8

// tape handling
void gResetTape();
void gBackward(const GTPtr& loss);  // runs lambdas in reverse, records dispatches

// tensors
GTPtr gEmpty(const Shape& s, bool rg);                      // step arena, uninitialized data
GTPtr gZeros(const Shape& s, bool rg);                      // step arena, zero-filled
GTPtr gParam(const Shape& s, const char* name);             // persistent param (requires grad)
GTPtr gViewShape(const GTPtr& x, const Shape& s);           // zero-copy reshape (data unchanged)
GTPtr gFromViewData(const Shape& s, BufRef data, bool rg);  // wrap existing buffer region

// ops (all mirror the CPU versions' semantics)
GTPtr gMatmul(const GTPtr& a, const GTPtr& b);
GTPtr gMatmulT(const GTPtr& a, const GTPtr& b);
GTPtr gMoeOut(const GTPtr& flat, const GTPtr& w1, const GTPtr& b1, const GTPtr& w2, const GTPtr& b2,
              const GTPtr& wSparse, size_t nGroups, size_t fg);
GTPtr gAdd(const GTPtr& a, const GTPtr& b);
GTPtr gMul(const GTPtr& a, const GTPtr& b);
GTPtr gScale(const GTPtr& a, float s, bool track = true);
GTPtr gGelu(const GTPtr& x);
GTPtr gSigmoid(const GTPtr& x);
GTPtr gSoftmaxLast(const GTPtr& x);
GTPtr gAttnSoftmax(const GTPtr& sloc, const GTPtr& smem, const BufRef& mask, size_t T, size_t M, float scale,
                   bool hasMask);
GTPtr gRmsNorm(const GTPtr& x, const GTPtr& w, float eps);
GTPtr gRowSum(const GTPtr& x);     // [.., d] -> [.., 1]
GTPtr gMeanSeq(const GTPtr& x);    // {B,T,d} -> {B,d}  (sequence mean pool)
GTPtr gReduceAll(const GTPtr& x);  // -> scalar {1}
GTPtr gMeanAll(const GTPtr& x);    // -> scalar {1} with 1/n scaling
GTPtr gCrossEntropy(const GTPtr& logits, const BufRef& tgt, const BufRef& mask, size_t rows, size_t v,
                    float lossScale = 0.0f);
GTPtr gEmbedding(const GTPtr& table, const BufRef& ids, size_t n);
GTPtr gRope(const GTPtr& x, const BufRef& pos, float base, bool inverse);
GTPtr gMaskFillLast(const GTPtr& x, const BufRef& mask, size_t period);
GTPtr gSliceSeq(const GTPtr& x, size_t start, size_t len);
GTPtr gConcatSeq(const GTPtr& a, const GTPtr& b);
GTPtr gConcatLast(const std::vector<GTPtr>& xs);
GTPtr gSplitHeads(const GTPtr& x, size_t heads);
GTPtr gMergeHeads(const GTPtr& x);
GTPtr gTransposeLast2(const GTPtr& x);
GTPtr gSelectDim(const GTPtr& x, size_t dim, size_t index);
GTPtr gUnsqueezeFirst(const GTPtr& x);
GTPtr gRepeatBatch(const GTPtr& x, size_t batch);
GTPtr gTopkSparse(const GTPtr& probs, size_t k, const BufRef& countsOut, BufRef* idxOut);
void gCompact(const GTPtr& sparse, size_t k, size_t rows, size_t g, const BufRef& idx, const BufRef& tIdxOut,
              const BufRef& wColOut, const BufRef& countsOut);
GTPtr gCompactWCol(const BufRef& wCol, size_t count, const BufRef& tIdx, const BufRef& counts, size_t g, size_t gid,
                   size_t rows, const GTPtr& sparse);
GTPtr gGatherRows(const GTPtr& src, const BufRef& idx, size_t n);
GTPtr gScatterAddRowsBase(const GTPtr& base, const BufRef& idx, const GTPtr& src, size_t n);
GTPtr gOneHotBalance(const GTPtr& probs, const BufRef& counts, size_t rows, size_t groups);
GTPtr gWrapView(const Shape& s, const BufRef& data, const BufRef& grad, bool rg);  // leaf-like view
GTPtr gCompactView(const BufRef& buf, const Shape& s, const BufRef& grad, bool rg, const BufRef& tIdx,
                   const BufRef& counts, size_t g, size_t rows,
                   const GTPtr& sparse);  // wCol view with compact backward

// helpers used by the trainer
BufRef gScratchAlloc(uint64_t bytes);
void gFillZero(const BufRef& b);
void gUploadNow(const BufRef& dst, const void* src, uint64_t bytes);
void gDownloadNow(const BufRef& src, void* dst, uint64_t bytes);
void gDumpCopyTags(const char* outPath);  // copy_nd 来源直方图（诊断用）

}  // namespace gpu
}  // namespace ofc
