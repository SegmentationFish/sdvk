// Vulkan compute engine for ofc training: device, arenas, kernel dispatch.
#pragma once

#include <volk.h>

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ofc {
namespace gpu {

// a view into a VkBuffer (params may live in one big persistent buffer,
// activations in per-step arenas)
struct BufRef {
    VkBuffer buf = VK_NULL_HANDLE;
    uint64_t offset = 0;
    uint64_t size = 0;
};

constexpr uint64_t kAlign = 256;

class Arena {
public:
    void init(VkDevice dev, VkPhysicalDevice phys, uint64_t bytes, bool deviceLocal, bool hostVisible);
    BufRef alloc(uint64_t size);
    void reset();
    void rewind(uint64_t used) { used_ = used; }
    void destroy();
    uint64_t used() const { return used_; }
    uint64_t cap() const { return cap_; }
    VkBuffer buffer() const { return buffer_; }
    void* mapPtr() const { return mapped_; }
    VkDeviceMemory memory() const { return mem_; }

private:
    VkDevice dev_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory mem_ = VK_NULL_HANDLE;
    uint64_t cap_ = 0, used_ = 0;
    void* mapped_ = nullptr;
};

struct Kernel {
    VkPipeline pipe = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    uint32_t nBindings = 0;
    uint32_t pushSize = 0;
    uint32_t writeMask = 0;  // bit i = binding i is written by the shader
};

// a (buffer, range) touched since the last barrier, for dependency-aware barriers
struct TouchRegion {
    VkBuffer buf;
    uint64_t off, end;
    bool written;
};

class Engine {
public:
    void init();
    void destroy();

    // loads and compiles all kernels listed in KERNELS (see .cpp) from dir/name.spv
    void loadKernels(const std::string& spvDir);
    void initArenas(uint64_t stepBytes, uint64_t scratchBytes);

    // ---- per-step command recording ----
    void beginStep();  // resets arenas + begins a new command recording
    void beginPass();  // begins a new command recording WITHOUT arena reset
    void dispatch(const char* kernel, const std::vector<BufRef>& bufs, const void* push, uint32_t gx, uint32_t gy = 1,
                  uint32_t gz = 1);
    void dispatchIndirect(const char* kernel, const std::vector<BufRef>& bufs, const void* push, BufRef argsBuf);
    void fill(BufRef dst, uint32_t value);  // vkCmdFillBuffer, size%4==0
    void copyBuf(BufRef dst, BufRef src);   // buffer copy
    void endStep();
    void waitStep();

    // ---- step replay: record once, resubmit every step (command graph shapes/scalars must be static) ----
    // requirement: the dispatch sequence and all push constants must be byte-identical every step (varying scalars go through device buffers);
    // call invalidateReplay() to force re-recording after interruptions such as NaN recovery.
    void enableStepReplay(bool on) { replayEnabled_ = on; }
    void invalidateReplay();
    bool replayActive() const { return replayMode_; }
    uint64_t lastCmdCount() const { return lastCmds_; }  // command count inside the most recently recorded graph
    const std::unordered_map<std::string, uint64_t>& cmdHist() const { return hist_; }

    // GPU phase timing (OFC_TIMESTAMPS=1): mark() inside the step, readable after waitStep
    void mark(const char* label);
    bool phaseTimes(std::vector<std::pair<std::string, double>>& out) const;
    void profileDump();  // per-kernel GPU ms table (OFC_KERNEL_PROFILE=1)

    // upload/download helpers (upload: recorded immediately in own submission,
    // use only outside a step; download: own submission + wait)
    void uploadNow(BufRef dst, const void* src, uint64_t bytes);
    // batched upload: accumulate into the persistent staging ring, one submit for all (avoids a full sync per tensor)
    void stageUpload(BufRef dst, const void* src, uint64_t bytes);
    void flushUploads();
    void downloadNow(BufRef src, void* dst, uint64_t bytes);

    // staging ring used for in-step uploads (host writes before beginStep
    // AFTER the previous waitStep) and readbacks (host reads after waitStep)
    BufRef stagingUpload(uint64_t bytes);  // host-visible coherent
    BufRef stagingReadback(uint64_t bytes);

    void barrier();  // full memory barrier between dispatches

    Arena& stepArena() { return stepArena_; }
    Arena& scratchArena() { return scratchArena_; }
    Arena& stagingArena() { return stagingArena_; }
    VkDevice device() const { return dev_; }
    VkPhysicalDevice physDevice() const { return phys_; }
    const std::string& deviceName() const { return deviceName_; }
    uint64_t lastStepMs() const { return lastStepMs_; }

private:
    Kernel& kernel(const char* name);
    void oneShot(const std::function<void(VkCommandBuffer)>& fn);
    void ensureDlStaging(uint64_t bytes);

    VkBuffer dlBuf_ = VK_NULL_HANDLE;
    VkBuffer upBuf_ = VK_NULL_HANDLE;
    VkDeviceMemory upMem_ = VK_NULL_HANDLE;
    void* upMap_ = nullptr;
    uint64_t upCap_ = 0, upUsed_ = 0;
    std::vector<BufRef> upDst_;
    std::vector<uint64_t> upSrcOff_, upSize_;
    VkDeviceMemory dlMem_ = VK_NULL_HANDLE;
    void* dlMap_ = nullptr;
    uint64_t dlCap_ = 0;

    // dependency-aware barrier state
    std::vector<TouchRegion> pending_;
    VkPipelineStageFlags srcStages_ = 0;
    bool safeBarriers_ = false;  // OFC_SAFE_BARRIERS=1 restores barrier-everything
    bool hazard(const std::vector<BufRef>& bufs, uint32_t writeMask);
    void barrierFor(const std::vector<BufRef>& bufs, uint32_t writeMask, const char* next);
    void track(const std::vector<BufRef>& bufs, uint32_t writeMask);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice dev_ = VK_NULL_HANDLE;
    uint32_t qfam_ = 0;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout emptyLayout_ = VK_NULL_HANDLE;  // unused placeholder
    bool pushDesc_ = false;

    VkCommandBuffer stepCmd_ = VK_NULL_HANDLE;
    VkCommandBuffer cachedCmd_ = VK_NULL_HANDLE;  // replay cache (command graph recorded once)
    bool replayEnabled_ = false;
    bool replayMode_ = false;
    VkFence stepFence_ = VK_NULL_HANDLE;
    bool inStep_ = false;
    uint64_t lastStepMs_ = 0;
    uint64_t recCmds_ = 0, lastCmds_ = 0;
    std::unordered_map<std::string, uint64_t> hist_;  // per-kernel recorded-command histogram
    std::string lastKernel_;
    VkQueryPool tsPool_ = VK_NULL_HANDLE;
    float tsPeriod_ = 1.0f;
    bool tsEnabled_ = false;
    std::vector<std::string> tsLabels_;
    std::vector<uint64_t> tsRaw_;

    // per-kernel GPU profiler (OFC_KERNEL_PROFILE=1): 2 query slots per dispatch
    static constexpr uint32_t KP_CAP = 65536;
    bool kpEnabled_ = false;
    uint32_t kpCount_ = 0;
    std::vector<const char*> kpNames_;
    std::vector<uint64_t> kpRaw_;
    std::unordered_map<std::string, std::pair<double, uint32_t>> kpAgg_;  // name -> (total ns, calls)
    void kpCollect_();

    std::unordered_map<std::string, Kernel> kernels_;  // keyed by shader name

    Arena stepArena_;
    Arena scratchArena_;  // persistent scratch (indirect args, counts, etc.)
    Arena stagingArena_;  // host-visible upload/readback ring
    uint64_t stagingUploadOff_ = 0;
    uint64_t stagingReadbackOff_ = 0;

    std::string deviceName_;
};

}  // namespace gpu
}  // namespace ofc
