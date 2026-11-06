// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
#include "vk_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace ofc {
namespace gpu {

static const char* vkResName(VkResult r) {
    switch (r) {
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "MEMORY_MAP_FAILED";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "FEATURE_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "EXTENSION_NOT_PRESENT";
        case VK_ERROR_TOO_MANY_OBJECTS: return "TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "FORMAT_NOT_SUPPORTED";
        default: return "UNKNOWN";
    }
}

#define VK_CHECK(x)                                                                      \
    do {                                                                                 \
        VkResult _r = (x);                                                               \
        if (_r != VK_SUCCESS) {                                                          \
            std::fprintf(stderr, "VK error %d at %s:%d\n", (int)_r, __FILE__, __LINE__); \
            throw std::runtime_error(std::string("vulkan: ") + vkResName(_r) + " at " +  \
                                     __FILE__ + ":" + std::to_string(__LINE__));         \
        }                                                                                \
    } while (0)

// kernel table: name, storage-buffer binding count, push-constant bytes
// writeMask: bit i = binding i is written by the shader (dependency-aware barriers)
struct KernelDef {
    const char* name;
    uint32_t nBindings;
    uint32_t pushSize;
    uint32_t writeMask;
};
static const KernelDef KERNELS[] = {
    {"gemm_nn", 3, 48, 0x4},
    {"gemm_nt", 3, 48, 0x4},
    {"gemm_tn", 3, 48, 0x4},
    {"gemm_nt_q8", 4, 48, 0x4},
    {"gemm_nt_q8_v", 4, 48, 0x4},
    {"gemm_nn_q8_v", 4, 48, 0x4},
    {"gemm_nt_q8_v_bm128", 4, 48, 0x4},
    {"gemm_nt_q8_v_b", 5, 48, 0x4},
    {"gemm_nt_q8_v_bm128_b", 5, 48, 0x4},
    {"gemm_nt_q8_v_bk32", 4, 48, 0x4},
    {"gemm_nt_q8_v_bm128_bk32", 4, 48, 0x4},
    {"gemm_nn_q8", 4, 48, 0x4},
    {"copy_nd", 2, 60, 0x1},
    {"ew2", 3, 56, 0x1},
    {"mul_grad", 3, 52, 0x1},
    {"rowdot", 3, 8, 0x4},
    {"colsum_mul_add", 3, 8, 0x1},
    {"bcast_scalar", 2, 8, 0x2},
    {"unary_fwd", 2, 8, 0x2},
    {"unary_bwd", 3, 8, 0x4},
    {"softmax_fwd", 2, 8, 0x2},
    {"softmax_bwd", 3, 8, 0x4},
    {"rmsnorm_fwd", 3, 16, 0x4},
    {"rmsnorm_bwd", 5, 16, 0x18},
    {"rowsum_fwd", 2, 8, 0x2},
    {"colsum_add", 2, 8, 0x2},
    {"reduce_all", 2, 8, 0x2},
    {"crossentropy_fwd", 4, 8, 0x4},
    {"crossentropy_bwd", 5, 16, 0x8},
    {"gather_rows", 3, 8, 0x4},
    {"scatter_add_rows", 3, 8, 0x1},
    {"repeat_batch_bwd", 2, 12, 0x2},
    {"rope_fwd", 3, 24, 0x4},
    {"rope_bwd", 3, 24, 0x4},
    {"mask_fill_fwd", 3, 8, 0x4},
    {"mask_fill_bwd", 3, 8, 0x4},
    {"topk_sparse_fwd", 5, 12, 0x1E},
    {"topk_sparse_bwd", 5, 12, 0x10},
    {"compact", 5, 12, 0x1C},
    {"compact_bwd", 4, 12, 0x8},
    {"onehot_balance_fwd", 3, 8, 0x4},
    {"onehot_balance_bwd", 3, 8, 0x4},
    {"adamw", 6, 4, 0x0D},
    {"adaf_stats", 4, 12, 0x6},
    {"adafactor", 7, 8, 0x01},
    {"transpose", 2, 8, 0x1},
    {"mean_seq", 2, 12, 0x2},
    {"norm_sq", 2, 4, 0x2},
    {"clip_finalize", 2, 8, 0x2},
    {"opt_norm", 3, 0, 0x2},
    {"opt_stats", 6, 8, 0x2},
    {"opt_meta", 2, 4, 0x1},
    {"opt_update", 6, 8, 0x05},
    {"bias_gelu_fwd", 3, 8, 0x5},
    {"gelu_bias_bwd", 3, 8, 0x5},
    {"moe_merge_fwd", 4, 12, 0x8},
    {"moe_merge_dy", 3, 12, 0x4},
    {"moe_merge_wg_grad", 5, 12, 0x18},
    {"reduce_g", 2, 8, 0x2},
    {"copy2", 3, 8, 0x3},
    {"mul2", 5, 4, 0x3},
    {"attn_softmax_fwd", 4, 20, 0x8},
    {"attn_softmax_bwd", 5, 20, 0x18},
    {"im2col", 2, 20, 0x2},
    {"im2col_s2", 2, 24, 0x2},  // shader added int off at +20
    {"layernorm_fwd", 4, 16, 0x8},
    {"causal_mask", 1, 12, 0x1},
    {"groupnorm", 4, 20, 0x8},
    {"upsample2x", 2, 12, 0x2},
    {"bias_add", 2, 8, 0x1},
    {"add2", 3, 4, 0x4},
    {"fused_attn", 4, 36, 0x8},
    {"fused_attn_tiled", 4, 36, 0x8},
};
static const size_t NUM_KERNELS = sizeof(KERNELS) / sizeof(KERNELS[0]);

static std::vector<char> readSpv(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot open shader %s\n", path.c_str());
        std::exit(1);
    }
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void Arena::init(VkDevice dev, VkPhysicalDevice phys, uint64_t bytes, bool deviceLocal, bool hostVisible) {
    if (dev_ != VK_NULL_HANDLE) destroy();
    dev_ = dev;
    cap_ = bytes;
    used_ = 0;
    mapped_ = nullptr;
    // A failed vkAllocateMemory throws after freeing buffer_, so mem_ must not retain the
    // previous arena's handle: the next destroy() would vkFreeMemory it a second time.
    buffer_ = VK_NULL_HANDLE;
    mem_ = VK_NULL_HANDLE;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VK_CHECK(vkCreateBuffer(dev, &bi, nullptr, &buffer_));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, buffer_, &req);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    uint64_t localMb = 0;
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++)
        if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            localMb = std::max(localMb, (uint64_t)mp.memoryHeaps[i].size >> 20);
    VkMemoryPropertyFlags want = deviceLocal ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT : 0;
    if (hostVisible) want |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t idx = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if (!(req.memoryTypeBits & (1u << i))) continue;
        if ((mp.memoryTypes[i].propertyFlags & want) == want) {
            idx = i;
            break;
        }
    }
    if (idx == UINT32_MAX) {
        std::fprintf(stderr, "arena: no memory type\n");
        std::exit(1);
    }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = idx;
    VkResult ar = vkAllocateMemory(dev, &ai, nullptr, &mem_);
    if (ar != VK_SUCCESS) {
        // VK_ERROR_OUT_OF_HOST_MEMORY here is really "the OS refused the backing store". The
        // driver mirrors every device-local buffer in system RAM, so on a small-RAM box this is
        // a system-memory failure, not a VRAM one; say which so the user knows what to shrink.
        char msg[640];
        if (hostVisible)
            std::snprintf(msg, sizeof msg,
                          "vulkan: cannot allocate %.1f MB host-visible memory (%s); the OS is out of "
                          "committed RAM. free system memory or lower the arena sizes",
                          req.size / 1048576.0, vkResName(ar));
        else
            std::snprintf(msg, sizeof msg,
                          "vulkan: cannot allocate %.1f MB device-local memory (%s). the driver also "
                          "mirrors this in system RAM, so this can be either VRAM pressure or a "
                          "system-RAM shortage; device has %.0f MB VRAM. lower --param-mb/--step-mb/"
                          "--scratch-mb or the canvas size",
                          req.size / 1048576.0, vkResName(ar), localMb + 0.0);
        vkDestroyBuffer(dev, buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
        mem_ = VK_NULL_HANDLE;
        throw OomError(msg);
    }
    VK_CHECK(vkBindBufferMemory(dev, buffer_, mem_, 0));
    if (hostVisible) VK_CHECK(vkMapMemory(dev, mem_, 0, bytes, 0, &mapped_));
    // The driver rounds allocations up, so the usable span is req.size, not the requested bytes.
    // alloc() must fail at the real limit instead of walking past it.
    cap_ = req.size;
    deviceLocal_ = deviceLocal;
    hostVisible_ = hostVisible;
    phys_ = phys;
}

void Arena::grow(uint64_t need) {
    // Growing means a fresh object, so anything already written would be lost. Only allow it at
    // the bottom of the pool; past that the run genuinely needs a bigger --step-mb and should say
    // so rather than silently corrupting activations.
    if (used_ != 0) {
        char msg[256];
        std::snprintf(msg, sizeof msg,
                      "arena out of memory: need %.2f MB at offset %.2f MB, cannot grow a pool that already "
                      "holds live data (used %.2f of %.2f MB)",
                      need / 1e6, used_ / 1e6, used_ / 1e6, cap_ / 1e6);
        throw OomError(msg);
    }
    uint64_t want = cap_ + (cap_ >> 1);
    if (want < need) want = need;
    want = (want + (1ull << 20) - 1) & ~((1ull << 20) - 1);
    VkBuffer oldBuf = buffer_;
    VkDeviceMemory oldMem = mem_;
    void* oldMap = mapped_;
    cap_ = 0;
    buffer_ = VK_NULL_HANDLE;
    mem_ = VK_NULL_HANDLE;
    mapped_ = nullptr;
    try {
        init(dev_, phys_, want, deviceLocal_, hostVisible_);
    } catch (...) {
        buffer_ = oldBuf;
        mem_ = oldMem;
        mapped_ = oldMap;
        throw;
    }
    vkDestroyBuffer(dev_, oldBuf, nullptr);
    vkFreeMemory(dev_, oldMem, nullptr);
    std::fprintf(stderr, "[vk] arena grown to %.0f MB\n", cap_ / 1048576.0);
}

BufRef Arena::allocGrow(uint64_t size) {
    growable_ = true;
    uint64_t off = (used_ + kAlign - 1) / kAlign * kAlign;
    if (off + size > cap_) grow(off + size);
    return alloc(size);
}

BufRef Arena::alloc(uint64_t size) {
    uint64_t off = (used_ + kAlign - 1) / kAlign * kAlign;
    if (off + size > cap_) {
        char msg[256];
        std::snprintf(msg, sizeof msg, "arena out of memory: need %.2f MB more (used %.2f of %.2f MB)", size / 1e6,
                      used_ / 1e6, cap_ / 1e6);
        throw OomError(msg);
    }
    used_ = off + size;
    return BufRef{buffer_, off, size};
}

void Arena::reset() { used_ = 0; }

void Arena::destroy() {
    if (dev_ == VK_NULL_HANDLE) return;
    if (mapped_) vkUnmapMemory(dev_, mem_);
    vkDestroyBuffer(dev_, buffer_, nullptr);
    vkFreeMemory(dev_, mem_, nullptr);
    dev_ = VK_NULL_HANDLE;
}


std::vector<std::string> vkListDevices() {
    std::vector<std::string> out;
    volkInitialize();
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "ofc-vk-list";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) return out;
    volkLoadInstance(inst);
    uint32_t n = 0;
    if (vkEnumeratePhysicalDevices(inst, &n, nullptr) != VK_SUCCESS || n == 0) {
        vkDestroyInstance(inst, nullptr);
        return out;
    }
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(inst, &n, devs.data());
    for (uint32_t i = 0; i < n; i++) {
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qs(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, qs.data());
        bool compute = false;
        for (auto& q : qs) if (q.queueFlags & VK_QUEUE_COMPUTE_BIT) { compute = true; break; }
        if (!compute) continue;
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(devs[i], &mp);
        uint64_t best = 0;
        for (uint32_t h = 0; h < mp.memoryHeapCount; h++)
            if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                best = std::max(best, (uint64_t)mp.memoryHeaps[h].size);
        char buf[256];
        std::snprintf(buf, sizeof buf, "%u|%s|%llu", i, p.deviceName,
                      (unsigned long long)(best >> 20));
        out.push_back(buf);
    }
    vkDestroyInstance(inst, nullptr);
    return out;
}

void Engine::init() {
    VK_CHECK(volkInitialize());
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "ofc-vk-engine";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &instance_));
    volkLoadInstance(instance_);

    uint32_t n = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &n, nullptr));
    std::vector<VkPhysicalDevice> devs(n);
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &n, devs.data()));
    std::vector<std::pair<VkPhysicalDevice, uint32_t>> cand;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qs(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, qs.data());
        for (uint32_t q = 0; q < qn; q++)
            if (qs[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                cand.push_back({devs[i], q});
                break;
            }
    }
    if (cand.empty()) {
        std::fprintf(stderr, "no compute device\n");
        std::exit(1);
    }
    size_t pick = (size_t)devIndex_;
    if (pick >= cand.size()) {
        std::fprintf(stderr, "device index %d out of range (%zu compute devices), using 0\n", devIndex_, cand.size());
        pick = 0;
    }
    phys_ = cand[pick].first;
    qfam_ = cand[pick].second;
    if (!phys_) {
        std::fprintf(stderr, "no compute device\n");
        std::exit(1);
    }
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(phys_, &p);
    deviceName_ = p.deviceName;
    {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
        uint64_t best = 0, total = 0;
        for (uint32_t i = 0; i < mp.memoryHeapCount; i++) {
            if (!(mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) continue;
            total += mp.memoryHeaps[i].size;
            best = std::max(best, (uint64_t)mp.memoryHeaps[i].size);
        }
        deviceLocalMb_ = best >> 20;
        std::fprintf(stderr, "device: %s  device-local %.0f MB\n", deviceName_.c_str(), deviceLocalMb_ + 0.0);
    }

    // enable push descriptors if available
    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(phys_, nullptr, &en, nullptr);
    std::vector<VkExtensionProperties> exts(en);
    vkEnumerateDeviceExtensionProperties(phys_, nullptr, &en, exts.data());
    for (auto& e : exts)
        if (std::strcmp(e.extensionName, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME) == 0) pushDesc_ = true;
    if (!pushDesc_) {
        std::fprintf(stderr, "VK_KHR_push_descriptor not supported\n");
        std::exit(1);
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = qfam_;
    qi.queueCount = 1;
    qi.pQueuePriorities = &prio;
    const char* dexts[] = {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qi;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = dexts;
    VK_CHECK(vkCreateDevice(phys_, &dci, nullptr, &dev_));
    volkLoadDevice(dev_);
    vkGetDeviceQueue(dev_, qfam_, 0, &queue_);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.queueFamilyIndex = qfam_;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(dev_, &pci, nullptr, &pool_));
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(dev_, &fi, nullptr, &stepFence_));
    stagingArena_.init(dev_, phys_, 256ull << 20, false, true);
    upCap_ = upUsed_ = 0;
    safeBarriers_ = std::getenv("OFC_SAFE_BARRIERS") != nullptr;
    if (safeBarriers_) std::fprintf(stderr, "[vk] OFC_SAFE_BARRIERS=1: barrier after every op\n");
    tsEnabled_ = std::getenv("OFC_TIMESTAMPS") != nullptr;
    kpEnabled_ = std::getenv("OFC_KERNEL_PROFILE") != nullptr;
    if (tsEnabled_ || kpEnabled_) {
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qs(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, qs.data());
        if (qfam_ >= qn || qs[qfam_].timestampValidBits == 0) {
            tsEnabled_ = false;
            kpEnabled_ = false;
        } else {
            VkQueryPoolCreateInfo qp{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qp.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qp.queryCount = kpEnabled_ ? KP_CAP : 16;
            if (vkCreateQueryPool(dev_, &qp, nullptr, &tsPool_) != VK_SUCCESS) {
                tsEnabled_ = false;
                kpEnabled_ = false;
                tsPool_ = VK_NULL_HANDLE;
            } else {
                tsPeriod_ = p.limits.timestampPeriod;
                std::fprintf(stderr, "[vk] timestamps ON (period %.1f ns%s)\n", tsPeriod_,
                             kpEnabled_ ? " kernel-profile" : "");
            }
        }
    }
}

void Engine::destroy() {
    if (dev_ == VK_NULL_HANDLE) return;
    // Draining the device first lets the driver retire queued work in order. Tearing down while
    // commands are still in flight is what pushes WDDM into a display reset (black screen).
    vkDeviceWaitIdle(dev_);
    if (stepCmd_) vkFreeCommandBuffers(dev_, pool_, 1, &stepCmd_);
    if (cachedCmd_) vkFreeCommandBuffers(dev_, pool_, 1, &cachedCmd_);
    stepCmd_ = cachedCmd_ = VK_NULL_HANDLE;
    if (stepFence_) vkDestroyFence(dev_, stepFence_, nullptr);
    stepFence_ = VK_NULL_HANDLE;
    for (auto& kv : kernels_) {
        vkDestroyPipeline(dev_, kv.second.pipe, nullptr);
        vkDestroyPipelineLayout(dev_, kv.second.layout, nullptr);
    }
    kernels_.clear();
    if (tsPool_) vkDestroyQueryPool(dev_, tsPool_, nullptr);
    tsPool_ = VK_NULL_HANDLE;
    stepArena_.destroy();
    scratchArena_.destroy();
    stagingArena_.destroy();
    if (dlBuf_) vkDestroyBuffer(dev_, dlBuf_, nullptr);
    if (oneBuf_) {
        vkDestroyBuffer(dev_, oneBuf_, nullptr);
        vkFreeMemory(dev_, oneMem_, nullptr);
    }
    if (dlMem_) vkFreeMemory(dev_, dlMem_, nullptr);
    dlBuf_ = VK_NULL_HANDLE;
    dlMem_ = VK_NULL_HANDLE;
    dlMap_ = nullptr;
    dlCap_ = 0;
    oneBuf_ = VK_NULL_HANDLE;
    oneMem_ = VK_NULL_HANDLE;
    oneMap_ = nullptr;
    oneCap_ = 0;
    if (upBuf_) vkDestroyBuffer(dev_, upBuf_, nullptr);
    if (upMem_) vkFreeMemory(dev_, upMem_, nullptr);
    upBuf_ = VK_NULL_HANDLE;
    upMem_ = VK_NULL_HANDLE;
    upMap_ = nullptr;
    upCap_ = upUsed_ = 0;
    upDst_.clear();
    upSrcOff_.clear();
    upSize_.clear();
    pending_.clear();
    vkDestroyCommandPool(dev_, pool_, nullptr);
    pool_ = VK_NULL_HANDLE;
    vkDestroyDevice(dev_, nullptr);
    dev_ = VK_NULL_HANDLE;
    vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
    phys_ = VK_NULL_HANDLE;
    queue_ = VK_NULL_HANDLE;
    inStep_ = replayMode_ = false;
}

Engine::~Engine() { destroy(); }

void Engine::initArenas(uint64_t stepBytes, uint64_t scratchBytes) {
    // A driver can round an allocation up (AMD pads each heap piece by a few percent), and the
    // padded span still has to fit. Asking exactly what is left next to a loaded param arena
    // therefore fails even though the numbers add up, so step down on failure instead of dying.
    for (;;) {
        try {
            stepArena_.init(dev_, phys_, stepBytes, true, false);
            break;
        } catch (const OomError& e) {
            uint64_t next = stepBytes > (64ull << 20) ? (stepBytes * 7 / 8) & ~((1ull << 20) - 1) : 0;
            if (!next) throw;
            std::fprintf(stderr, "[vk] step arena %.0f MB refused, retrying at %.0f MB\n", stepBytes / 1048576.0,
                         next / 1048576.0);
            (void)e;
            stepBytes = next;
        }
    }
    for (;;) {
        try {
            scratchArena_.init(dev_, phys_, scratchBytes, true, false);
            break;
        } catch (const OomError&) {
            uint64_t next = scratchBytes > (64ull << 20) ? (scratchBytes * 7 / 8) & ~((1ull << 20) - 1) : 0;
            if (!next) throw;
            std::fprintf(stderr, "[vk] scratch arena %.0f MB refused, retrying at %.0f MB\n",
                         scratchBytes / 1048576.0, next / 1048576.0);
            scratchBytes = next;
        }
    }
    stepArena_.setQueue(queue_);
    stepArena_.setCmdPool(pool_);
    scratchArena_.setQueue(queue_);
    scratchArena_.setCmdPool(pool_);
}

void Arena::moveDown(uint64_t dst, uint64_t off, uint64_t bytes) {
    if (bytes == 0 || dst == off) return;
    if (dst > off) throw std::runtime_error("arena moveDown only moves toward the base");
    if (!queue_ || !cmdPool_ || !buffer_) throw std::runtime_error("arena moveDown without a queue");
    // The ranges overlap when the tensor is wider than the gap it moves into, and a plain copy
    // would read what a previous iteration just wrote. Bounce through the top of the pool, which
    // is free at this point because the caller just rewound past the source.
    VkDeviceSize bounce = cap_ - ((bytes + 255) & ~255ull);
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool = cmdPool_;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(dev_, &cbi, &cb));
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    if (off < dst + bytes) {
        VkBufferCopy c1{off, bounce, bytes};
        vkCmdCopyBuffer(cb, buffer_, buffer_, 1, &c1);
        // Transfer commands do not make their writes visible to the following transfer by
        // themselves; without this barrier the copy back reads stale bounce bytes.
        VkMemoryBarrier mbc{};
        mbc.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mbc.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mbc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mbc, 0,
                             nullptr, 0, nullptr);
        VkBufferCopy c2{bounce, dst, bytes};
        vkCmdCopyBuffer(cb, buffer_, buffer_, 1, &c2);
    } else {
        VkBufferCopy c{off, dst, bytes};
        vkCmdCopyBuffer(cb, buffer_, buffer_, 1, &c);
    }
    VkMemoryBarrier mb2{};
    mb2.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb2, 0,
                         nullptr, 0, nullptr);
    VK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue_));
    vkFreeCommandBuffers(dev_, cmdPool_, 1, &cb);
}

void Arena::moveDownInCmd(VkCommandBuffer cb, uint64_t dst, uint64_t off, uint64_t bytes) {
    if (bytes == 0 || dst == off) return;
    if (dst > off) throw std::runtime_error("arena moveDown only moves toward the base");
    // Bounce through the first free bytes above the live range: nothing has been handed out past
    // used_ yet, and every command recorded after this copy executes after it, so reusing that
    // scratch later in the stream is safe.
    VkDeviceSize bounce = (used_ + 255) & ~255ull;
    if (bounce + bytes > cap_) throw OomError("arena moveDown: no room for the bounce copy");
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    if (off < dst + bytes) {
        VkBufferCopy c1{off, bounce, bytes};
        vkCmdCopyBuffer(cb, buffer_, buffer_, 1, &c1);
        // Transfer commands do not make their writes visible to the following transfer by
        // themselves; without this barrier the copy back reads stale bounce bytes.
        VkMemoryBarrier mbc{};
        mbc.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mbc.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mbc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mbc, 0,
                             nullptr, 0, nullptr);
        VkBufferCopy c2{bounce, dst, bytes};
        vkCmdCopyBuffer(cb, buffer_, buffer_, 1, &c2);
    } else {
        VkBufferCopy c{off, dst, bytes};
        vkCmdCopyBuffer(cb, buffer_, buffer_, 1, &c);
    }
    VkMemoryBarrier mb2{};
    mb2.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb2, 0,
                         nullptr, 0, nullptr);
    used_ = dst + bytes;
}

void Engine::compactStepArena(uint64_t dst, uint64_t off, uint64_t bytes) {
    if (bytes == 0 || dst == off) return;
    if (inStep_ && stepCmd_ != VK_NULL_HANDLE) {
        stepArena_.moveDownInCmd(stepCmd_, dst, off, bytes);
    } else {
        stepArena_.moveDown(dst, off, bytes);
    }
}

void Engine::loadKernels(const std::string& spvDir) {
    for (size_t i = 0; i < NUM_KERNELS; i++) {
        const KernelDef& kd = KERNELS[i];
        auto code = readSpv(spvDir + "/" + kd.name + ".spv");
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = code.size();
        smi.pCode = (const uint32_t*)code.data();
        VkShaderModule mod = nullptr;
        VK_CHECK(vkCreateShaderModule(dev_, &smi, nullptr, &mod));

        std::vector<VkDescriptorSetLayoutBinding> binds(kd.nBindings);
        for (uint32_t b = 0; b < kd.nBindings; b++) {
            binds[b].binding = b;
            binds[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[b].descriptorCount = 1;
            binds[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dli.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        dli.bindingCount = kd.nBindings;
        dli.pBindings = binds.data();
        VkDescriptorSetLayout dsl = nullptr;
        VK_CHECK(vkCreateDescriptorSetLayout(dev_, &dli, nullptr, &dsl));

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size = kd.pushSize;
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &dsl;
        if (kd.pushSize) {
            pli.pushConstantRangeCount = 1;
            pli.pPushConstantRanges = &pcr;
        }
        VkPipelineLayout layout = nullptr;
        VK_CHECK(vkCreatePipelineLayout(dev_, &pli, nullptr, &layout));

        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = mod;
        cpi.stage.pName = "main";
        cpi.layout = layout;
        Kernel k;
        VK_CHECK(vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &cpi, nullptr, &k.pipe));
        k.layout = layout;
        k.nBindings = kd.nBindings;
        k.pushSize = kd.pushSize;
        k.writeMask = kd.writeMask;
        vkDestroyShaderModule(dev_, mod, nullptr);
        kernels_[kd.name] = k;
    }
}

Kernel& Engine::kernel(const char* name) {
    auto it = kernels_.find(name);
    if (it == kernels_.end()) {
        std::fprintf(stderr, "kernel not loaded: %s\n", name);
        std::exit(1);
    }
    return it->second;
}

void Engine::oneShot(const std::function<void(VkCommandBuffer)>& fn) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb = nullptr;
    VK_CHECK(vkAllocateCommandBuffers(dev_, &ai, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    fn(cb);
    VK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = nullptr;
    VK_CHECK(vkCreateFence(dev_, &fi, nullptr, &fence));
    VK_CHECK(vkQueueSubmit(queue_, 1, &si, fence));
    VK_CHECK(vkWaitForFences(dev_, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(dev_, fence, nullptr);
    vkFreeCommandBuffers(dev_, pool_, 1, &cb);
}

static VkBuffer createStaging(VkDevice dev, VkPhysicalDevice phys, uint64_t size, VkDeviceMemory* outMem) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer buf = nullptr;
    VK_CHECK(vkCreateBuffer(dev, &bi, nullptr, &buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, buf, &req);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    uint32_t idx = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if (!(req.memoryTypeBits & (1u << i))) continue;
        VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if ((mp.memoryTypes[i].propertyFlags & want) == want) {
            idx = i;
            break;
        }
    }
    if (idx == UINT32_MAX) {
        std::fprintf(stderr, "staging: no mem type\n");
        std::exit(1);
    }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = idx;
    VkDeviceMemory mem = nullptr;
    VK_CHECK(vkAllocateMemory(dev, &ai, nullptr, &mem));
    VK_CHECK(vkBindBufferMemory(dev, buf, mem, 0));
    *outMem = mem;
    return buf;  // caller owns both; free the buffer and the memory
}

void Engine::flushUploads() {
    if (upDst_.empty()) return;
    oneShot([&](VkCommandBuffer cb) {
        for (size_t i = 0; i < upDst_.size(); i++) {
            VkBufferCopy cp{upSrcOff_[i], upDst_[i].offset, upSize_[i]};
            vkCmdCopyBuffer(cb, upBuf_, upDst_[i].buf, 1, &cp);
        }
    });
    upDst_.clear();
    upSrcOff_.clear();
    upSize_.clear();
    upUsed_ = 0;
}

void Engine::releaseUploadStaging() {
    if (!upBuf_) return;
    flushUploads();
    if (upMap_) vkUnmapMemory(dev_, upMem_);
    vkDestroyBuffer(dev_, upBuf_, nullptr);
    vkFreeMemory(dev_, upMem_, nullptr);
    upBuf_ = VK_NULL_HANDLE;
    upMem_ = VK_NULL_HANDLE;
    upMap_ = nullptr;
    upCap_ = upUsed_ = 0;
}

void* Engine::beginUpload(BufRef dst, uint64_t bytes) {
    if (bytes == 0) return nullptr;
    if (upUsed_ + bytes > upCap_) {
        flushUploads();
        if (bytes > upCap_) {
            if (upBuf_) {
                if (upMap_) vkUnmapMemory(dev_, upMem_);
                vkDestroyBuffer(dev_, upBuf_, nullptr);
                vkFreeMemory(dev_, upMem_, nullptr);
                upBuf_ = VK_NULL_HANDLE;
                upMem_ = VK_NULL_HANDLE;
                upMap_ = nullptr;
            }
            uint64_t want = (bytes + (16ull << 20) - 1) & ~((16ull << 20) - 1);
            VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bi.size = want;
            bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            VK_CHECK(vkCreateBuffer(dev_, &bi, nullptr, &upBuf_));
            VkMemoryRequirements req;
            vkGetBufferMemoryRequirements(dev_, upBuf_, &req);
            VkPhysicalDeviceMemoryProperties mp;
            vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
            uint32_t idx = UINT32_MAX;
            for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
                if (!(req.memoryTypeBits & (1u << i))) continue;
                VkMemoryPropertyFlags want2 =
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                if ((mp.memoryTypes[i].propertyFlags & want2) == want2) {
                    idx = i;
                    break;
                }
            }
            if (idx == UINT32_MAX) {
                std::fprintf(stderr, "upload staging: no mem type%c", 10);
                std::exit(1);
            }
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.allocationSize = req.size;
            ai.memoryTypeIndex = idx;
            VK_CHECK(vkAllocateMemory(dev_, &ai, nullptr, &upMem_));
            VK_CHECK(vkBindBufferMemory(dev_, upBuf_, upMem_, 0));
            VK_CHECK(vkMapMemory(dev_, upMem_, 0, want, 0, &upMap_));
            upCap_ = want;
        }
    }
    void* slot = (char*)upMap_ + upUsed_;
    upDst_.push_back(dst);
    upSrcOff_.push_back(upUsed_);
    upSize_.push_back(bytes);
    upUsed_ += (bytes + 255) & ~255ull;
    return slot;
}

void Engine::endUpload() {}

void Engine::stageUpload(BufRef dst, const void* src, uint64_t bytes) {
    void* slot = beginUpload(dst, bytes);
    if (!slot) return;
    std::memcpy(slot, src, (size_t)bytes);
}

void Engine::uploadNow(BufRef dst, const void* src, uint64_t bytes) {
    if (inStep_) {
        // in-step upload: written into host-visible staging (deterministic offsets, same allocation order after the per-step reset);
        // the copy command goes into the command graph (re-executed from cache on replay; the host only refreshes staging contents each step)
        BufRef slot = stagingArena_.alloc(bytes);
        std::memcpy((char*)stagingArena_.mapPtr() + slot.offset, src, (size_t)bytes);
        copyBuf(dst, slot);
        return;
    }
    if (bytes > oneCap_) {
        if (oneBuf_) {
            vkDestroyBuffer(dev_, oneBuf_, nullptr);
            vkFreeMemory(dev_, oneMem_, nullptr);
            oneBuf_ = VK_NULL_HANDLE;
            oneMem_ = VK_NULL_HANDLE;
            oneMap_ = nullptr;
        }
        oneCap_ = std::max<uint64_t>(bytes, 16ull << 20);
        oneBuf_ = createStaging(dev_, phys_, oneCap_, &oneMem_);
        VK_CHECK(vkMapMemory(dev_, oneMem_, 0, VK_WHOLE_SIZE, 0, &oneMap_));
    }
    std::memcpy(oneMap_, src, (size_t)bytes);
    VkBufferCopy cp{0, dst.offset, bytes};
    oneShot([&](VkCommandBuffer cb) { vkCmdCopyBuffer(cb, oneBuf_, dst.buf, 1, &cp); });
}

void Engine::ensureDlStaging(uint64_t bytes) {
    if (bytes <= dlCap_ && dlBuf_) return;
    if (dlBuf_) {
        if (dlMap_) vkUnmapMemory(dev_, dlMem_);
        vkDestroyBuffer(dev_, dlBuf_, nullptr);
        vkFreeMemory(dev_, dlMem_, nullptr);
        dlBuf_ = VK_NULL_HANDLE;
        dlMem_ = VK_NULL_HANDLE;
        dlMap_ = nullptr;
    }
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VK_CHECK(vkCreateBuffer(dev_, &bi, nullptr, &dlBuf_));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev_, dlBuf_, &req);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
    uint32_t idx = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if (!(req.memoryTypeBits & (1u << i))) continue;
        VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if ((mp.memoryTypes[i].propertyFlags & want) == want) {
            idx = i;
            break;
        }
    }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = idx;
    VK_CHECK(vkAllocateMemory(dev_, &ai, nullptr, &dlMem_));
    VK_CHECK(vkBindBufferMemory(dev_, dlBuf_, dlMem_, 0));
    VK_CHECK(vkMapMemory(dev_, dlMem_, 0, bytes, 0, &dlMap_));
    dlCap_ = bytes;
}

void Engine::downloadNow(BufRef src, void* dst, uint64_t bytes) {
    ensureDlStaging(bytes);
    VkBufferCopy cp{src.offset, 0, bytes};
    oneShot([&](VkCommandBuffer cb) { vkCmdCopyBuffer(cb, src.buf, dlBuf_, 1, &cp); });
    std::memcpy(dst, dlMap_, (size_t)bytes);
}

void Engine::downloadNow(BufRef src, void* dst, uint64_t bytes, uint64_t srcOff) {
    ensureDlStaging(bytes);
    VkBufferCopy cp{src.offset + srcOff, 0, bytes};
    oneShot([&](VkCommandBuffer cb) { vkCmdCopyBuffer(cb, src.buf, dlBuf_, 1, &cp); });
    std::memcpy(dst, dlMap_, (size_t)bytes);
}

BufRef Engine::stagingUpload(uint64_t bytes) {
    BufRef r = stagingArena_.alloc(bytes);  // host-visible arena
    return r;
}

BufRef Engine::stagingReadback(uint64_t bytes) {
    BufRef r = stagingArena_.alloc(bytes);
    return r;
}

void Engine::beginStep() {
    tsLabels_.clear();
    stagingArena_.reset();
    stepArena_.reset();
    if (replayEnabled_ && cachedCmd_) {
        replayMode_ = true;  // cached command graph: this step only replays host-side (allocations keep deterministic offsets)
        inStep_ = true;
        static bool announced = false;
        if (!announced) {
            std::fprintf(stderr, "[replay] cached command graph in use\n");
            announced = true;
        }
        return;
    }
    replayMode_ = false;
    kpCount_ = 0;
    kpNames_.clear();
    beginPass();
    if (tsEnabled_) vkCmdResetQueryPool(stepCmd_, tsPool_, 0, 16);
    if (kpEnabled_) vkCmdResetQueryPool(stepCmd_, tsPool_, 0, KP_CAP);
}

void Engine::mark(const char* label) {
    if (!tsEnabled_) return;
    uint32_t idx = (uint32_t)tsLabels_.size();
    if (idx >= 16) return;
    tsLabels_.push_back(label);
    if (replayMode_) return;
    vkCmdWriteTimestamp(stepCmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, tsPool_, idx);
}

bool Engine::phaseTimes(std::vector<std::pair<std::string, double>>& out) const {
    out.clear();
    if (!tsEnabled_ || tsRaw_.size() < 2 || tsLabels_.size() != tsRaw_.size()) return false;
    for (size_t i = 1; i < tsRaw_.size(); i++) {
        double ms = (double)(tsRaw_[i] - tsRaw_[i - 1]) * (double)tsPeriod_ / 1e6;
        out.push_back({tsLabels_[i], ms});
    }
    return true;
}

void Engine::kpCollect_() {
    if (!kpEnabled_ || !kpCount_ || kpNames_.size() * 2 != kpCount_) return;
    kpRaw_.resize(kpCount_);
    if (vkGetQueryPoolResults(dev_, tsPool_, 0, kpCount_, kpCount_ * 8, kpRaw_.data(), 8, VK_QUERY_RESULT_64_BIT) !=
        VK_SUCCESS)
        return;
    for (uint32_t i = 0; i < kpCount_; i += 2) {
        double ns = (double)(kpRaw_[i + 1] - kpRaw_[i]) * (double)tsPeriod_;
        auto& a = kpAgg_[kpNames_[i / 2]];
        a.first += ns;
        a.second++;
    }
}

void Engine::profileDump() {
    if (!kpEnabled_) { return; }
    if (kpAgg_.empty()) {
        std::fprintf(stderr, "[kp] no kernel data captured\n");
        return;
    }
    std::vector<std::pair<std::string, std::pair<double, uint32_t>>> v(kpAgg_.begin(), kpAgg_.end());
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
    double grand = 0;
    uint32_t calls = 0;
    for (auto& e : v) {
        grand += e.second.first;
        calls += e.second.second;
    }
    std::fprintf(stderr, "--- kernel GPU profile (%zu kernels, %u calls, %.1f ms total) ---\n", v.size(), calls,
                 grand / 1e6);
    for (auto& e : v) {
        double ms = e.second.first / 1e6;
        std::fprintf(stderr, "%-30s calls %6u  total %9.2f ms  avg %7.3f ms  %5.1f%%\n", e.first.c_str(),
                     e.second.second, ms, ms / e.second.second, 100.0 * e.second.first / grand);
    }
}

void Engine::beginPass() {
    pending_.clear();
    srcStages_ = 0;
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(dev_, &ai, &stepCmd_));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    // replay resubmits the same buffer; ONE_TIME_SUBMIT allows a single submit and must be dropped
    bi.flags = replayEnabled_ ? 0 : VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(stepCmd_, &bi));
    inStep_ = true;
    recCmds_ = 0;
}

bool Engine::hazard(const std::vector<BufRef>& bufs, uint32_t writeMask) {
    static const bool noBarriers = std::getenv("OFC_NO_BARRIERS") != nullptr;
    if (noBarriers) return false;  // timing experiments only (results are not trustworthy)
    if (safeBarriers_) return true;
    for (size_t i = 0; i < bufs.size(); i++) {
        bool w = ((writeMask >> i) & 1u) != 0;
        uint64_t off = bufs[i].offset, end = bufs[i].offset + bufs[i].size;
        for (auto& t : pending_) {
            if (t.buf != bufs[i].buf) continue;
            if (off >= t.end || t.off >= end) continue;  // disjoint ranges
            if (w || t.written) return true;             // RAW / WAW / WAR on same range
        }
    }
    return false;
}

void Engine::track(const std::vector<BufRef>& bufs, uint32_t writeMask) {
    for (size_t i = 0; i < bufs.size(); i++)
        pending_.push_back({bufs[i].buf, bufs[i].offset, bufs[i].offset + bufs[i].size, ((writeMask >> i) & 1u) != 0});
}

// targeted buffer barriers: emit VkBufferMemoryBarrier only for genuinely overlapping buffers (cheaper than a global VkMemoryBarrier)
void Engine::barrierFor(const std::vector<BufRef>& bufs, uint32_t writeMask, const char* next) {
    static const bool noBarriers = std::getenv("OFC_NO_BARRIERS") != nullptr;
    if (noBarriers) return;
    if (safeBarriers_) {
        barrier();
        return;
    }
    VkBuffer bl[8];
    uint32_t nb = 0;
    VkAccessFlags srcAcc[8] = {}, dstAcc[8] = {};
    uint64_t rStart[8] = {}, rEnd[8] = {};
    for (size_t i = 0; i < bufs.size(); i++) {
        bool w = ((writeMask >> i) & 1u) != 0;
        bool overlap = false, hasW = false, needR = false, needW = false;
        uint64_t lo = bufs[i].offset, hi = bufs[i].offset + bufs[i].size;
        for (auto& t : pending_) {
            if (t.buf != bufs[i].buf) continue;
            if (bufs[i].offset >= t.end || t.off >= bufs[i].offset + bufs[i].size) continue;
            overlap = true;
            if (t.off < lo) lo = t.off;
            if (t.end > hi) hi = t.end;
            if (t.written) hasW = true;
            if (w) needR = needW = true;       // write binding: read-modify-write
            else if (t.written) needR = true;  // RAW
            // pure WAR: execution dependency only, no memory visibility mask
        }
        if (!overlap) continue;
        bool dup = false;
        uint32_t k = 0;
        for (; k < nb; k++)
            if (bl[k] == bufs[i].buf) {
                dup = true;
                break;
            }
        if (!dup) {
            if (nb == 8) {
                barrier();
                return;
            }
            k = nb++;
            bl[k] = bufs[i].buf;
            rStart[k] = lo;
            rEnd[k] = hi;
        } else {
            if (lo < rStart[k]) rStart[k] = lo;
            if (hi > rEnd[k]) rEnd[k] = hi;
        }
        if (hasW) srcAcc[k] |= VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        if (needR) dstAcc[k] |= VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        if (needW) dstAcc[k] |= VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    }
    if (nb == 0) return;
    hist_["<barrier>"]++;
    hist_[std::string("B:") + lastKernel_ + "->" + next]++;
    static const bool noFlush = std::getenv("OFC_BARRIER_NOFLUSH") != nullptr;
    const VkPipelineStageFlags ct = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkPipelineStageFlags src = srcStages_ ? srcStages_ : ct;
    VkBufferMemoryBarrier bb[8];
    for (uint32_t k = 0; k < nb; k++) {
        bb[k] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        bb[k].srcAccessMask = noFlush ? 0 : srcAcc[k];
        bb[k].dstAccessMask = noFlush ? 0 : dstAcc[k];
        bb[k].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb[k].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb[k].buffer = bl[k];
        bb[k].offset = rStart[k] & ~3ull;
        bb[k].size = (rEnd[k] - bb[k].offset + 3ull) & ~3ull;
    }
    vkCmdPipelineBarrier(stepCmd_, src, ct, 0, 0, nullptr, nb, bb, 0, nullptr);
    // drop only the pending entries covered by this range; keep the rest
    std::vector<TouchRegion> keep;
    keep.reserve(pending_.size());
    for (auto& t : pending_) {
        bool covered = false;
        for (uint32_t k = 0; k < nb; k++)
            if (t.buf == bl[k] && t.off >= rStart[k] && t.end <= rEnd[k]) covered = true;
        if (!covered) keep.push_back(t);
    }
    pending_.swap(keep);
    srcStages_ |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
}

void Engine::barrier() {
    if (pending_.empty() && srcStages_ == 0) return;  // nothing outstanding to order
    hist_["<barrier>"]++;
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                       VK_ACCESS_TRANSFER_WRITE_BIT;
    // compute/transfer stages only: ALL_COMMANDS triggers a full cache flush on Polaris, which is very costly
    const VkPipelineStageFlags ct = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkPipelineStageFlags src = srcStages_ ? srcStages_ : ct;
    vkCmdPipelineBarrier(stepCmd_, src, ct, 0, 1, &mb, 0, nullptr, 0, nullptr);
    pending_.clear();
    srcStages_ = 0;
}

void Engine::dispatch(const char* kernelName, const std::vector<BufRef>& bufs, const void* push, uint32_t gx,
                      uint32_t gy, uint32_t gz) {
    if (replayMode_) return;  // cached: skip recording (hazards/barriers are baked in)
    Kernel& k = kernel(kernelName);
    if (bufs.size() != k.nBindings) {
        std::fprintf(stderr, "kernel %s expects %u bindings, got %zu\n", kernelName, k.nBindings, bufs.size());
        std::exit(1);
    }
    VkDescriptorBufferInfo infos[8];
    VkWriteDescriptorSet writes[8];
    for (uint32_t b = 0; b < k.nBindings; b++) {
        infos[b].buffer = bufs[b].buf;
        infos[b].offset = bufs[b].offset;
        infos[b].range = bufs[b].size;
        writes[b] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[b].dstBinding = b;
        writes[b].descriptorCount = 1;
        writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[b].pBufferInfo = &infos[b];
    }
    barrierFor(bufs, k.writeMask, kernelName);
    bool kp = kpEnabled_ && kpCount_ + 2 <= KP_CAP;
    if (kp) vkCmdWriteTimestamp(stepCmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, tsPool_, kpCount_);
    vkCmdBindPipeline(stepCmd_, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipe);
    vkCmdPushDescriptorSetKHR(stepCmd_, VK_PIPELINE_BIND_POINT_COMPUTE, k.layout, 0, k.nBindings, writes);
    if (k.pushSize) vkCmdPushConstants(stepCmd_, k.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, k.pushSize, push);
    vkCmdDispatch(stepCmd_, gx, gy, gz);
    if (kp) {
        vkCmdWriteTimestamp(stepCmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, tsPool_, kpCount_ + 1);
        kpNames_.push_back(kernelName);
        kpCount_ += 2;
    }
    lastKernel_ = kernelName;
    recCmds_++;
    hist_[kernelName]++;
    track(bufs, k.writeMask);
    srcStages_ |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
}

void Engine::dispatchIndirect(const char* kernelName, const std::vector<BufRef>& bufs, const void* push,
                              BufRef argsBuf) {
    if (replayMode_) return;
    Kernel& k = kernel(kernelName);
    VkDescriptorBufferInfo infos[8];
    VkWriteDescriptorSet writes[8];
    for (uint32_t b = 0; b < k.nBindings; b++) {
        infos[b].buffer = bufs[b].buf;
        infos[b].offset = bufs[b].offset;
        infos[b].range = bufs[b].size;
        writes[b] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[b].dstBinding = b;
        writes[b].descriptorCount = 1;
        writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[b].pBufferInfo = &infos[b];
    }
    barrierFor(bufs, k.writeMask, kernelName);
    bool kp = kpEnabled_ && kpCount_ + 2 <= KP_CAP;
    if (kp) vkCmdWriteTimestamp(stepCmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, tsPool_, kpCount_);
    vkCmdBindPipeline(stepCmd_, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipe);
    vkCmdPushDescriptorSetKHR(stepCmd_, VK_PIPELINE_BIND_POINT_COMPUTE, k.layout, 0, k.nBindings, writes);
    if (k.pushSize) vkCmdPushConstants(stepCmd_, k.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, k.pushSize, push);
    vkCmdDispatchIndirect(stepCmd_, argsBuf.buf, argsBuf.offset);
    if (kp) {
        vkCmdWriteTimestamp(stepCmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, tsPool_, kpCount_ + 1);
        kpNames_.push_back(kernelName);
        kpCount_ += 2;
    }
    lastKernel_ = kernelName;
    recCmds_++;
    hist_[kernelName]++;
    track(bufs, k.writeMask);
    srcStages_ |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
}

void Engine::fill(BufRef dst, uint32_t value) {
    if (replayMode_) return;
    std::vector<BufRef> b{dst};
    barrierFor(b, 0x1, "<fill>");
    vkCmdFillBuffer(stepCmd_, dst.buf, dst.offset, dst.size, value);
    lastKernel_ = "<fill>";
    recCmds_++;
    hist_["<fill>"]++;
    track(b, 0x1);
    srcStages_ |= VK_PIPELINE_STAGE_TRANSFER_BIT;
}

void Engine::copyBuf(BufRef dst, BufRef src) {
    if (replayMode_) return;
    std::vector<BufRef> b{dst, src};
    barrierFor(b, 0x1, "<copy>");
    VkBufferCopy cp{};
    cp.srcOffset = src.offset;
    cp.dstOffset = dst.offset;
    cp.size = src.size < dst.size ? src.size : dst.size;
    vkCmdCopyBuffer(stepCmd_, src.buf, dst.buf, 1, &cp);
    recCmds_++;
    hist_["<copy>"]++;
    lastKernel_ = "<copy>";
    track(b, 0x1);
    srcStages_ |= VK_PIPELINE_STAGE_TRANSFER_BIT;
}

void Engine::endStep() {
    if (replayMode_) return;  // replay: command graph already cached
    lastCmds_ = recCmds_;
    VK_CHECK(vkEndCommandBuffer(stepCmd_));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &stepCmd_;
    VK_CHECK(vkQueueSubmit(queue_, 1, &si, stepFence_));
    inStep_ = false;
    if (replayEnabled_ && !cachedCmd_) cachedCmd_ = stepCmd_;  // keep the first recorded graph for replay
}

void Engine::waitStep() {
    auto t0 = std::chrono::steady_clock::now();
    if (replayMode_) {
        // the fence was reset in the previous waitStep; submit the cached graph and wait
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cachedCmd_;
        VK_CHECK(vkQueueSubmit(queue_, 1, &si, stepFence_));
        VK_CHECK(vkWaitForFences(dev_, 1, &stepFence_, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(dev_, 1, &stepFence_));
        if (tsEnabled_ && !tsLabels_.empty()) {
            tsRaw_.resize(tsLabels_.size());
            vkGetQueryPoolResults(dev_, tsPool_, 0, (uint32_t)tsRaw_.size(), tsRaw_.size() * 8, tsRaw_.data(), 8,
                                  VK_QUERY_RESULT_64_BIT);
        }
        kpCollect_();
        replayMode_ = false;
        inStep_ = false;
        auto t1 = std::chrono::steady_clock::now();
        lastStepMs_ = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        return;
    }
    VK_CHECK(vkWaitForFences(dev_, 1, &stepFence_, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(dev_, 1, &stepFence_));
    if (tsEnabled_ && !tsLabels_.empty()) {
        tsRaw_.resize(tsLabels_.size());
        vkGetQueryPoolResults(dev_, tsPool_, 0, (uint32_t)tsRaw_.size(), tsRaw_.size() * 8, tsRaw_.data(), 8,
                              VK_QUERY_RESULT_64_BIT);
    }
    kpCollect_();
    if (!(replayEnabled_ && cachedCmd_ == stepCmd_)) {
        vkFreeCommandBuffers(dev_, pool_, 1, &stepCmd_);
        stepCmd_ = nullptr;
    } else {
        stepCmd_ = nullptr;  // cached, keep it
    }
    auto t1 = std::chrono::steady_clock::now();
    lastStepMs_ = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
}

void Engine::invalidateReplay() {
    if (cachedCmd_) {
        vkFreeCommandBuffers(dev_, pool_, 1, &cachedCmd_);
        cachedCmd_ = nullptr;
    }
    replayMode_ = false;
}

std::string Engine::deviceReport() const {
    VkPhysicalDeviceProperties p{};
    vkGetPhysicalDeviceProperties(phys_, &p);
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
    char line[512];
    std::string s;
    std::snprintf(line, sizeof line, "device: %s\n", p.deviceName);
    s += line;
    std::snprintf(line, sizeof line, "api: %u.%u.%u  driver: %u.%u.%u  vendor: 0x%04x device: 0x%04x\n",
                  VK_VERSION_MAJOR(p.apiVersion), VK_VERSION_MINOR(p.apiVersion), VK_VERSION_PATCH(p.apiVersion),
                  VK_VERSION_MAJOR(p.driverVersion), VK_VERSION_MINOR(p.driverVersion),
                  VK_VERSION_PATCH(p.driverVersion), p.vendorID, p.deviceID);
    s += line;
    std::snprintf(line, sizeof line, "shared memory: %s, unified memory: %s\n",
                  p.limits.maxComputeSharedMemorySize >= 32 * 1024 ? "ok" : "small",
                  (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) ? "yes" : "no");
    s += line;
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++) {
        const VkMemoryHeap& h = mp.memoryHeaps[i];
        std::string flags;
        if (h.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) flags += "device-local";
        if (h.flags & VK_MEMORY_HEAP_MULTI_INSTANCE_BIT) flags += flags.empty() ? "multi-instance" : ",multi-instance";
        std::snprintf(line, sizeof line, "heap %u: %.0f MB%s%s%s\n", i, h.size / 1048576.0,
                      flags.empty() ? "" : " (", flags.c_str(), flags.empty() ? "" : ")");
        s += line;
    }
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        const VkMemoryType& t = mp.memoryTypes[i];
        std::string f;
        if (t.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) f += "device-local ";
        if (t.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) f += "host-visible ";
        if (t.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) f += "coherent ";
        std::snprintf(line, sizeof line, "memtype %u: heap %u %s\n", i, t.heapIndex, f.c_str());
        s += line;
    }
    std::snprintf(line, sizeof line, "device-local budget: %.0f MB\n", deviceLocalMb_);
    s += line;
    return s;
}

}  // namespace gpu
}  // namespace ofc
