// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
//
// Prints every Vulkan memory heap with its budget, so an allocation failure can be told apart
// from "the desktop is holding the rest". Read-only: it never allocates device memory.
#include <cstdio>
#include <cstdint>
#include <vector>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "volk/volk.h"

int main() {
    if (volkInitialize() != VK_SUCCESS) {
        std::printf("volkInitialize failed: no loader\n");
        return 1;
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) {
        std::printf("vkCreateInstance failed\n");
        return 1;
    }
    volkLoadInstance(inst);

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    if (!n) {
        std::printf("no physical device\n");
        return 1;
    }
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(inst, &n, devs.data());

    for (uint32_t d = 0; d < n; d++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[d], &p);
        std::printf("device %u: %s (api %u.%u.%u)\n", d, p.deviceName, VK_VERSION_MAJOR(p.apiVersion),
                    VK_VERSION_MINOR(p.apiVersion), VK_VERSION_PATCH(p.apiVersion));
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(devs[d], &mp);
        for (uint32_t h = 0; h < mp.memoryHeapCount; h++) {
            const VkMemoryHeap& heap = mp.memoryHeaps[h];
            std::printf("  heap %u: %.0f MB%s\n", h, heap.size / 1048576.0,
                        (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? "  [device-local]" : "");
            for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
                const VkMemoryType& t = mp.memoryTypes[i];
                if (t.heapIndex != h) continue;
                std::printf("    type %2u: local=%d hostVisible=%d hostCoherent=%d cached=%d\n", i,
                            (t.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? 1 : 0,
                            (t.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? 1 : 0,
                            (t.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? 1 : 0,
                            (t.propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? 1 : 0);
            }
        }
    }
    vkDestroyInstance(inst, nullptr);
    return 0;
}
