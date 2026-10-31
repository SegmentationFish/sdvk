// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
//
// LolavaMem — a small dynamic allocator for GPU arenas.
//
// One contiguous byte range, aligned offsets handed out to several logical owners (weights,
// per-job activations, transient uploads) so the heap is not carved into fixed pools that each
// waste their tail. Blocks are kept in a physical list; allocation is first-fit with immediate
// splitting, free coalesces with both neighbours, so steady-state fragmentation stays low.
// Host-side only: offsets, not VkDeviceMemory.
//
// Complexity: allocate O(blocks) (a few dozen in practice), free O(1) lookup + O(1) coalesce.
#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <cstdio>

namespace ofc::gpu {

class LolavaMem {
public:
    static constexpr uint64_t kInvalid = ~0ull;

    void init(uint64_t bytes, uint64_t alignment = 256);
    uint64_t alloc(uint64_t bytes);
    void free(uint64_t offset);

    uint64_t capacity() const { return capacity_; }
    uint64_t used() const { return used_; }
    uint64_t freeBytes() const { return capacity_ - used_; }
    uint64_t largestFree() const;
    uint64_t allocCount() const { return live_; }
    double fragmentation() const;
    void reset();
    bool check() const;
    // dump the physical block list (debug/CLI)
    void dump(const char* tag) const;

private:
    struct Block {
        uint64_t off = 0;
        uint64_t size = 0;   // whole block incl. one alignment unit of header
        int next = -1;
        bool free = false;
        bool used = false;
    };
    static constexpr uint64_t kMinBlock = 512;
    uint64_t alignUp(uint64_t v) const { return (v + alignment_ - 1) / alignment_ * alignment_; }
    int newTail(uint64_t off, uint64_t size, bool free);

    uint64_t alignment_ = 256;
    uint64_t capacity_ = 0;
    uint64_t used_ = 0;
    uint64_t live_ = 0;
    std::vector<Block> blocks_;
    std::unordered_map<uint64_t, int> byPayload_;
    int head_ = -1;
};

}  // namespace ofc::gpu
