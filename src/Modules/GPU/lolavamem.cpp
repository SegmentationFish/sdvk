// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
#include "Modules/GPU/lolavamem.hpp"

#include <algorithm>

namespace ofc::gpu {

void LolavaMem::init(uint64_t bytes, uint64_t alignment) {
    alignment_ = alignment ? alignment : 256;
    capacity_ = bytes / alignment_ * alignment_;
    used_ = 0;
    live_ = 0;
    head_ = -1;
    blocks_.clear();
    byPayload_.clear();
    blocks_.reserve(256);
    if (capacity_ >= kMinBlock) head_ = newTail(0, capacity_, true);
}

int LolavaMem::newTail(uint64_t off, uint64_t size, bool free) {
    Block b;
    b.off = off;
    b.size = size;
    b.free = free;
    b.used = true;
    b.next = -1;
    blocks_.push_back(b);
    int i = (int)blocks_.size() - 1;
    if (head_ == -1) head_ = i;
    else {
        int t = head_;
        while (blocks_[t].next != -1) t = blocks_[t].next;
        blocks_[t].next = i;
    }
    return i;
}

uint64_t LolavaMem::alloc(uint64_t bytes) {
    if (bytes == 0) return kInvalid;
    uint64_t need = alignUp(bytes) + alignment_;
    if (need < kMinBlock) need = kMinBlock;

    int prev = -1;
    for (int i = head_; i != -1; prev = i, i = blocks_[i].next) {
        Block& b = blocks_[i];
        if (!b.free || b.size < need) continue;
        // take [i]: split off the tail when enough is left for another block
        if (b.size >= need + kMinBlock) {
            uint64_t restOff = b.off + need;
            uint64_t restSize = b.size - need;
            Block r;
            r.off = restOff;
            r.size = restSize;
            r.free = true;
            r.used = true;
            r.next = b.next;
            blocks_.push_back(r);
            int ri = (int)blocks_.size() - 1;
            blocks_[i].size = need;
            blocks_[i].next = ri;
            blocks_[i].free = false;
            b = blocks_[i];  // reference may be stale after push_back; re-read
        } else {
            b.free = false;
        }
        used_ += blocks_[i].size;
        live_++;
        uint64_t payload = blocks_[i].off + alignment_;
        byPayload_[payload] = i;
        (void)prev;
        return payload;
    }
    return kInvalid;
}

void LolavaMem::free(uint64_t offset) {
    auto it = byPayload_.find(offset);
    if (it == byPayload_.end()) return;
    int i = it->second;
    byPayload_.erase(it);
    if (i < 0 || i >= (int)blocks_.size() || !blocks_[i].used || blocks_[i].free) return;
    used_ -= blocks_[i].size;
    live_--;
    blocks_[i].free = true;

    // coalesce with the previous block if it is free
    int prev = -1;
    for (int k = head_; k != i; k = blocks_[k].next) prev = k;
    if (prev != -1 && blocks_[prev].free) {
        blocks_[prev].size += blocks_[i].size;
        blocks_[prev].next = blocks_[i].next;
        blocks_[i].used = false;
        i = prev;
    }
    int n = blocks_[i].next;
    if (n != -1 && blocks_[n].free) {
        blocks_[i].size += blocks_[n].size;
        blocks_[i].next = blocks_[n].next;
        blocks_[n].used = false;
    }
}

uint64_t LolavaMem::largestFree() const {
    uint64_t best = 0;
    for (int i = head_; i != -1; i = blocks_[i].next)
        if (blocks_[i].free) best = std::max(best, blocks_[i].size);
    return best;
}

double LolavaMem::fragmentation() const {
    uint64_t fb = freeBytes();
    if (!fb) return 0.0;
    return 1.0 - (double)largestFree() / (double)fb;
}

void LolavaMem::reset() {
    uint64_t cap = capacity_, al = alignment_;
    init(cap, al);
}

bool LolavaMem::check() const {
    uint64_t sum = 0;
    int prev = -1;
    int count = 0;
    for (int i = head_; i != -1; i = blocks_[i].next) {
        const Block& b = blocks_[i];
        if (!b.used) return false;
        if (b.size == 0 || b.size % alignment_) return false;
        if (b.off % alignment_) return false;
        if (prev != -1 && blocks_[prev].off + blocks_[prev].size != b.off) return false;
        sum += b.size;
        prev = i;
        if (++count > 100000) return false;
    }
    return sum == capacity_;
}

void LolavaMem::dump(const char* tag) const {
    std::printf("[lolavamem %s] cap=%llu used=%llu live=%llu\n", tag,
                (unsigned long long)capacity_, (unsigned long long)used_,
                (unsigned long long)live_);
    for (int i = head_; i != -1; i = blocks_[i].next)
        std::printf("  %s off=%llu size=%llu\n", blocks_[i].free ? "free" : "USED",
                    (unsigned long long)blocks_[i].off, (unsigned long long)blocks_[i].size);
}

}  // namespace ofc::gpu
