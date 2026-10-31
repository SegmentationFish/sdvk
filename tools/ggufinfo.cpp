// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
//
// Prints the tensor type histogram of a GGUF file and the arena the sdvk loader would need for
// it, so "why does this model not fit" can be answered without running the full pipeline.
#include "Modules/Model/Gguf.hpp"

#include <cstdio>
#include <map>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: ggufinfo <file.gguf>\n");
        return 2;
    }
    ofc::GgufFile g;
    std::string err;
    if (!g.open(argv[1], &err)) {
        std::printf("error: %s\n", err.c_str());
        return 1;
    }
    std::map<uint32_t, std::pair<int, uint64_t>> hist;
    uint64_t totalQ8 = 0, totalMisc = 0;
    for (auto& t : g.tensors()) {
        auto& e = hist[(uint32_t)t.type];
        e.first++;
        e.second += t.nbytes;
        if (t.type == ofc::GgmlType::Q8_0) {
            uint64_t elems = t.nElems();
            uint64_t blocks = elems / 32;
            totalQ8 += blocks * 34;  // int8 payload + fp32 scale
        } else {
            totalMisc += t.nElems() * 4;
        }
    }
    std::printf("file: %s (%llu tensors, %.1f MB)\n", argv[1], (unsigned long long)g.tensorCount(),
                g.fileSize() / 1048576.0);
    for (auto& kv : hist) {
        std::printf("  type %-3u %-12s %5d tensors  %8.1f MB\n", kv.first,
                    ofc::ggmlTypeName((ofc::GgmlType)kv.first), kv.second.first, kv.second.second / 1048576.0);
    }
    std::printf("arena estimate: Q8_0 %.1f MB + fp32 %.1f MB = %.1f MB\n", totalQ8 / 1048576.0,
                totalMisc / 1048576.0, (totalQ8 + totalMisc) / 1048576.0);
    return 0;
}
