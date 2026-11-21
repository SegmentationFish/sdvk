// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
//
// Prints the tensor type histogram of a GGUF file and the arena the sdvk loader would need for
// it, so "why does this model not fit" can be answered without running the full pipeline.
// --tensors lists every tensor (name, type, ggml dims, logical shape, bytes); --tensor <substr>
// filters that list; --stats adds a numeric summary per tensor, which is what tells a corrupted
// export apart from a dtype mismatch.
#include "Modules/Model/Gguf.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using ofc::GgufFile;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: ggufinfo <file.gguf> [--tensors] [--tensor <substr>] [--stats] [--limit <n>]\n");
        return 2;
    }
    bool listAll = false, stats = false;
    std::string filter;
    size_t limit = 0;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--tensors") {
            listAll = true;
        } else if (a == "--stats") {
            stats = true;
        } else if (a == "--tensor" && i + 1 < argc) {
            filter = argv[++i];
            listAll = true;
        } else if (a == "--limit" && i + 1 < argc) {
            limit = (size_t)std::strtoull(argv[++i], nullptr, 10);
        } else {
            std::printf("unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    GgufFile g;
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
        bool asQ8 = t.dims.size() > 1 &&
                    (t.type == ofc::GgmlType::Q8_0 || ofc::ggmlTypeIsKQuant(t.type));
        if (asQ8) {
            uint64_t elems = t.nElems();
            uint64_t blocks = elems / 32;
            totalQ8 += blocks * 34;  // int8 payload + fp32 scale (K-quants are re-encoded)
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
    if (g.has("general.architecture")) std::printf("architecture: %s\n", g.getStr("general.architecture", "?").c_str());
    if (g.has("general.name")) std::printf("name: %s\n", g.getStr("general.name", "?").c_str());

    if (!listAll) {
        // a quick structural verdict: which component families are present and how many
        int nUnet = 0, nTe1 = 0, nTe2 = 0, nVaeEnc = 0, nVaeDec = 0, nLabel = 0;
        for (auto& t : g.tensors()) {
            const std::string& n = t.name;
            if (n.rfind("model.diffusion_model.", 0) == 0 || n.rfind("input_blocks.", 0) == 0 ||
                n.rfind("output_blocks.", 0) == 0 || n.rfind("middle_block.", 0) == 0)
                nUnet++;
            if (n.find("cond_stage_model.transformer.") != std::string::npos ||
                n.find("te1.") != std::string::npos || n.find("text_encoder.") != std::string::npos)
                nTe1++;
            if (n.find("cond_stage_model.1.") != std::string::npos || n.find("te2.") != std::string::npos ||
                n.find("text_encoder_2.") != std::string::npos)
                nTe2++;
            if (n.find("first_stage_model.encoder.") != std::string::npos || n.find("vae.encoder.") != std::string::npos)
                nVaeEnc++;
            if (n.find("first_stage_model.decoder.") != std::string::npos || n.find("vae.decoder.") != std::string::npos)
                nVaeDec++;
            if (n.find("label_emb.") != std::string::npos) nLabel++;
        }
        std::printf("components: unet %d, te1 %d, te2 %d, vae-enc %d, vae-dec %d, label_emb %d -> %s\n", nUnet, nTe1,
                    nTe2, nVaeEnc, nVaeDec, nLabel, nLabel ? "SDXL" : "SD1.5");
        return 0;
    }

    size_t shown = 0, matched = 0;
    for (auto& t : g.tensors()) {
        if (!filter.empty() && t.name.find(filter) == std::string::npos) continue;
        matched++;
        auto sh = t.shape();
        std::printf("%-62s %-6s nd=%u dims=(%llu,%llu,%llu,%llu) shape=(%llu,%llu) %.2f MB\n", t.name.c_str(),
                    ofc::ggmlTypeName(t.type), (unsigned)t.dims.size(), (unsigned long long)t.dims[0],
                    (unsigned long long)(t.dims.size() > 1 ? t.dims[1] : 0),
                    (unsigned long long)(t.dims.size() > 2 ? t.dims[2] : 0),
                    (unsigned long long)(t.dims.size() > 3 ? t.dims[3] : 0), (unsigned long long)sh.first,
                    (unsigned long long)sh.second, t.nbytes / 1048576.0);
        if (stats) {
            std::string e2;
            std::vector<float> v = g.readFloat(t.name, &e2);
            if (v.empty()) {
                std::printf("    stats: %s\n", e2.c_str());
            } else {
                size_t n = v.size(), probe = n > 65536 ? 65536 : n, step = n / probe ? n / probe : 1;
                double s1 = 0, s2 = 0, mx = 0;
                size_t c = 0;
                for (size_t i = 0; i < n; i += step) {
                    s1 += v[i];
                    s2 += (double)v[i] * v[i];
                    if (std::fabs(v[i]) > mx) mx = std::fabs(v[i]);
                    c++;
                }
                double mean = s1 / (double)c, var = s2 / (double)c - mean * mean;
                std::printf("    stats: n=%zu first=%.6g mean=%.6g std=%.6g absmax=%.6g\n", n, v[0], mean,
                            std::sqrt(var < 0 ? 0 : var), mx);
            }
        }
        if (limit && ++shown >= limit) {
            std::printf("... truncated at %zu listed rows\n", shown);
            break;
        }
    }
    if (!filter.empty()) std::printf("matched %zu tensors\n", matched);
    return 0;
}
