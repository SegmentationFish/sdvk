// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
// Runner API: one image in, one image out, plus a model handle that survives between jobs.
// Both entry points are synchronous and own their GPU work; neither parses a command line.
#pragma once

#include "GenParams.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ofc {
namespace sdxl {

struct Image {
    uint32_t width = 0, height = 0;
    std::vector<float> rgb;  // CHW, [0,1] after the [-1,1] decode mapping
};

struct Progress {
    enum class Kind { Load, Encode, Step, Decode, Save, Done, Warning };
    Kind kind = Kind::Step;
    int step = 0, steps = 0;
    double progress = 0.0;  // 0..1 overall
    double ms = 0.0;
    std::string text;
};

struct RunResult {
    bool ok = true;
    std::string error;
    Image image;
    std::string outPath;
    std::string modelName;
    std::string device;
    uint32_t seed = 0;
    uint32_t width = 0, height = 0;
    double totalMs = 0.0;
    double unetMs = 0.0;
    double vaeMs = 0.0;
    int steps = 0;
    double secPerStep = 0.0;
    bool latinOnly = false;  // tokenizer produced no word ids
};

using ProgressFn = std::function<void(const Progress&)>;

// Loads the UNet once and keeps it resident; the VAE load is reused when the file is unchanged.
struct EvalModel {
    bool loaded = false;
    std::string unetPath, vaePath;
    std::string deviceName;
    std::string modelName;
    std::vector<std::string> warnings;
    void* impl = nullptr;  // internal state; do not touch
};

bool loadModel(EvalModel& m, const GenParams& p, const ProgressFn& onProgress, std::string& err);
void unloadModel(EvalModel& m);
bool ensureDevice(const std::string& spvDir, std::string& err);

// Full generate() equivalent: encode, sample, write the latent, optional TAESD preview, decode,
// write the image. Throws only on programmer error; runtime failures land in RunResult::error.
RunResult run(EvalModel& m, const GenParams& p, const ProgressFn& onProgress);

// VAE-only path: read an .ofcl latent, decode it with p.models.vae, write p.outPath.
RunResult decodeLatentFile(EvalModel& m, const GenParams& p, const std::string& latentPath, const ProgressFn& onProgress);

// Standalone PNG writer (also used by the CLI's --decode path and by tests).
bool writePngFile(const std::string& path, const std::vector<float>& rgbChw, uint32_t w, uint32_t h, int compress,
                  const std::string& info);

// Device diagnostics for the CLI: Vulkan device/heap report plus canvas suggestions; with
// bench=true it also measures upload/download bandwidth and Q8/fp32 GEMM throughput.
std::string deviceCheck(const std::string& spvDir, bool bench);

}  // namespace sdxl
}  // namespace ofc
