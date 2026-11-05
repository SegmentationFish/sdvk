// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
// Generation parameters: one struct shared by the CLI, the HTTP bridge and the web UI.
// Every caller builds a GenParams and hands it to Sdvk run/runner(); nothing here touches Vulkan.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ofc {
namespace sdxl {

enum class SamplerKind { Euler = 0, EulerAncestral = 1 };

inline const char* samplerName(SamplerKind s) {
    switch (s) {
        case SamplerKind::EulerAncestral:
            return "euler_a";
        default:
            return "euler";
    }
}

inline bool samplerFromName(const std::string& n, SamplerKind& out) {
    if (n == "euler" || n.empty()) {
        out = SamplerKind::Euler;
        return true;
    }
    if (n == "euler_a" || n == "euler-ancestral" || n == "eulera") {
        out = SamplerKind::EulerAncestral;
        return true;
    }
    return false;
}

// Where each model component comes from. Empty paths fall back to the directory scan in
// ModelSet.cpp, so a user can point --model-dir at one folder and pass nothing else.
struct ModelPaths {
    std::string modelDir;
    std::string unet;   // .gguf (GGUF) or .safetensors (diffusers / single-file)
    std::string te1;    // CLIP-L
    std::string te2;    // OpenCLIP bigG
    std::string vae;
    std::string tok1;   // directory holding vocab.json + merges.txt
    std::string tok2;
    std::string taesd;  // optional tiny decoder, .safetensors
};

struct Rect {
    uint32_t x = 0, y = 0, w = 0, h = 0;
    bool empty() const { return w == 0 || h == 0; }
};

struct GenParams {
    // ---- prompt
    std::string prompt;
    std::string negative;
    std::vector<int> condTokens;    // optional pre-tokenized (length is checked, not trusted)
    std::vector<int> uncondTokens;
    std::string embCondFile;        // raw 77x2048 fp32, skips the text encoders
    std::string embUncondFile;
    std::string embPoolFile;
    std::string embPoolUFile;

    // ---- sampler
    int steps = 20;
    float cfg = 7.5f;
    float rescale = 0.0f;
    float eta = 0.0f;
    float denoise = 1.0f;
    bool vpred = false;
    bool zsnr = false;
    bool dynthresh = false;
    std::string spacing = "leading";
    SamplerKind sampler = SamplerKind::Euler;
    uint32_t seed = 1234;
    bool seedGiven = false;  // --seed given; otherwise a random seed is drawn per run
    std::string promptsFile;  // --prompts: one prompt per line, reuses the resident UNet
    int device = 0;           // Vulkan compute device index (see /api/devices)
    float hires = 1.0f;      // latent upscale factor before VAE decode (1 = off)
    float hiresRefine = 0.0f;  // >0: re-denoise the upscaled latent in tiles at this strength

    // ---- canvas
    uint32_t width = 512;
    uint32_t height = 0;  // 0 => square (width)
    Rect crop;            // non-empty: img2img/refine region in image pixels
    uint32_t cropOffsetX = 0;
    uint32_t cropOffsetY = 0;
    std::string initImage;  // PNG / PPM, opt-in (useInit)
    bool useInit = false;
    // Inpaint: a second image where white means "repaint here". Needs initImage. The mask is
    // reduced to latent resolution and feathered before it is blended into every sampler step.
    std::string maskImage;
    uint32_t maskFeather = 8;  // feather radius in latent px (0 = hard edge)
    bool maskInvert = false;   // black means repaint
    bool roundtrip = false;    // encode -> decode only, no sampling; validates the VAE pair
    bool noOutpaint = false;

    // ---- models
    ModelPaths models;
    uint32_t estVramMb = 0;
    std::string lora;      // path to a LoRA safetensors file (Kohya or PEFT keys)
    float loraScale = 1.0f;  // user multiplier on alpha/rank

    // ---- output
    std::string outPath = "sdvkgen.png";
    int pngCompress = 1;  // 0 = stored, 1 = fixed-Huffman LZ77
    std::string latentOut = "sdvkgen_latent.bin";
    bool noVae = false;
    bool saveLatent = true;
    std::string previewOut;  // TAESD preview, written before the full decode
    bool infoOut = true;     // write the parameters as a text chunk into the PNG

    // ---- runtime
    uint32_t stepMb = 0, scratchMb = 0, vStepMb = 0;
    uint32_t vaeTile = 48;    // tile edge in latent px; 0 forces the full-image pass
    uint32_t vaeOverlap = 8;  // latent px of overlap between VAE tiles
    uint32_t paramMb = 0;    // UNet weight arena, 0 = auto from the file size
    uint32_t teParamMb = 0;  // text encoder arena, 0 = auto from the file sizes
    std::string spvDir = "shaders";

    // Polled once per sampler step (and once before the VAE decode). Returning true aborts the
    // job with error "cancelled". Left empty by the CLI; the HTTP server wires it to /api/cancel.
    std::function<bool()> shouldCancel;

    // ---- debug
    bool textcheck = false;
    bool clipdbg = false;
    bool unetdbg = false;
    std::string dbgStage;
    bool probe = false;
    std::string probeX;
    float probeT = 999.0f;

    uint32_t heightOrWidth() const { return height ? height : width; }
};

// human readable parameter block embedded into the PNG (and echoed by /api/info)
std::string infoText(const GenParams& p, const std::string& modelName, const std::string& device);

}  // namespace sdxl
}  // namespace ofc
