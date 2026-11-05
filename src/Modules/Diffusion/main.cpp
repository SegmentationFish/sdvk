// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
// Command line front end. Everything here maps argv onto GenParams and prints human output;
// the image itself is produced by sdxl::run(), the same call the HTTP bridge makes.
#include "Modules/Diffusion/Eval.hpp"
#include "Modules/Diffusion/GenParams.hpp"
#include "Modules/Diffusion/ModelSet.hpp"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace ofc::sdxl;

namespace fs = std::filesystem;

static const char* kVersion = "0.3.0";
static const char* kLicense = "GPL-3.0-only";

// Ctrl+C used to kill the process with several GB of device memory live, which makes the WDDM
// driver tear the context down mid-command and the display reset (black screen for seconds).
// Catch it, let the sampler stop at the next safe point and release everything in order.
static volatile std::sig_atomic_t gStop = 0;
static void onSignal(int) { gStop = 1; }

static void usage() {
    std::printf(
        "sdvk %s - SDXL text-to-image on a self-built Vulkan compute engine\n"
        "\n"
        "usage:\n"
        "  sdvkgen \"<prompt>\" [options]              generate an image\n"
        "  sdvkgen -m <model-dir> \"<prompt>\"          point at a model folder instead\n"
        "  sdvkgen --decode <latent.bin> [--out f]    decode a saved latent\n"
        "  sdvkgen --models                           list what a model folder contains\n"
        "\n"
        "model:\n"
        "  -m, --model-dir <dir>   folder with unet/ text_encoder/ text_encoder_2/ vae/ tokenizer*/\n"
        "  --unet <file>           UNet: .gguf or .safetensors (diffusers or full checkpoint)\n"
        "  --te1/--te2 <file>      CLIP-L / CLIP-G components or one shared checkpoint\n"
        "  --vae <file>            VAE safetensors (fp16/bf16/fp32)\n"
        "  --tok/--tok2 <dir>      tokenizer folders holding vocab.json + merges.txt\n"
        "  --taesd <file>          optional tiny decoder\n"
        "  --lora <file>           merge a LoRA safetensors into the Q8 UNet at load\n"
        "  --lora-scale <f>        LoRA strength multiplier (default 1.0)\n"
        "  --list-models <dir>     print how a folder is classified and exit\n"
        "\n"
        "sampler:\n"
        "  --prompt <text>         prompt (or give it as the first positional argument)\n"
        "  --neg <text>            negative prompt\n"
        "  --steps <n>             sampler steps (default 20)\n"
        "  --cfg <f>               guidance scale (default 7.5; 3.0 recommended for v-pred)\n"
        "  --seed <n>              noise seed (default: random each run)\n"
        "  --prompts <file>        one prompt per line; the UNet stays loaded across the batch\n"
        "  --sampler <name>        euler | euler_a | heun | lms | dpm2 | dpm2_a | dpmpp_2m | dpmpp_2s_a | ddim\n"
        "                          (default euler; heun/dpm2/dpm2_a use 2 model evals per step)\n"
        "  --spacing <s>           leading | trailing (SDXL checkpoints use trailing)\n"
        "  --vpred                 v-prediction sampling\n"
        "  --zsnr                  zero terminal SNR schedule\n"
        "  --rescale <f>           CFG rescale, 0..1 (0 = off)\n"
        "  --dynthresh             dynamic thresholding of x0\n"
        "  --eta <f>               ancestral noise strength (euler_a only)\n"
        "\n"
        "canvas:\n"
        "  --size <n | WxH>        square edge, or width x height (multiple of 8; default 512)\n"
        "  --width/--height <n>    non-square canvas\n"
        "  --init <image>          img2img seed image (PNG or PPM), VAE-encoded into the latent\n"
        "  --denoise <f>           img2img strength, 0..1 (default 0.75 with --init)\n"
        "  --crop <x,y,w,h>        region of the seed image to use\n"
        "  --mask <image>          inpaint mask (white = repaint); needs --init\n"
        "  --mask-feather <n>      mask edge feather in latent px (default 8, 0 = hard)\n"
        "  --mask-invert           black means repaint\n"
        "  --roundtrip             encode the seed image and decode it back, no sampling\n"
        "                          (prints PSNR; the check that the VAE pair agrees)\n"
        "\n"
        "output:\n"
        "  --out <file>            .png (default sdvkgen.png) or .ppm\n"
        "  --png-compress <n>      1 = compressed (default), 0 = stored\n"
        "  --no-vae                stop after sampling and write only the latent\n"
        "  --latent-out <file>     latent path (default sdvkgen_latent.bin)\n"
        "  --no-latent             do not write the latent\n"
        "  --no-info               do not embed the parameters into the PNG\n"
        "\n"
        "runtime:\n"
        "  --spv <dir>             shader (.spv) directory (default shaders)\n"
        "  --param-mb <n>          UNet weight arena in MB (default: from the file size)\n"
        "  --step-mb/--scratch-mb/--vstep-mb <n>   arena sizes in MB\n"
        "  --vae-tile <n>          decode the VAE in n-latent-px tiles (0 = auto)\n"
        "  --vae-overlap <n>       tile overlap in latent px (default 8)\n"
        "  --hires <f>             post-decode bicubic upscale (1 = off, 2 = 2x output)\n"
        "  --hires-denoise <d>     with --hires: latent upscale + tiled image-to-image re-denoise\n"
        "                          at strength d (0 = plain bicubic, 0.3-0.4 recommended)\n"
        "  -v, --verbose           print timing detail\n"
        "  --check                 report device, heaps and canvas suggestions\n"
        "  --bench                 --check plus bandwidth and GEMM benchmarks\n"
        "  --textcheck/--clipdbg/--unetdbg/--dbgstage/--probe/--probe-x/--probe-t   debug hooks\n"
        "  -h, --help              this text\n"
        "  --version               version\n"
        "\n"
        "env: OFC_KERNEL_PROFILE=1 per-kernel GPU table on stderr\n"
        "\n"
        "example:\n"
        "  sdvkgen -m models/noobai \"sunflower field, masterpiece\" --vpred --spacing trailing \\\n"
        "    --steps 28 --cfg 3.0 --size 768 --neg \"worst quality, lowres\"\n",
        kVersion);
    std::printf("\nlicense: %s (free software, no warranty)\n", kLicense);
}

// Prompt guess: the whole argv order is preserved, the first non-flag token is the prompt.
static bool isFlag(const std::string& a) { return !a.empty() && a[0] == '-'; }

struct Parsed {
    GenParams p;
    bool help = false, version = false, listModels = false;
    bool check = false, bench = false;
    std::string listDir;
    std::string decodeIn;
    int verbose = 0;
    std::string err;
};

static bool need(int i, int argc, const char* flag, Parsed& out) {
    if (i + 1 < argc) return true;
    out.err = std::string(flag) + " needs a value";
    return false;
}

static Parsed parseArgs(int argc, char** argv) {
    Parsed o;
    GenParams& p = o.p;
    p.models.modelDir = "";
    p.steps = 20;
    p.cfg = 7.5f;
    p.width = 512;
    p.pngCompress = 1;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto val = [&]() -> std::string { return argv[i + 1]; };
        if (a == "-h" || a == "--help") {
            o.help = true;
        } else if (a == "--version" || a == "-V") {
            o.version = true;
        } else if (a == "--check") {
            o.check = true;
        } else if (a == "--bench") {
            o.check = true;
            o.bench = true;
        } else if (a == "-m" || a == "--model-dir") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.models.modelDir = val();
            i++;
        } else if (a == "--unet") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.models.unet = val();
            i++;
        } else if (a == "--te1") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.models.te1 = val();
            i++;
        } else if (a == "--te2") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.models.te2 = val();
            i++;
        } else if (a == "--vae") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.models.vae = val();
            i++;
        } else if (a == "--tok") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.models.tok1 = val();
            i++;
        } else if (a == "--tok2") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.models.tok2 = val();
            i++;
        } else if (a == "--lora") {
            p.lora = val();
        } else if (a == "--lora-scale") {
            p.loraScale = std::strtof(val().c_str(), nullptr);
        } else if (a == "--taesd") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.models.taesd = val();
            i++;
        } else if (a == "--list-models") {
            if (!need(i, argc, a.c_str(), o)) break;
            o.listModels = true;
            o.listDir = val();
            i++;
        } else if (a == "--models") {
            o.listModels = true;
        } else if (a == "--prompt") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.prompt = val();
            i++;
        } else if (a == "--neg" || a == "--negative") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.negative = val();
            i++;
        } else if (a == "--steps") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.steps = std::atoi(val().c_str());
            i++;
        } else if (a == "--cfg" || a == "--scale") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.cfg = std::strtof(val().c_str(), nullptr);
            i++;
        } else if (a == "--rescale") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.rescale = std::strtof(val().c_str(), nullptr);
            i++;
        } else if (a == "--eta") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.eta = std::strtof(val().c_str(), nullptr);
            i++;
        } else if (a == "--denoise" || a == "--strength") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.denoise = std::strtof(val().c_str(), nullptr);
            i++;
        } else if (a == "--size") {
            if (!need(i, argc, a.c_str(), o)) break;
            unsigned w = 0, h = 0;
            if (std::sscanf(val().c_str(), "%ux%u", &w, &h) == 2) {
                p.width = w;
                p.height = h;
            } else {
                p.width = (uint32_t)std::strtoul(val().c_str(), nullptr, 10);
                p.height = 0;
            }
            i++;
        } else if (a == "--width") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.width = (uint32_t)std::strtoul(val().c_str(), nullptr, 10);
            i++;
        } else if (a == "--height") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.height = (uint32_t)std::strtoul(val().c_str(), nullptr, 10);
            i++;
        } else if (a == "--seed") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.seed = (uint32_t)std::strtoul(val().c_str(), nullptr, 10);
            p.seedGiven = true;
            i++;
        } else if (a == "--device") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.device = std::atoi(val().c_str());
            i++;
        } else if (a == "--prompts") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.promptsFile = val();
            i++;
        } else if (a == "--sampler") {
            if (!need(i, argc, a.c_str(), o)) break;
            if (!samplerFromName(val(), p.sampler))
                o.err = "unknown sampler '" + val() +
                        "' (euler, euler_a, heun, lms, dpm2, dpm2_a, dpmpp_2m, dpmpp_2s_a, ddim)";
            i++;
        } else if (a == "--spacing") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.spacing = val();
            i++;
        } else if (a == "--vpred" || a == "--v-pred") {
            p.vpred = true;
        } else if (a == "--zsnr") {
            p.zsnr = true;
        } else if (a == "--dynthresh") {
            p.dynthresh = true;
        } else if (a == "--init") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.initImage = val();
            p.useInit = true;
            p.denoise = 0.75f;
            i++;
        } else if (a == "--mask") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.maskImage = val();
            i++;
        } else if (a == "--mask-feather") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.maskFeather = (uint32_t)std::max(0, std::atoi(val().c_str()));
            i++;
        } else if (a == "--mask-invert") {
            p.maskInvert = true;
        } else if (a == "--roundtrip") {
            p.roundtrip = true;
            p.useInit = true;
        } else if (a == "--crop") {
            if (!need(i, argc, a.c_str(), o)) break;
            unsigned x = 0, y = 0, w = 0, h = 0;
            if (std::sscanf(val().c_str(), "%u,%u,%u,%u", &x, &y, &w, &h) != 4) {
                o.err = "--crop expects x,y,w,h";
            } else {
                p.crop = Rect{x, y, w, h};
            }
            i++;
        } else if (a == "--out" || a == "-o") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.outPath = val();
            i++;
        } else if (a == "--png-compress") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.pngCompress = std::atoi(val().c_str()) ? 1 : 0;
            i++;
        } else if (a == "--no-vae") {
            p.noVae = true;
        } else if (a == "--latent-out") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.latentOut = val();
            i++;
        } else if (a == "--no-latent") {
            p.saveLatent = false;
        } else if (a == "--no-info") {
            p.infoOut = false;
        } else if (a == "--decode") {
            if (!need(i, argc, a.c_str(), o)) break;
            o.decodeIn = val();
            i++;
        } else if (a == "--spv") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.spvDir = val();
            i++;
        } else if (a == "--param-mb") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.paramMb = (uint32_t)std::strtoul(val().c_str(), nullptr, 10);
            i++;
        } else if (a == "--step-mb") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.stepMb = (uint32_t)std::strtoul(val().c_str(), nullptr, 10);
            i++;
        } else if (a == "--scratch-mb") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.scratchMb = (uint32_t)std::strtoul(val().c_str(), nullptr, 10);
            i++;
        } else if (a == "--vstep-mb") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.vStepMb = (uint32_t)std::strtoul(val().c_str(), nullptr, 10);
            i++;
        } else if (a == "--vae-tile") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.vaeTile = (uint32_t)std::strtoul(val().c_str(), nullptr, 10);
            i++;
        } else if (a == "--vae-overlap") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.vaeOverlap = (uint32_t)std::strtoul(val().c_str(), nullptr, 10);
            i++;
        } else if (a == "--hires") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.hires = (float)std::atof(val().c_str());
            if (p.hires < 1.0f || p.hires > 4.0f) o.err = "--hires expects 1.0 .. 4.0";
            i++;
        } else if (a == "--hires-denoise") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.hiresRefine = (float)std::atof(val().c_str());
            if (p.hiresRefine < 0.0f || p.hiresRefine > 1.0f) o.err = "--hires-denoise expects 0 .. 1";
            i++;
        } else if (a == "--lat-in") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.initImage = val();
            i++;
        } else if (a == "--emb-cond" || a == "--emb-uncond" || a == "--emb-pool" || a == "--emb-poolu") {
            if (!need(i, argc, a.c_str(), o)) break;
            std::string v = val();
            if (a == "--emb-cond") p.embCondFile = v;
            else if (a == "--emb-uncond") p.embUncondFile = v;
            else if (a == "--emb-pool") p.embPoolFile = v;
            else p.embPoolUFile = v;
            i++;
        } else if (a == "--textcheck") {
            p.textcheck = true;
        } else if (a == "--clipdbg") {
            p.clipdbg = true;
        } else if (a == "--unetdbg") {
            p.unetdbg = true;
        } else if (a == "--dbgstage") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.dbgStage = val();
            i++;
        } else if (a == "--probe") {
            p.probe = true;
        } else if (a == "--probe-x") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.probeX = val();
            i++;
        } else if (a == "--probe-t") {
            if (!need(i, argc, a.c_str(), o)) break;
            p.probeT = std::strtof(val().c_str(), nullptr);
            i++;
        } else if (a == "-v" || a == "--verbose") {
            o.verbose++;
        } else if (isFlag(a)) {
            if (o.err.empty()) o.err = "unknown option " + a;
        } else {
            positional.push_back(a);
        }
        if (!o.err.empty()) break;
    }
    if (p.prompt.empty() && !positional.empty()) p.prompt = positional.front();
    if (p.height == 0) p.height = p.width;
    // A fixed default seed made every default-parameter run reproduce the same noise field, and
    // some noise fields sit on a model out-of-distribution spot (over-exposed, magenta skin at
    // 768-class canvases). Draw a fresh seed unless the caller pinned one.
    if (!p.seedGiven) p.seed = std::random_device{}();
    if (p.useInit && p.initImage.empty()) o.err = "--init needs a path";
    if (!p.maskImage.empty() && p.initImage.empty()) o.err = "--mask needs --init";
    return o;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);
#ifdef SIGBREAK
    std::signal(SIGBREAK, onSignal);
#endif
    if (argc < 2) {
        usage();
        return 1;
    }
    Parsed o = parseArgs(argc, argv);
    if (o.help) {
        usage();
        return 0;
    }
    if (o.version) {
        std::printf("sdvkgen %s\nlicense: %s\n", kVersion, kLicense);
        return 0;
    }
    if (!o.err.empty()) {
        std::printf("error: %s\n(run with --help)\n", o.err.c_str());
        return 1;
    }
    GenParams& p = o.p;

    if (o.check) {
        std::printf("%s", deviceCheck(p.spvDir.empty() ? "shaders" : p.spvDir, o.bench).c_str());
        return 0;
    }

    if (o.listModels) {
        std::string dir = o.listDir.empty() ? p.models.modelDir : o.listDir;
        if (dir.empty()) dir = ".";
        GenParams probe = p;
        probe.models.modelDir = dir;
        ModelSet set;
        std::string err;
        probe.prompt = "x";
        bool ok = resolveModels(probe, set, err);
        if (!ok) {
            std::printf("error: %s\n", err.c_str());
            return 1;
        }
        std::printf("model dir: %s\n", dir.c_str());
        std::printf("  unet:    %s\n", probe.models.unet.c_str());
        std::printf("  clip-l:  %s\n", probe.models.te1.c_str());
        std::printf("  clip-g:  %s\n", probe.models.te2.c_str());
        std::printf("  vae:     %s\n", probe.models.vae.c_str());
        if (!probe.models.tok1.empty()) std::printf("  tok1:    %s\n", probe.models.tok1.c_str());
        if (!probe.models.tok2.empty()) std::printf("  tok2:    %s\n", probe.models.tok2.c_str());
        if (!probe.models.taesd.empty()) std::printf("  taesd:   %s\n", probe.models.taesd.c_str());
        for (auto& w : set.warn) std::printf("  note: %s\n", w.c_str());
        return 0;
    }

    // --decode: VAE only, no prompt and no text encoders
    if (!o.decodeIn.empty()) {
        EvalModel m;
        ProgressFn onProgress = [&o](const Progress& pr) {
            if (pr.kind == Progress::Kind::Warning) {
                std::printf("warning: %s\n", pr.text.c_str());
            } else if (o.verbose > 0) {
                std::printf("[%.0f%%] %s\n", pr.progress * 100.0, pr.text.c_str());
            }
        };
        RunResult r = decodeLatentFile(m, p, o.decodeIn, onProgress);
        if (!r.ok) {
            std::printf("error: %s\n", r.error.c_str());
            return 1;
        }
        std::printf("decoded %s -> %s  %ux%u  %.0f ms\n", o.decodeIn.c_str(), r.outPath.c_str(), r.width, r.height,
                    r.totalMs);
        return 0;
    }

    if (p.prompt.empty() && p.promptsFile.empty()) {
        std::printf("error: no prompt given\n(run with --help)\n");
        return 1;
    }
    if (p.width % 64 || p.height % 64 || p.width < 64 || p.height < 64) {
        std::printf("error: --width/--height must be multiples of 64 and at least 64 (e.g. 832x1216, 768x1024)\n");
        return 1;
    }

    EvalModel m;
    ProgressFn onProgress = [&o](const Progress& pr) {
        if (pr.kind == Progress::Kind::Warning) {
            std::printf("warning: %s\n", pr.text.c_str());
        } else if (o.verbose > 0) {
            std::printf("[%.0f%%] %s (%.0f ms)\n", pr.progress * 100.0, pr.text.c_str(), pr.ms);
        }
    };
    auto t0 = std::chrono::steady_clock::now();
    p.shouldCancel = [] { return gStop != 0; };
    // --prompts keeps the process (and therefore the 2.9 GB resident UNet) alive across prompts.
    // Two separate processes cannot share Vulkan device memory at all, so batching inside one
    // process is the only way to avoid reloading the weights for every image.
    std::vector<std::string> batch;
    if (!p.promptsFile.empty()) {
        std::ifstream f(p.promptsFile);
        if (!f) {
            std::printf("error: cannot read %s\n", p.promptsFile.c_str());
            return 1;
        }
        std::string line;
        while (std::getline(f, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) line.pop_back();
            size_t b = line.find_first_not_of(" \t");
            if (b == std::string::npos || line[b] == '#') continue;
            batch.push_back(line.substr(b));
        }
        if (batch.empty()) {
            std::printf("error: no prompts in %s\n", p.promptsFile.c_str());
            return 1;
        }
    } else {
        batch.push_back(p.prompt);
    }
    RunResult r;
    for (size_t bi = 0; bi < batch.size(); bi++) {
        GenParams q = p;
        q.prompt = batch[bi];
        if (!p.seedGiven) q.seed = std::random_device{}();
        if (batch.size() > 1) {
            std::string path = p.outPath;
            char suffix[16];
            std::snprintf(suffix, sizeof suffix, "_%03u", (unsigned)bi);
            size_t dot = path.find_last_of('.');
            if (dot == std::string::npos) path = path + suffix + ".png";
            else path = path.substr(0, dot) + suffix + path.substr(dot);
            q.outPath = path;
        }
        if (batch.size() > 1) {
            std::printf("\n[%zu/%zu] %.70s%s\n", bi + 1, batch.size(), batch[bi].c_str(),
                        batch[bi].size() > 70 ? "..." : "");
        }
        r = run(m, q, onProgress);
        if (!r.ok) break;
    }
    if (!r.ok) {
        std::printf("error: %s\n", r.error.c_str());
        return 1;
    }
    double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n%s  %s  %ux%u  seed %u  %d steps\n", r.ok ? "ok" : "failed", r.modelName.c_str(), r.width,
                r.height, r.seed, r.steps);
    if (!r.outPath.empty()) std::printf("  image: %s\n", r.outPath.c_str());
    std::printf("  total: %.2f s  (%.3f s/step)\n", total, r.secPerStep);
    return 0;
}
