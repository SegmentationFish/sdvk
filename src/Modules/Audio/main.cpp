// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
// sdvkaudio: Stable Audio Open Small (text -> 44.1 kHz stereo) on the sdvk engine.
#include "SaGen.hpp"
#include "SaDit.hpp"
#include "Modules/Model/Gguf.hpp"
#include "Modules/GPU/gpu_ops.hpp"
#include "SaVae.hpp"
#include "Wav.hpp"

#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace ofc::audio;

static void usage() {
    std::printf(
        "sdvkaudio - Stable Audio Open Small (text to 44.1 kHz stereo)\n"
        "\n"
        "usage: sdvkaudio -m <stable-audio-small dir> \"prompt\" [options]\n"
        "\n"
        "  -m, --model <dir>    model folder (model.safetensors, t5-base/, vae/)\n"
        "  --seconds <f>        clip length in seconds (default 6)\n"
        "  --steps <n>          pingpong sampling steps (default 8)\n"
        "  --seed <n>           noise seed (default 0)\n"
        "  --out <file.wav>     output path (default sdvk_audio.wav)\n"
        "  --spv <dir>          shader directory (default build/shaders)\n"
        "  --threads <n>        CPU threads for the host stages\n"
        "  --stream             decode and write the WAV incrementally\n"
        "  --chunk <frames>     latent frames per streamed chunk (default 8)\n"
        "  --overlap <frames>   decode context per side (default 4)\n"
        "  --t5check <dir>      dump tokenizer ids + T5 hidden state to <dir>, exit\n"
        "  --vae-decode <f32>   decode a raw [128, frames] f32 latent with --vae <file>\n"
        "  --frames <n>         latent frames for --vae-decode\n"
        "  --vae <file>         MiniMax vocoder safetensors (dav)\n"
        "  --dit <file.gguf>    MiniMax Music 3 DiT + condition encoder GGUF\n"
        "  --dit-check <dir>    run the DiT on <dir>/{cond_in,dit_x}.f32 and dump results there\n"
        "  --layers <n>         limit the DiT to the first n blocks (reference comparison)\n"
        "  --gguf-dump <gguf> <tensor> <out.f32>   dequantize one GGUF tensor to f32, exit\n"
        "  -h, --help           this text\n");
}

int main(int argc, char** argv) {
    SaParams p;
    std::string t5check;
    std::string vaeDecode, vaePath, ditPath, ditCheck;
    int frames = 0, layers = 0;
    if (argc >= 2 && std::string(argv[1]) == "--gemm-check") {
        ofc::gpu::G.allocGradArena = false;
        ofc::gpu::G.stepBytes = 256ull << 20;
        ofc::gpu::G.scratchBytes = 64ull << 20;
        ofc::gpu::G.paramBytes = 64ull << 20;
        ofc::gpu::gInit("build/shaders");
        float e32 = 0, e8 = 0;
        for (int N : {128, 2048}) {
            saSelfTestGemm(86, N, 2048, e32, e8);
            std::printf("gemm M86 N%d K2048: fp32 rel %.3e  q8 rel %.3e\\n", N, e32, e8);
        }
        return 0;
    }
    if (argc >= 5 && std::string(argv[1]) == "--gguf-dump") {
        ofc::GgufFile g;
        std::string err;
        if (!g.open(argv[2], &err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        std::vector<float> v = g.readFloat(argv[3], &err);
        if (!err.empty()) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        FILE* f = std::fopen(argv[4], "wb");
        if (!f) return 1;
        std::fwrite(v.data(), 4, v.size(), f);
        std::fclose(f);
        const ofc::GgufTensor* t = g.find(argv[3]);
        auto sh = t->shape();
        std::printf("%s: %zu floats (%llu x %llu)\n", argv[3], v.size(), (unsigned long long)sh.first,
                    (unsigned long long)sh.second);
        return 0;
    }
    std::string modelDir;
    std::string prompt;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto val = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "-m" || a == "--model") {
            p.modelDir = val();
        } else if (a == "--seconds") {
            p.seconds = val();
            p.secondsTotal = std::strtof(p.seconds.c_str(), nullptr);
        } else if (a == "--steps") {
            p.steps = std::atoi(val());
        } else if (a == "--seed") {
            p.seed = std::atoi(val());
        } else if (a == "--out") {
            p.outWav = val();
        } else if (a == "--spv") {
            p.spvDir = val();
        } else if (a == "--threads") {
            p.threads = std::atoi(val());
        } else if (a == "--stream") {
            p.stream = true;
        } else if (a == "--chunk") {
            p.chunkFrames = std::atoi(val());
        } else if (a == "--overlap") {
            p.overlapFrames = std::atoi(val());
        } else if (a == "--t5check") {
            t5check = val();
        } else if (a == "--vae-decode") {
            vaeDecode = val();
        } else if (a == "--vae") {
            vaePath = val();
        } else if (a == "--frames") {
            frames = std::atoi(val());
        } else if (a == "--dit") {
            ditPath = val();
        } else if (a == "--dit-check") {
            ditCheck = val();
        } else if (a == "--layers") {
            layers = std::atoi(val());
        } else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "error: unknown option %s\n", a.c_str());
            usage();
            return 1;
        } else {
            prompt = a;
        }
    }
    p.prompt = prompt;
    if (!vaeDecode.empty()) {
        if (vaePath.empty() || frames <= 0) {
            std::fprintf(stderr, "error: --vae-decode needs --vae <file> and --frames <n>\n");
            return 1;
        }
        FILE* f = std::fopen(vaeDecode.c_str(), "rb");
        if (!f) {
            std::fprintf(stderr, "error: cannot open %s\n", vaeDecode.c_str());
            return 1;
        }
        std::vector<float> lat((size_t)128 * frames);
        size_t got = std::fread(lat.data(), 4, lat.size(), f);
        std::fclose(f);
        if (got != lat.size()) {
            std::fprintf(stderr, "error: latent has %zu floats, expected %zu\n", got, lat.size());
            return 1;
        }
        SaVae vae;
        std::string err;
        if (!vae.load(vaePath, p.spvDir, err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        std::vector<float> stereo;
        if (!vae.decode(lat, frames, stereo, err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        ofc::audio::Audio au;
        au.sampleRate = SaVae::kSampleRate;
        au.channels = 2;
        au.samples = std::move(stereo);
        if (!writeWav(p.outWav, au, 16, err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        std::printf("wrote %s (%.2f s stereo %d Hz)\n", p.outWav.c_str(), au.seconds(), au.sampleRate);
        return 0;
    }
    if (!ditCheck.empty()) {
        if (ditPath.empty()) {
            std::fprintf(stderr, "error: --dit-check needs --dit <gguf>\n");
            return 1;
        }
        SaDit dit;
        std::string err;
        if (!dit.load(ditPath, p.spvDir, layers, err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        auto rd = [&](const char* name, size_t n) {
            std::vector<float> v(n);
            std::string path = ditCheck + "/" + name;
            FILE* f = std::fopen(path.c_str(), "rb");
            if (!f || std::fread(v.data(), 4, n, f) != n) {
                std::fprintf(stderr, "error: cannot read %s (%zu floats)\n", path.c_str(), n);
                std::exit(1);
            }
            std::fclose(f);
            return v;
        };
        auto wr = [&](const char* name, const std::vector<float>& v) {
            std::string path = ditCheck + "/" + name;
            FILE* f = std::fopen(path.c_str(), "wb");
            if (f) {
                std::fwrite(v.data(), 4, v.size(), f);
                std::fclose(f);
            }
        };
        std::vector<float> condIn = rd("cond_in.f32", 25 * 8 * 4096);
        std::vector<float> condOut;
        int latLen = 0;
        if (!dit.encodeCond(condIn, 25, condOut, latLen, err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        wr("eng_cond.f32", condOut);
        std::vector<float> x = rd("dit_x.f32", 128 * 86);
        std::vector<float> tv = rd("dit_t.f32", 1);
        std::vector<float> vel;
        if (!dit.forward(x, 86, tv[0], condOut, vel, err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        wr("eng_dit.f32", vel);
        std::printf("dit-check: cond %dx%d, latent %dx86 written to %s\n", latLen, 2048, 128, ditCheck.c_str());
        return 0;
    }
    if (p.modelDir.empty()) {
        std::fprintf(stderr, "error: -m <model dir> is required\n");
        return 1;
    }
    if (!t5check.empty()) {
        saDumpT5(p, t5check);
        return 0;
    }
    if (prompt.empty()) {
        std::fprintf(stderr, "error: a prompt is required\n");
        return 1;
    }
    std::string err;
    if (!saGenerate(p, err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    return 0;
}
