// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
// Stable Audio / MiniMax Music 3 generation driver: given a condition track, run the flow-matching
// sampler on the DiT and decode the latent with the vocoder. The autoregressive stage that produces
// the condition track is a separate component.
#include "SaGen.hpp"

#include "SaDit.hpp"
#include "SaVae.hpp"
#include "Wav.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace ofc {
namespace audio {

bool saGenerate(const SaParams& p, std::string& err) {
    if (p.ditPath.empty() || p.vaePath.empty()) {
        err = "generation needs --dit <gguf> and --vae <dav.safetensors>";
        return false;
    }
    // condition track [frames, 2048] produced by the autoregressive stage
    std::vector<float> cond;
    int condFrames = 0;
    if (!p.condPath.empty()) {
        FILE* f = std::fopen(p.condPath.c_str(), "rb");
        if (!f) {
            err = "cannot open condition track " + p.condPath;
            return false;
        }
        std::fseek(f, 0, SEEK_END);
        long bytes = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        cond.resize((size_t)bytes / 4);
        if (std::fread(cond.data(), 4, cond.size(), f) != cond.size()) {
            std::fclose(f);
            err = "short condition track";
            return false;
        }
        std::fclose(f);
        condFrames = (int)(cond.size() / kCondWidth);
        std::printf("cond: %d frames\n", condFrames);
    } else {
        err = "no --cond track: the autoregressive (LM) stage is not ported yet";
        return false;
    }
    const int L = condFrames;
    if (L < 2) {
        err = "condition track too short";
        return false;
    }

    SaDit dit;
    if (!dit.load(p.ditPath, p.spvDir, p.ditLayers, err)) return false;

    std::mt19937 rng((uint32_t)p.seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t)kLatentChannels * L);
    for (auto& v : x) v = nd(rng);

    // Sigmas from the reference scheduler: 1 - linspace(1, 1/N, N), ascending, then 1.0.
    // sigma is the flow time (0 = noise), so Euler is x += (sigma_next - sigma) * v.
    std::vector<float> sig((size_t)p.steps + 1);
    for (int i = 0; i < p.steps; i++)
        sig[(size_t)i] = 1.0f - (1.0f - (float)i / (float)p.steps) ;
    for (int i = 0; i < p.steps; i++) sig[(size_t)i] = (float)i / (float)p.steps;
    sig[(size_t)p.steps] = 1.0f;

    std::vector<float> vc, vu, xc, xcu;
    std::vector<float> zeroCond((size_t)L * kCondWidth, 0.0f);
    for (int s = 0; s < p.steps; s++) {
        float t = sig[(size_t)s];
        float tn = sig[(size_t)s + 1];
        if (!dit.forward(x, L, t, cond, vc, err)) return false;
        if (p.cfg != 1.0f) {
            if (!dit.forward(x, L, t, zeroCond, vu, err)) return false;
            for (size_t i = 0; i < vc.size(); i++) vc[i] = vu[i] + (vc[i] - vu[i]) * p.cfg;
        }
        for (size_t i = 0; i < x.size(); i++) x[i] += (tn - t) * vc[i];
        if (p.verbose) {
            double rms = 0;
            for (float f : x) rms += (double)f * f;
            std::printf("step %d/%d  t %.4f -> %.4f  |x| rms %.4f\n", s + 1, p.steps, t, tn,
                        std::sqrt(rms / (double)x.size()));
        }
    }

    SaVae vae;
    if (!vae.load(p.vaePath, p.spvDir, err)) return false;
    std::vector<float> stereo;
    if (!vae.decode(x, L, stereo, err)) return false;
    Audio au;
    au.sampleRate = SaVae::kSampleRate;
    au.channels = 2;
    au.samples = std::move(stereo);
    if (!writeWav(p.outWav, au, 16, err)) return false;
    std::printf("wrote %s (%.2f s stereo %d Hz)\n", p.outWav.c_str(), au.seconds(), au.sampleRate);
    return true;
}

}  // namespace audio
}  // namespace ofc
