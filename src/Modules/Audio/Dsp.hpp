// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
// Host-side DSP for the audio workflow nodes: gain/normalise, fades, silence trim, windowed-sinc
// resampling, FFT convolution reverb and an STFT spectrogram. Pure CPU, no dependencies, so every
// one of these is unit-testable without touching the GPU.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "Wav.hpp"

namespace ofc {
namespace audio {

// Radix-2 in-place complex FFT; both vectors must be the same power-of-two length.
void fftInPlace(std::vector<float>& re, std::vector<float>& im);

float peakDb(const Audio& a);  // max |sample| in dBFS, -inf becomes -144
float rmsDb(const Audio& a);
void applyGain(Audio& a, float db);              // clamped so nothing leaves [-1,1]
void normalizePeak(Audio& a, float targetDb);    // e.g. -1 dBFS
void normalizeRms(Audio& a, float targetDb);     // loudness-ish, e.g. -18 dBFS
// curve: 0 linear, 1 logarithmic (slow start, matches most DAWs), 2 smoothstep
void fade(Audio& a, double inSec, double outSec, int curve);
void trimSilence(Audio& a, float thresholdDb, double padSec);

// Windowed-sinc (Blackman) resampler, 32 taps per side. Cutoff follows the lower of the two rates
// so downsampling does not alias. Returns false on a zero/absurd rate.
bool resample(Audio& a, uint32_t newRate, std::string& err);

struct Spectrogram {
    uint32_t bins = 0;    // fftSize/2 + 1
    uint32_t frames = 0;
    uint32_t fftSize = 0;
    uint32_t hop = 0;
    uint32_t rate = 0;
    std::vector<float> db;  // [frame*bins + bin], dBFS
    float minDb = -100.0f;
    float maxDb = 0.0f;
};
Spectrogram spectrogram(const Audio& a, uint32_t fftSize, uint32_t hop);

// Maps the spectrogram onto a CHW RGB image in [0,1] (mono ramp + a blue-ish top end so the
// shapes read at a glance), ready for writePngFile.
std::vector<float> spectrogramImage(const Spectrogram& s, uint32_t outW, uint32_t outH);

// Synthetic reverb impulse: exponentially decaying noise, per-channel decorrelated, normalised so
// the wet signal keeps roughly the dry level at mix = 1.
std::vector<float> makeReverbIr(uint32_t rate, uint32_t channels, float seconds, float decay, uint32_t seed);
// Wet/dry FFT convolution. mix 0 = dry, 1 = fully wet.
void reverb(Audio& a, const std::vector<float>& ir, float mix);

// One node in the audio graph. The HTTP layer fills OpArgs straight from the request body, so a
// new knob only needs a field here plus a case in applyOp.
struct OpArgs {
    double gainDb = 0.0;
    double targetDb = -1.0;   // peak / rms normalise target
    double inSec = 0.0;       // fade in
    double outSec = 0.0;      // fade out
    double padSec = 0.05;     // trim: silence kept around the cut
    double thresholdDb = -50.0;
    double seconds = 1.2;     // reverb tail
    double mix = 0.3;
    double decay = 1.0;
    int curve = 1;
    uint32_t rate = 48000;
    uint32_t fftSize = 2048;
    uint32_t hop = 512;
    uint32_t seed = 1;
    int bits = 24;
};

// op: gain | normalize-peak | normalize-rms | fade | trim | resample | reverb | mono
// Returns false with err set when the op is unknown or the buffer cannot satisfy it.
bool applyOp(Audio& a, const std::string& op, const OpArgs& args, std::string& detail, std::string& err);

}  // namespace audio
}  // namespace ofc
