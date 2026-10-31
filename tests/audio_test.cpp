// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
// Host-side checks for the audio module. These need no GPU and no model, so they run on CI and
// are the objective evidence that the WAV/DSP layer is correct:
//   - PCM 16/24 and float 32 roundtrip through the reader and the writer
//   - 48k -> 44.1k resampling keeps total harmonic distortion under -80 dB on a pure sine
//   - FFT convolution agrees with direct convolution to better than -90 dB
//   - fade / normalise / trim / spectrogram land on their expected values
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "Modules/Audio/Dsp.hpp"
#include "Modules/Audio/Wav.hpp"

using namespace ofc::audio;

static int gFail = 0;
static void check(bool ok, const std::string& what, const std::string& detail = std::string()) {
    std::printf("  %-4s %-52s %s\n", ok ? "PASS" : "FAIL", what.c_str(), detail.c_str());
    if (!ok) gFail++;
}

static Audio sine(uint32_t rate, uint32_t channels, double freq, double seconds, double amp) {
    Audio a;
    a.sampleRate = rate;
    a.channels = channels;
    const uint64_t n = (uint64_t)std::llround(seconds * rate);
    a.samples.resize((size_t)(n * channels));
    for (uint64_t i = 0; i < n; i++) {
        const float v = (float)(amp * std::sin(2.0 * 3.14159265358979323846 * freq * (double)i / (double)rate));
        for (uint32_t c = 0; c < channels; c++) a.samples[(size_t)(i * channels + c)] = v;
    }
    return a;
}

int main() {
    const std::string tmp = "audio_test_tmp.wav";
    std::printf("audio module checks\n");

    // ---------------------------------------------------------------- wav roundtrip
    {
        Audio a = sine(44100, 2, 997.0, 0.25, 0.8);
        for (int bits : {16, 24, 32}) {
            std::string err;
            if (!writeWav(tmp, a, bits, err)) {
                check(false, "writeWav " + std::to_string(bits) + " bit", err);
                continue;
            }
            Audio b;
            if (!readWav(tmp, b, err)) {
                check(false, "readWav " + std::to_string(bits) + " bit", err);
                continue;
            }
            const bool metaOk = b.sampleRate == a.sampleRate && b.channels == a.channels &&
                                b.samples.size() == a.samples.size();
            double maxErr = 0;
            for (size_t i = 0; i < std::min(a.samples.size(), b.samples.size()); i++)
                maxErr = std::max(maxErr, (double)std::fabs(a.samples[i] - b.samples[i]));
            // 16 bit quantises to one LSB, 24 bit to 2^-23; float 32 is exact for these values
            const double tol = bits == 32 ? 0.0 : (bits == 24 ? 2.0 / 8388607.0 : 1.5 / 32767.0);
            char d[128];
            std::snprintf(d, sizeof d, "max err %.3e (tol %.3e), %u Hz %u ch", maxErr, tol, b.sampleRate, b.channels);
            check(metaOk && maxErr <= tol, "wav roundtrip " + std::to_string(bits) + " bit", d);
        }
    }

    // ---------------------------------------------------------------- resample distortion
    {
        Audio a = sine(48000, 1, 1000.0, 0.5, 0.5);
        std::string err;
        const bool ok = resample(a, 44100, err);
        check(ok && a.sampleRate == 44100 && !a.samples.empty(), "resample 48000 -> 44100",
              std::to_string(a.frames()) + " frames " + err);
        // Least-squares fit of the fundamental, then measure what is left over. An FFT would work
        // too, but a 1 kHz tone is not periodic in any convenient window length and the analysis
        // window's own skirt reads as -56 dB of "distortion" that the resampler never produced.
        const size_t from = a.samples.size() / 4, len = 16384;
        const double w = 2.0 * 3.14159265358979323846 * 1000.0 / 44100.0;
        double sss = 0, scc = 0, ssc = 0, sy1 = 0, sy2 = 0, syTot = 0, s1 = 0, s2 = 0, s0 = 0;
        for (size_t i = 0; i < len; i++) {
            const double t = (double)(from + i);
            const double y = a.samples[from + i];
            const double c1 = std::sin(w * t), c2 = std::cos(w * t);
            sss += c1 * c1;
            scc += c2 * c2;
            ssc += c1 * c2;
            s1 += c1;
            s2 += c2;
            s0 += 1.0;
            sy1 += y * c1;
            sy2 += y * c2;
            syTot += y;
        }
        // solve the 3x3 normal equations for [A, B, D] over (sin, cos, 1)
        const double m[3][3] = {{sss, ssc, s1}, {ssc, scc, s2}, {s1, s2, s0}};
        const double v[3] = {sy1, sy2, syTot};
        const double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                           m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        double cof[3];
        for (int i = 0; i < 3; i++) {
            double t[3][3];
            std::memcpy(t, m, sizeof t);
            for (int r = 0; r < 3; r++) t[r][i] = v[r];
            cof[i] = (t[0][0] * (t[1][1] * t[2][2] - t[1][2] * t[2][1]) -
                      t[0][1] * (t[1][0] * t[2][2] - t[1][2] * t[2][0]) +
                      t[0][2] * (t[1][0] * t[2][1] - t[1][1] * t[2][0])) /
                     det;
        }
        double res = 0, fund = 0;
        for (size_t i = 0; i < len; i++) {
            const double t = (double)(from + i);
            const double fit = cof[0] * std::sin(w * t) + cof[1] * std::cos(w * t) + cof[2];
            const double d = a.samples[from + i] - fit;
            res += d * d;
            fund += fit * fit;
        }
        const double thdDb = 10.0 * std::log10(std::max(1e-30, res) / std::max(1e-30, fund));
        char d[192];
        std::snprintf(d, sizeof d, "fundamental %.3f, residual %.1f dB, %u frames@44100", std::sqrt(fund / len),
                      thdDb, (unsigned)a.frames());
        check(ok && thdDb < -80.0, "resample distortion < -80 dB", d);
    }

    // ---------------------------------------------------------------- convolution vs direct
    {
        // keep the wet signal inside [-1,1] so the mix stage's clamp never kicks in: this test is
        // about the convolution maths, not about headroom.
        Audio a = sine(48000, 1, 440.0, 1024.0 / 48000.0, 0.15);
        const auto ir = makeReverbIr(48000, 1, 512.0f / 48000.0f, 1.0f, 7u);
        const size_t n = a.frames(), m = ir.size();
        std::vector<float> direct(n, 0.0f);
        for (size_t i = 0; i < n; i++) {
            double acc = 0;
            for (size_t k = 0; k < m && k <= i; k++) acc += (double)a.samples[i - k] * (double)ir[k];
            direct[i] = (float)acc;
        }
        Audio wet = a;
        reverb(wet, ir, 1.0f);
        double maxErr = 0, peak = 0;
        for (size_t i = 0; i < n; i++) {
            maxErr = std::max(maxErr, (double)std::fabs(wet.samples[i] - direct[i]));
            peak = std::max(peak, (double)std::fabs(direct[i]));
        }
        const double relDb = 20.0 * std::log10(std::max(1e-30, maxErr) / std::max(1e-30, peak));
        char d[128];
        std::snprintf(d, sizeof d, "max err %.3e of peak %.3f = %.1f dB", maxErr, peak, relDb);
        check(relDb < -90.0, "fft convolution matches direct < -90 dB", d);
    }

    // ---------------------------------------------------------------- envelope helpers
    {
        Audio a = sine(44100, 1, 100.0, 0.5, 1.0);
        fade(a, 0.1, 0.1, 1);
        const uint64_t n = a.frames();
        const bool ends = std::fabs(a.samples[0]) < 1e-6f && std::fabs(a.samples[n - 1]) < 1e-6f;
        check(ends, "fade starts and ends at zero");

        Audio b = sine(44100, 1, 100.0, 0.2, 0.25);
        normalizePeak(b, -1.0f);
        const double pk = peakDb(b);
        char d[96];
        std::snprintf(d, sizeof d, "peak %.3f dBFS", pk);
        check(std::fabs(pk + 1.0f) < 0.1f, "normalize-peak hits -1 dBFS", d);

        Audio c = sine(44100, 1, 100.0, 0.2, 0.5);
        Audio padded;
        padded.sampleRate = c.sampleRate;
        padded.channels = 1;
        padded.samples.assign((size_t)(4410 * 3), 0.0f);
        for (size_t i = 0; i < c.samples.size(); i++) padded.samples[(size_t)4410 + i] = c.samples[i];
        const double before = padded.seconds();
        trimSilence(padded, -50.0f, 0.0);
        char d2[96];
        std::snprintf(d2, sizeof d2, "%.3fs -> %.3fs", before, padded.seconds());
        check(std::fabs(padded.seconds() - c.seconds()) < 0.01, "trim removes the silent head and tail", d2);
    }

    // ---------------------------------------------------------------- spectrogram
    {
        Audio a = sine(48000, 1, 1000.0, 0.5, 0.7);
        const Spectrogram s = spectrogram(a, 2048, 512);
        uint32_t peakBin = 0;
        float peakDb = -200.0f;
        const uint32_t mid = s.frames / 2;
        for (uint32_t b = 0; b < s.bins; b++) {
            const float v = s.db[(size_t)mid * s.bins + b];
            if (v > peakDb) {
                peakDb = v;
                peakBin = b;
            }
        }
        const double freq = (double)peakBin * a.sampleRate / (double)s.fftSize;
        char d[160];
        std::snprintf(d, sizeof d, "%u frames x %u bins, peak bin %u = %.1f Hz (%.1f dB)", s.frames, s.bins, peakBin,
                      freq, peakDb);
        check(s.frames > 0 && s.bins == 1025 && std::fabs(freq - 1000.0) < 30.0, "spectrogram peak lands on 1 kHz", d);
        auto img = spectrogramImage(s, 256, 128);
        double sum = 0;
        for (float v : img) sum += v;
        check(img.size() == (size_t)3 * 256 * 128 && sum > 0.0, "spectrogram image renders",
              std::to_string(img.size()) + " floats");
    }

    // ---------------------------------------------------------------- op chain (what the node graph sends)
    {
        Audio a = sine(44100, 2, 220.0, 0.3, 0.4);
        OpArgs args;
        std::string detail, err;
        bool ok = true;
        ok = ok && applyOp(a, "normalize-peak", args, detail, err);
        ok = ok && applyOp(a, "fade", args, detail, err);
        args.rate = 48000;
        ok = ok && applyOp(a, "resample", args, detail, err);
        args.seconds = 0.3;
        args.mix = 0.2;
        ok = ok && applyOp(a, "reverb", args, detail, err);
        ok = ok && applyOp(a, "mono", args, detail, err);
        check(ok && a.channels == 1 && a.sampleRate == 48000 && !applyOp(a, "nope", args, detail, err),
              "op chain runs and rejects unknown ops", err);
    }

    std::remove(tmp.c_str());
    std::printf("%s: %d failed\n", gFail ? "FAILED" : "OK", gFail);
    return gFail ? 1 : 0;
}
