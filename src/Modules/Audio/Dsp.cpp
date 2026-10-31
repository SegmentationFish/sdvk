// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
#include "Dsp.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

namespace ofc {
namespace audio {
namespace {

constexpr double kPi = 3.14159265358979323846;

double dbToLin(double db) { return std::pow(10.0, db / 20.0); }
float linToDb(double v) { return v <= 1e-7 ? -144.0f : (float)(20.0 * std::log10(v)); }

void normaliseRange(Audio& a) {
    for (auto& v : a.samples) v = std::min(1.0f, std::max(-1.0f, v));
}

uint32_t nextPow2(uint64_t n) {
    uint32_t p = 1;
    while ((uint64_t)p < n) p <<= 1;
    return p;
}

// Zeroth-order modified Bessel function of the first kind (series form, converges fast for the
// beta values used here).
double besselI0(double x) {
    double sum = 1.0, term = 1.0;
    const double q = x * x / 4.0;
    for (int k = 1; k < 40; k++) {
        term *= q / ((double)k * (double)k);
        sum += term;
        if (term < sum * 1e-16) break;
    }
    return sum;
}

// Kaiser window, x in [0,1]. beta 10 puts the sidelobes near -100 dB, which is what the resampler
// needs: a Blackman window stops at about -58 dB and showed up as -56 dB of out-of-band garbage.
double kaiser(double x, double beta) {
    const double t = 2.0 * x - 1.0;
    const double r = 1.0 - t * t;
    return besselI0(beta * std::sqrt(std::max(0.0, r))) / besselI0(beta);
}

double sinc(double x) {
    if (std::fabs(x) < 1e-9) return 1.0;
    return std::sin(kPi * x) / (kPi * x);
}

}  // namespace

void fftInPlace(std::vector<float>& re, std::vector<float>& im) {
    const uint32_t n = (uint32_t)re.size();
    if (n <= 1 || (n & (n - 1))) return;  // only powers of two
    // bit-reversal permutation
    for (uint32_t i = 1, j = 0; i < n; i++) {
        uint32_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (uint32_t len = 2; len <= n; len <<= 1) {
        const double ang = -2.0 * kPi / (double)len;
        const double wr = std::cos(ang), wi = std::sin(ang);
        for (uint32_t i = 0; i < n; i += len) {
            double cr = 1.0, ci = 0.0;
            for (uint32_t k = 0; k < len / 2; k++) {
                const uint32_t a = i + k, b = i + k + len / 2;
                const double xr = re[b] * cr - im[b] * ci;
                const double xi = re[b] * ci + im[b] * cr;
                re[b] = (float)(re[a] - xr);
                im[b] = (float)(im[a] - xi);
                re[a] = (float)(re[a] + xr);
                im[a] = (float)(im[a] + xi);
                const double nr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = nr;
            }
        }
    }
}

float peakDb(const Audio& a) {
    double m = 0;
    for (float v : a.samples) m = std::max(m, (double)std::fabs(v));
    return linToDb(m);
}

float rmsDb(const Audio& a) {
    if (a.samples.empty()) return -144.0f;
    double acc = 0;
    for (float v : a.samples) acc += (double)v * v;
    return linToDb(std::sqrt(acc / (double)a.samples.size()));
}

void applyGain(Audio& a, float db) {
    const float g = (float)dbToLin(db);
    for (auto& v : a.samples) v *= g;
    normaliseRange(a);
}

void normalizePeak(Audio& a, float targetDb) {
    const float p = peakDb(a);
    if (p <= -143.0f) return;
    applyGain(a, targetDb - p);
}

void normalizeRms(Audio& a, float targetDb) {
    const float r = rmsDb(a);
    if (r <= -143.0f) return;
    applyGain(a, targetDb - r);
}

void fade(Audio& a, double inSec, double outSec, int curve) {
    if (a.empty()) return;
    const uint64_t n = a.frames();
    const uint64_t inN = std::min<uint64_t>((uint64_t)std::llround(inSec * a.sampleRate), n);
    const uint64_t outN = std::min<uint64_t>((uint64_t)std::llround(outSec * a.sampleRate), n);
    auto shape = [&](double t) {
        t = std::min(1.0, std::max(0.0, t));
        switch (curve) {
            case 1:
                return std::pow(t, 0.5);  // equal-power-ish rise, the usual DAW fade
            case 2:
                return t * t * (3.0 - 2.0 * t);
            default:
                return t;
        }
    };
    for (uint64_t i = 0; i < inN; i++) {
        const float g = (float)shape((double)i / (double)std::max<uint64_t>(1, inN - 1));
        for (uint32_t c = 0; c < a.channels; c++) a.samples[(size_t)(i * a.channels + c)] *= g;
    }
    for (uint64_t i = 0; i < outN; i++) {
        const uint64_t idx = n - outN + i;
        const float g = (float)shape((double)(outN - 1 - i) / (double)std::max<uint64_t>(1, outN - 1));
        for (uint32_t c = 0; c < a.channels; c++) a.samples[(size_t)(idx * a.channels + c)] *= g;
    }
}

void trimSilence(Audio& a, float thresholdDb, double padSec) {
    if (a.empty()) return;
    const float thr = (float)dbToLin(thresholdDb);
    const uint64_t n = a.frames();
    uint64_t first = 0, last = n;
    for (; first < n; first++) {
        bool loud = false;
        for (uint32_t c = 0; c < a.channels; c++)
            if (std::fabs(a.samples[(size_t)(first * a.channels + c)]) > thr) loud = true;
        if (loud) break;
    }
    for (; last > first; last--) {
        bool loud = false;
        for (uint32_t c = 0; c < a.channels; c++)
            if (std::fabs(a.samples[(size_t)((last - 1) * a.channels + c)]) > thr) loud = true;
        if (loud) break;
    }
    const uint64_t pad = (uint64_t)std::llround(padSec * a.sampleRate);
    first = first > pad ? first - pad : 0;
    last = std::min(n, last + pad);
    if (last <= first) return;
    std::vector<float> keep(a.samples.begin() + (size_t)(first * a.channels),
                            a.samples.begin() + (size_t)(last * a.channels));
    a.samples.swap(keep);
}

bool resample(Audio& a, uint32_t newRate, std::string& err) {
    if (!newRate) {
        err = "resample: target rate is 0";
        return false;
    }
    if (a.empty()) {
        err = "resample: empty audio";
        return false;
    }
    if (newRate == a.sampleRate) return true;
    const double ratio = (double)newRate / (double)a.sampleRate;
    // Normalise the cutoff so the transition band sits inside Nyquist of the lower rate: a bare
    // min(1, ratio) cuts exactly at the new Nyquist and folds part of the transition band back in.
    const double cutoff = 0.96 * std::min(1.0, ratio);
    const int taps = 48;
    const double beta = 10.0;
    const uint64_t inN = a.frames();
    const uint64_t outN = (uint64_t)std::llround((double)inN * ratio);
    std::vector<float> out((size_t)(outN * a.channels), 0.0f);
    for (uint32_t c = 0; c < a.channels; c++) {
        for (uint64_t i = 0; i < outN; i++) {
            const double t = (double)i / ratio;  // position in input samples
            const int64_t base = (int64_t)std::floor(t);
            double acc = 0.0;
            for (int k = -taps; k <= taps; k++) {
                const int64_t idx = base + k;
                const double dist = t - (double)idx;
                // Kaiser peaks at dist = 0 and reaches zero at the tap limit. Scaling by cutoff
                // keeps DC gain at unity, so no per-sample renormalisation is needed (that would
                // modulate the amplitude with the fractional phase and add sidebands).
                const double u = std::min(1.0, std::max(0.0, 0.5 + 0.5 * dist / (double)taps));
                const double h = cutoff * sinc(cutoff * dist) * kaiser(u, beta);
                const int64_t s = std::min<int64_t>(std::max<int64_t>(idx, 0), (int64_t)inN - 1);
                acc += (double)a.samples[(size_t)((uint64_t)s * a.channels + c)] * h;
            }
            out[(size_t)(i * a.channels + c)] = (float)acc;
        }
    }
    a.samples.swap(out);
    a.sampleRate = newRate;
    return true;
}

Spectrogram spectrogram(const Audio& a, uint32_t fftSize, uint32_t hop) {
    Spectrogram s;
    if (a.empty()) return s;
    fftSize = std::max<uint32_t>(64, 1u << (uint32_t)std::llround(std::log2((double)fftSize)));
    hop = std::max<uint32_t>(1, hop);
    s.fftSize = fftSize;
    s.hop = hop;
    s.rate = a.sampleRate;
    s.bins = fftSize / 2 + 1;
    const uint64_t n = a.frames();
    s.frames = (uint32_t)(n / hop) + 1;
    s.db.assign((size_t)s.frames * s.bins, -144.0f);
    std::vector<float> win(fftSize);
    for (uint32_t i = 0; i < fftSize; i++)
        win[i] = (float)(0.5 - 0.5 * std::cos(2.0 * kPi * i / (double)(fftSize - 1)));
    std::vector<float> re(fftSize), im(fftSize);
    for (uint32_t f = 0; f < s.frames; f++) {
        const uint64_t start = (uint64_t)f * hop;
        for (uint32_t i = 0; i < fftSize; i++) {
            const uint64_t idx = start + i;
            double acc = 0;
            if (idx < n)
                for (uint32_t c = 0; c < a.channels; c++) acc += a.samples[(size_t)(idx * a.channels + c)];
            re[i] = (float)(acc / std::max<uint32_t>(1, a.channels)) * win[i];
            im[i] = 0.0f;
        }
        fftInPlace(re, im);
        for (uint32_t b = 0; b < s.bins; b++) {
            const double mag = std::sqrt((double)re[b] * re[b] + (double)im[b] * im[b]) * 2.0 / (double)fftSize;
            s.db[(size_t)f * s.bins + b] = linToDb(mag);
        }
    }
    s.minDb = -100.0f;
    s.maxDb = 0.0f;
    return s;
}

std::vector<float> spectrogramImage(const Spectrogram& s, uint32_t outW, uint32_t outH) {
    std::vector<float> img((size_t)3 * outW * outH, 0.0f);
    if (!s.frames || !s.bins || !outW || !outH) return img;
    for (uint32_t y = 0; y < outH; y++) {
        // top row = highest frequency, so the picture reads the way a DAW draws it
        const uint32_t bin = (uint32_t)((double)(outH - 1 - y) / (double)std::max(1u, outH - 1) * (s.bins - 1));
        for (uint32_t x = 0; x < outW; x++) {
            const uint32_t fr = (uint32_t)((double)x / (double)std::max(1u, outW - 1) * (s.frames - 1));
            float db = s.db[(size_t)fr * s.bins + bin];
            float t = (db - s.minDb) / std::max(1e-3f, s.maxDb - s.minDb);
            t = std::min(1.0f, std::max(0.0f, t));
            // mild gamma so quiet detail stays visible without blowing out the loud parts
            t = std::pow(t, 0.75f);
            const uint32_t o = y * outW + x;
            img[o] = t;
            img[(size_t)outW * outH + o] = t * 0.92f + 0.08f * std::sqrt(t);
            img[(size_t)2 * outW * outH + o] = std::min(1.0f, t * 1.1f);
        }
    }
    return img;
}

std::vector<float> makeReverbIr(uint32_t rate, uint32_t channels, float seconds, float decay, uint32_t seed) {
    channels = std::max<uint32_t>(1, channels);
    const uint32_t n = (uint32_t)std::max<double>(16.0, (double)seconds * rate);
    std::vector<float> ir((size_t)n * channels, 0.0f);
    std::mt19937 rng(seed ? seed : 1u);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    double energy = 0;
    for (uint32_t i = 0; i < n; i++) {
        const double t = (double)i / (double)n;
        // -60 dB at t = 1 scaled by decay: decay 1 = a natural rt60, higher = faster tail
        const double env = std::pow(10.0, -3.0 * std::pow(t, std::max(0.2f, decay)) );
        for (uint32_t c = 0; c < channels; c++) {
            const float v = (float)(uni(rng) * env);
            ir[(size_t)i * channels + c] = v;
            energy += (double)v * v;
        }
    }
    // Unit *energy*, not unit RMS: convolving with an IR whose sum of squares is 1 keeps the wet
    // signal at roughly the dry level. Normalising the RMS instead multiplied the output by
    // sqrt(taps) and clipped everything (a 0.8 s tail at 48 kHz is 38k taps, ~68x too hot).
    const double norm = energy > 1e-12 ? 1.0 / std::sqrt(energy) : 1.0;
    for (auto& v : ir) v = (float)(v * norm);
    return ir;
}

void reverb(Audio& a, const std::vector<float>& ir, float mix) {
    if (a.empty() || ir.empty()) return;
    const uint32_t ch = a.channels;
    if (!ch) return;
    const uint64_t n = a.frames();
    const uint64_t m = ir.size() / ch;  // IR must be interleaved with the same channel count
    if (!m) return;
    const uint32_t fftN = nextPow2(n + m - 1);
    std::vector<float> wet((size_t)n * ch, 0.0f);
    std::vector<float> re(fftN), im(fftN);
    for (uint32_t c = 0; c < ch; c++) {
        std::fill(re.begin(), re.end(), 0.0f);
        std::fill(im.begin(), im.end(), 0.0f);
        for (uint64_t i = 0; i < n; i++) re[(size_t)i] = a.samples[(size_t)(i * ch + c)];
        fftInPlace(re, im);
        std::vector<float> hr(fftN), hi(fftN);
        for (uint64_t i = 0; i < m; i++) hr[(size_t)i] = ir[(size_t)(i * ch + c)];
        fftInPlace(hr, hi);
        for (uint32_t i = 0; i < fftN; i++) {
            const double r = (double)re[i] * hr[i] - (double)im[i] * hi[i];
            const double v = (double)re[i] * hi[i] + (double)im[i] * hr[i];
            re[i] = (float)r;
            im[i] = (float)v;
        }
        for (uint32_t i = 0; i < fftN; i++) im[i] = -im[i];
        fftInPlace(re, im);
        for (uint64_t i = 0; i < n; i++) wet[(size_t)(i * ch + c)] = re[(size_t)i] / (float)fftN;
    }
    const float dry = 1.0f - std::min(1.0f, std::max(0.0f, mix));
    for (size_t i = 0; i < a.samples.size(); i++) a.samples[i] = a.samples[i] * dry + wet[i] * (1.0f - dry);
    normaliseRange(a);
}

bool applyOp(Audio& a, const std::string& op, const OpArgs& args, std::string& detail, std::string& err) {
    char buf[192];
    if (a.empty()) {
        err = "audio buffer is empty";
        return false;
    }
    if (op == "gain") {
        applyGain(a, (float)args.gainDb);
        std::snprintf(buf, sizeof buf, "gain %+.1f dB -> peak %.1f dBFS", args.gainDb, peakDb(a));
    } else if (op == "normalize-peak") {
        normalizePeak(a, (float)args.targetDb);
        std::snprintf(buf, sizeof buf, "peak %.1f dBFS", peakDb(a));
    } else if (op == "normalize-rms") {
        normalizeRms(a, (float)args.targetDb);
        std::snprintf(buf, sizeof buf, "rms %.1f dBFS, peak %.1f dBFS", rmsDb(a), peakDb(a));
    } else if (op == "fade") {
        fade(a, args.inSec, args.outSec, args.curve);
        std::snprintf(buf, sizeof buf, "fade in %.2fs / out %.2fs (curve %d)", args.inSec, args.outSec, args.curve);
    } else if (op == "trim") {
        const double before = a.seconds();
        trimSilence(a, (float)args.thresholdDb, args.padSec);
        std::snprintf(buf, sizeof buf, "trim %.2fs -> %.2fs", before, a.seconds());
    } else if (op == "resample") {
        const uint32_t from = a.sampleRate;
        if (!resample(a, args.rate, err)) return false;
        std::snprintf(buf, sizeof buf, "%u Hz -> %u Hz (%.2fs)", from, a.sampleRate, a.seconds());
    } else if (op == "reverb") {
        auto ir = makeReverbIr(a.sampleRate, a.channels, (float)args.seconds, (float)args.decay, args.seed);
        reverb(a, ir, (float)args.mix);
        std::snprintf(buf, sizeof buf, "reverb %.2fs, mix %.2f", args.seconds, args.mix);
    } else if (op == "mono") {
        const uint32_t from = a.channels;
        toMono(a);
        std::snprintf(buf, sizeof buf, "%u ch -> 1 ch", from);
    } else {
        err = "unknown audio op: " + op;
        return false;
    }
    detail = buf;
    return true;
}

}  // namespace audio
}  // namespace ofc
