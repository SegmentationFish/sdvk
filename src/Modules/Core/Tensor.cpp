// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 SegmentationFish
// sdvk - see LICENSE at the repository root. No warranty.
#include "Tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>

namespace ofc {

size_t numelOf(const Shape& s) {
    size_t n = 1;
    for (auto v : s) n *= v;
    return n;
}

std::string shapeStr(const Shape& s) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < s.size(); i++) {
        if (i) os << ",";
        os << s[i];
    }
    os << "]";
    return os.str();
}

Tensor::Tensor(Shape s) : shape(std::move(s)) { data.assign(numelOf(shape), 0.0f); }

size_t Tensor::numel() const { return data.size(); }
size_t Tensor::dim(size_t i) const { return shape[i]; }
size_t Tensor::rank() const { return shape.size(); }

float& Tensor::at(size_t i) { return data[i]; }
const float& Tensor::at(size_t i) const { return data[i]; }

Tensor& Tensor::reshape(Shape s) {
    if (numelOf(s) != data.size()) throw std::runtime_error("reshape size mismatch");
    shape = std::move(s);
    return *this;
}

void Tensor::zeroGrad() { grad.assign(data.size(), 0.0f); }

void Tensor::requireGrad() {
    requiresGrad = true;
    grad.assign(data.size(), 0.0f);
}

double Tensor::sum() const {
    double s = 0;
    for (float v : data) s += v;
    return s;
}

double Tensor::mean() const {
    if (data.empty()) return 0.0;
    return sum() / (double)data.size();
}

void Tensor::addGradFrom(const Tensor& src) {
    if (grad.size() != src.grad.size()) grad.resize(src.grad.size(), 0.0f);
    for (size_t i = 0; i < src.grad.size(); i++) grad[i] += src.grad[i];
}

static TensorPtr make(Shape s, bool rg) {
    auto t = std::make_shared<Tensor>(std::move(s));
    t->requiresGrad = rg;
    if (rg) t->grad.assign(numelOf(t->shape), 0.0f);
    return t;
}

TensorPtr tensor(Shape s, bool rg) { return make(std::move(s), rg); }
TensorPtr zeros(Shape s, bool rg) { return make(std::move(s), rg); }

TensorPtr ones(Shape s, bool rg) {
    auto t = make(std::move(s), rg);
    std::fill(t->data.begin(), t->data.end(), 1.0f);
    return t;
}

TensorPtr constant(Shape s, float v, bool rg) {
    auto t = make(std::move(s), rg);
    std::fill(t->data.begin(), t->data.end(), v);
    return t;
}

TensorPtr fromData(Shape s, std::vector<float> d, bool rg) {
    if (d.size() != numelOf(s)) throw std::runtime_error("fromData size mismatch " + shapeStr(s));
    auto t = make(s, rg);
    t->data = std::move(d);
    return t;
}

TensorPtr fromData(Shape s, const float* d, bool rg) {
    return fromData(std::move(s), std::vector<float>(d, d + numelOf(s)), rg);
}

TensorPtr cloneTensor(const TensorPtr& t) { return fromData(t->shape, t->data, t->requiresGrad); }

static std::mt19937_64 g_rng(1234567);

void seedRng(uint64_t s) { g_rng.seed(s); }

float randnSample() {
    static std::normal_distribution<float> dist(0.0f, 1.0f);
    return dist(g_rng);
}

TensorPtr randn(Shape s, float stddev, bool rg) {
    auto t = make(std::move(s), rg);
    for (auto& v : t->data) v = randnSample() * stddev;
    return t;
}

TensorPtr randnLike(const TensorPtr& t, float stddev) { return randn(t->shape, stddev, t->requiresGrad); }

TensorPtr kaiming(Shape s) {
    float fan = 1.0f;
    if (s.size() >= 2) fan = (float)(s[s.size() - 2] * s[s.size() - 1]);
    return randn(s, std::sqrt(2.0f / fan));
}

TensorPtr xavier(Shape fanIn, Shape fanOut) {
    float a = std::sqrt(6.0f / ((float)numelOf(fanIn) + (float)numelOf(fanOut)));
    auto t = randn({fanIn[0], fanOut[0]});
    for (auto& v : t->data) v *= a;
    return t;
}

void zeroAllGrads(const std::vector<TensorPtr>& params) {
    for (auto& p : params)
        if (p) p->grad.assign(p->data.size(), 0.0f);
}

double gradNorm(const std::vector<TensorPtr>& params) {
    double s = 0;
    for (auto& p : params) {
        if (!p) continue;
        for (float v : p->grad) s += (double)v * v;
    }
    return std::sqrt(s);
}

void clipGrads(const std::vector<TensorPtr>& params, double maxNorm) {
    if (maxNorm <= 0) return;
    double n = gradNorm(params);
    if (n <= maxNorm || n == 0) return;
    double scale = maxNorm / n;
    for (auto& p : params) {
        if (!p) continue;
        for (float& v : p->grad) v = (float)(v * scale);
    }
}

}  // namespace ofc
