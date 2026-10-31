// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Segmented Fish
// sdvk - see LICENSE at the repository root. No warranty.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ofc {

using Shape = std::vector<size_t>;

size_t numelOf(const Shape& s);
std::string shapeStr(const Shape& s);

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

class Tensor {
public:
    Shape shape;
    std::vector<float> data;
    std::vector<float> grad;
    bool requiresGrad = false;
    std::function<void()> backward;
    std::string name;

    Tensor() = default;
    explicit Tensor(Shape s);

    size_t numel() const;
    size_t dim(size_t i) const;
    size_t rank() const;
    float& at(size_t i);
    const float& at(size_t i) const;
    Tensor& reshape(Shape s);
    void zeroGrad();
    void requireGrad();
    double sum() const;
    double mean() const;
    void addGradFrom(const Tensor& src);
};

TensorPtr tensor(Shape s, bool rg = false);
TensorPtr zeros(Shape s, bool rg = false);
TensorPtr ones(Shape s, bool rg = false);
TensorPtr constant(Shape s, float v, bool rg = false);
TensorPtr fromData(Shape s, std::vector<float> d, bool rg = false);
TensorPtr fromData(Shape s, const float* d, bool rg = false);
TensorPtr cloneTensor(const TensorPtr& t);

void seedRng(uint64_t s);
float randnSample();
TensorPtr randn(Shape s, float stddev = 1.0f, bool rg = false);
TensorPtr randnLike(const TensorPtr& t, float stddev);
TensorPtr xavier(Shape fanIn, Shape fanOut);
TensorPtr kaiming(Shape s);

void backward(const TensorPtr& loss);
void zeroAllGrads(const std::vector<TensorPtr>& params);
double gradNorm(const std::vector<TensorPtr>& params);
void clipGrads(const std::vector<TensorPtr>& params, double maxNorm);

}  // namespace ofc
