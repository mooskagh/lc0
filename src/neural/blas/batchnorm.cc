/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2018 The LCZero Authors

 Leela Chess is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 Leela Chess is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "neural/blas/batchnorm.h"

#include "utils/weights_adapter.h"

#include <algorithm>
#include <cmath>

namespace lczero {

namespace {
static constexpr float kEpsilon = 1e-5f;
constexpr int kWidth = 8;
constexpr int kHeight = 8;
constexpr int kSquares = kWidth * kHeight;
}  // namespace

void InvertVector(std::vector<float>* vec) {
  for (auto& x : *vec) x = 1.0f / std::sqrt(x + kEpsilon);
}

void OffsetVector(std::vector<float>* means, const std::vector<float>& biases) {
  std::transform(means->begin(), means->end(), biases.begin(), means->begin(),
                 std::minus<float>());
}

void ApplyBatchNormalization(const size_t batch_size, const size_t channels,
                             float* data, const float* means,
                             const float* stddivs, const float* eltwise) {
  for (size_t i = 0; i < batch_size; i++) {
    for (size_t c = 0; c < channels; ++c) {
      auto mean = means[c];
      auto scale_stddiv = stddivs[c];

      if (eltwise == nullptr) {
        // Classical BN
        auto arr = &data[c * kSquares];
        for (size_t b = 0; b < kSquares; b++) {
          float val = scale_stddiv * (arr[b] - mean);
          arr[b] = val > 0 ? val : 0;
        }
      } else {
        // BN + residual add
        auto arr = &data[c * kSquares];
        auto res = &eltwise[c * kSquares];
        for (size_t b = 0; b < kSquares; b++) {
          float val = res[b] + (scale_stddiv * (arr[b] - mean));
          arr[b] = val > 0 ? val : 0;
        }
      }
    }
    data += channels * kSquares;
    if (eltwise != nullptr) eltwise += channels * kSquares;
  }
}

ConvBlock::ConvBlock(const pblczero::Weights::ConvBlock& block)
    : weights(LayerAdapter(block.weights()).as_vector()),
      biases(LayerAdapter(block.biases()).as_vector()),
      bn_means(LayerAdapter(block.bn_means()).as_vector()),
      bn_stddivs(LayerAdapter(block.bn_stddivs()).as_vector()) {}

void ConvBlock::InvertStddev() { InvertVector(&bn_stddivs); }

void ConvBlock::OffsetMeans() { OffsetVector(&bn_means, biases); }

std::vector<float> ConvBlock::GetInvertedStddev() const {
  std::vector<float> stddivs = bn_stddivs;  // copy
  InvertVector(&stddivs);
  return stddivs;
}

std::vector<float> ConvBlock::GetOffsetMeans() const {
  std::vector<float> means = bn_means;  // copy
  OffsetVector(&means, biases);
  return means;
}

}  // namespace lczero
