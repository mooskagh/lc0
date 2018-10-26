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

#pragma once

#include <cstddef>
#include <vector>

#include "neural/network.h"

namespace lczero {

void ApplyBatchNormalization(const size_t batch_size, const size_t channels,
                             float* data, const float* means,
                             const float* stddivs,
                             const float* eltwise = nullptr);

struct ConvBlock {
  ConvBlock() = default;
  ConvBlock(const pblczero::Weights::ConvBlock&);

  // Invert the bn_stddivs elements of a ConvBlock.
  void InvertStddev();
  // Offset bn_means by biases of a ConvBlock.
  void OffsetMeans();
  // Return a vector of inverted bn_stddivs of a ConvBlock.
  std::vector<float> GetInvertedStddev() const;
  // Return a vector of bn_means offset by biases of a ConvBlock.
  std::vector<float> GetOffsetMeans() const;

  std::vector<float> weights;
  std::vector<float> biases;
  std::vector<float> bn_means;
  std::vector<float> bn_stddivs;
};

}  // namespace lczero
