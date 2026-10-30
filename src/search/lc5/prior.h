/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

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

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#pragma once

#include <cstdint>

#include "utils/fp16_utils.h"

namespace lczero::lc5 {

// A prior stored as IEEE fp16. It keeps about three significant digits and
// orders non-negative values like float, with subnormals down to about 6e-8.
// Converts implicitly from and to float.
struct PackedPrior {
  uint16_t bits = 0;

  PackedPrior() = default;
  PackedPrior(float prior) : bits(FP32toFP16(prior)) {}
  operator float() const { return FP16toFP32(bits); }
};

}  // namespace lczero::lc5
