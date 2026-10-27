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

namespace lczero::lc5 {

// A value sample from the perspective of the side to move at its node.
struct SearchValue {
  float q = 0.0f;
  float d = 0.0f;
  float m = 0.0f;
  SearchValue Parent() const { return {-q, d, m + 1.0f}; }
};

struct ValueStats {
  uint64_t visits = 0;
  double q_sum = 0.0;
  double d_sum = 0.0;
  double m_sum = 0.0;

  float Q() const { return Mean(q_sum); }
  float D() const { return Mean(d_sum); }
  float M() const { return Mean(m_sum); }
  void Add(const SearchValue& value) {
    ++visits;
    q_sum += value.q;
    d_sum += value.d;
    m_sum += value.m;
  }

 private:
  float Mean(double sum) const {
    return visits ? static_cast<float>(sum / visits) : 0.0f;
  }
};

}  // namespace lczero::lc5
