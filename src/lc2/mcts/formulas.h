/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors

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

#include <cmath>

#include "utils/fastmath.h"

namespace lc2 {

struct NodeValue {
  float wl;
  float d;
  float ml;
};

inline float ComputeFPUReduction(float q, float visited_policy,
                                 float fpu_value) {
  return q * std::sqrt(visited_policy) * fpu_value;
}

inline float ComputeQ(float wl, float d, float /* ml */, float draw_score) {
  return wl + draw_score * d;
}

inline float ComputeUFactor(size_t n, float init, float k, float base) {
  float cpuct = init + (k ? k * lczero::FastLog((n + base) / base) : 0.0f);
  return cpuct * std::sqrt(std::max(n - 1, 1ul));
}

inline float ComputeU(float u_factor, size_t n) { return u_factor / (1 + n); }

// inline PositionKey UpdatePositionKey(const PositionKey& /* previous_key */,
//                               const lczero::ChessBoard& /* previous_board */,
//                               lczero::Move /* move */,
//                               const lczero::ChessBoard& new_board) {
//   return PositionKey(new_board.Hash());
// }

}  // namespace lc2