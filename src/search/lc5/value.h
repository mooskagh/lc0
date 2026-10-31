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

#include <algorithm>
#include <cstdint>

namespace lczero::lc5 {

// A value sample from the perspective of the side to move at its node.
struct SearchValue {
  float q = 0.0f;
  float d = 0.0f;
  float m = 0.0f;
  SearchValue Parent() const { return {-q, d, m + 1.0f}; }
};

// Game-theoretic bounds on a value from the side to move: -1 loss, 0 draw,
// 1 win. Proven when lower == upper; `plies` is then the distance to the end
// of the game.
struct Bounds {
  int8_t lower = -1;
  int8_t upper = 1;
  uint8_t plies = 0;  // Saturates at 255.

  bool Proven() const { return lower == upper; }
  // The bounds one ply up, for the other side; plies count only when proven.
  Bounds Parent() const {
    return {static_cast<int8_t>(-upper), static_cast<int8_t>(-lower),
            static_cast<uint8_t>(Proven() ? std::min(255, plies + 1) : 0)};
  }
  // The proven bounds of an exact value, such as a checkmate or a draw.
  static Bounds Exact(float q, float m) {
    const int8_t result = q > 0.5f ? 1 : q < -0.5f ? -1 : 0;
    return {result, result, static_cast<uint8_t>(std::clamp(m, 0.0f, 255.0f))};
  }
  bool operator==(const Bounds&) const = default;
};

// Running means, as in classic: each Add moves the mean towards the sample.
struct ValueStats {
  uint32_t visits = 0;
  float q = 0.0f;
  float d = 0.0f;
  float m = 0.0f;
  // Maintained by the policy from the messages; Add leaves them alone.
  Bounds bounds{};

  float Q() const { return q; }
  float D() const { return d; }
  float M() const { return m; }
  void Add(const SearchValue& value) {
    ++visits;
    const float n = static_cast<float>(visits);
    q += (value.q - q) / n;
    d += (value.d - d) / n;
    m += (value.m - m) / n;
  }
};

}  // namespace lczero::lc5
