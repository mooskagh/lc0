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

#include <cstdint>
#include <string>

#include "chess/position.h"

#pragma once

namespace lc2 {

struct NodeHead {
  enum class Terminal : uint8_t { NonTerminal, EndOfGame, Tablebase, TwoFold };
  struct Flags {
    // Bit fields using parts of uint8_t fields initialized in the constructor.
    // Whether or not this node end game (with a winning of either sides or
    // draw).
    Terminal terminal_type_ : 2;
    // Best and worst result for this node.
    lczero::GameResult lower_bound_ : 2;
    lczero::GameResult upper_bound_ : 2;

    bool tail_is_valid : 1;
    bool unused : 1;
  };

  double q;
  float d;
  float moves_left;
  Flags flags;
  uint8_t num_edges;

  uint32_t num_parents;

  static constexpr size_t kEdgesInHead = 3;
  uint16_t edge_p[kEdgesInHead + 1];
  uint16_t moves[kEdgesInHead];
  float edge_s[kEdgesInHead];
  uint32_t edge_n[kEdgesInHead];
};
template<int X> struct Debug;

static_assert(sizeof(NodeHead) == 64, "Unexpected size of NodeHead");

using NodeTail = std::string;

}  // namespace lc2