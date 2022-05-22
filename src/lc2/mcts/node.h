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

#include <array>
#include <cstdint>
#include <string>

#include "chess/position.h"
#include "utils/floats.h"

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

    // bool tail_is_valid : 1;
    bool is_being_processed : 1;
  };

  uint32_t n;
  Flags flags;
  uint8_t num_edges;
  uint8_t num_filled_edges;
  uint8_t padding1[1];
  float q_wl;
  float q_d;
  float q_ml;

  static constexpr size_t kEdgesInHead = 4;
  std::array<float, kEdgesInHead + 1> edge_p;
  std::array<uint32_t, kEdgesInHead> edge_n;
  std::array<float, kEdgesInHead> edge_q_wl;
  std::array<float, kEdgesInHead> edge_q_d;
  std::array<float, kEdgesInHead> edge_q_ml;
  std::array<uint16_t, kEdgesInHead + 1> moves;
  uint8_t padding2[6];

  bool TailIsValid() const { return num_filled_edges > edge_n.size(); }
};

using NodeTail = std::string;

}  // namespace lc2