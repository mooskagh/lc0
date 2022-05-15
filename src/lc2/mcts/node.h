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

    bool tail_is_valid : 1;
    bool is_being_processed : 1;
  };

  uint32_t n;
  Flags flags;
  uint8_t num_edges;
  lczero::SigmoidFloat16 q_wl;
  lczero::ProbFloat16 q_d;
  lczero::MLFloat16 q_ml;

  static constexpr size_t kEdgesInHead = 4;
  std::array<lczero::ProbFloat16, kEdgesInHead + 1> edge_p;
  std::array<uint16_t, kEdgesInHead + 1> moves;
  std::array<uint32_t, kEdgesInHead> edge_n;
  std::array<lczero::SigmoidFloat16, kEdgesInHead> edge_q;
};

using NodeTail = std::string;

struct UnpackedNode {
  std::vector<float> p;
  std::vector<uint16_t> moves;
  std::vector<uint32_t> n;
  std::vector<float> q;

  void UnpackFromHead(const NodeHead& head);
  void UnpackFromHeadAndTail(const NodeHead& head, const NodeTail& tail);
  void UpdateNIntoHead(NodeHead* head);
  void UpdateNIntoHeadAndTail(NodeHead* head, NodeTail* tail);
};

inline float GetNodeQ(const NodeHead& head) {
  return static_cast<float>(head.q_wl);
}

}  // namespace lc2