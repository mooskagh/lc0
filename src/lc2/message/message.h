/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors

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
#include <iostream>
#include <memory>
#include <string>

#include "chess/position.h"
#include "lc2/node/traits.h"

namespace lczero {
namespace lc2 {

struct Message {
  using NT = WdlNodeTraits;

  enum class Class {
    kNode = 0x01,
    kRoot = 0x02,
    kEval = 0x03,
  };

  enum Type {
    kUnknown = 0,

    // Universal message types.
    kAuxAbort = 0x0001,

    // Node-specific message types.
    kNodeGather = 0x0101,
    kNodeBlacklist = 0x0102,
    kNodeBackProp = 0x0103,

    // Root-specific message types.
    kRootInitial = 0x0201,  // Initial nodes injection when search starts.
    kRootCollision = 0x0202,
    kRootEvalSkipped = 0x0203,
    kRootOutOfOrderEvalReady = 0x0204,
    kRootBlacklistDone = 0x0205,
    kRootBackPropDone = 0x0206,

    // Eval-specific types.
    kEvalEval = 0x0301,
    kEvalSkip = 0x0302
  };

  // Message type.
  Type type = kUnknown;

  // Instead of sending multiple identical messages, it's better to send just
  // one and set number of replicas.
  // TODO(crem) Come up with better name for this variable.
  uint16_t arity = 0;

  // From which epoch the message was sent. Is used to track stale messages.
  uint32_t epoch = 0;

  // Number of gathering attempt, zero-based.
  // uint16_t attempt = 0;

  // List of positions from start of the game.
  // TODO(crem) Make it from the position root instead.
  // Warning: this surely won't be visible in profiler for first moves, but may
  // be a problem later in the game.
  PositionHistory position_history;

  // Indices of moves from the current game head to the leaf.
  std::vector<uint8_t> move_idx;

  // TODO(crem) Add comment here. DO NOT SUBMIT
  bool node_height_is_odd;

  // TODO(crem) Add comment here. DO NOT SUBMIT
  NT::Q child_q;

  // Eval result.
  struct Eval {
    // Eval of the node.
    NT::WDL wdl;
    // Moves from this position.
    MoveList edges;
    // Priors for outgoing edges.
    std::vector<NT::P> p_edge;
  };
  std::optional<Eval> eval_result;

  std::unique_ptr<Message> SplitOff(int how_much);
};

std::string MessageTypeToString(Message::Type);
std::ostream& operator<<(std::ostream& os, const Message& t);

}  // namespace lc2
}  // namespace lczero