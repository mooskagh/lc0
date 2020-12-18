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

#include <vector>

#include "chess/bitboard.h"
#include "lc2/node/traits.h"

namespace lczero {
namespace lc2 {

struct Node {
  using NT = WdlNodeTraits;

  // Did eval for this node complete (true) or is still in progress (false).
  bool eval_completed = false;

  // Number of finished visits.
  NT::N n;

  // Eval of the node.
  NT::WDL q;

  // Moves from this position.
  MoveList edges;
  // Per-edge N value.
  std::vector<NT::N> n_edge;
  // Current value for outgoing edges.
  std::vector<NT::Q> q_edge;
  // Priors for outgoing edges.
  std::vector<NT::P> p_edge;

  size_t num_edges() const { return edges.size(); }
};

}  // namespace lc2
}  // namespace lczero