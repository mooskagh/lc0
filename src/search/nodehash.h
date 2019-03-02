/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2019 The LCZero Authors

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

#include "chess/board.h"
#include "utils/array-arena.h"

namespace lczero {

class LeanEdge {
 private:
  float p_;
  float q_;  // Q = W - L + (aggressiveness * D).
  uint32_t n_;
  // TODO(crem) Move is not really needed as it can be regenerated, try removing
  // it later.
  Move move_;
};

static_assert(sizeof(LeanEdge) == 16, "Unexpected size of LeanEdge");

class LeanNode {
 private:
  atomic_flag is_used_;
  uint8_t num_moves_;
  float z_;  // Z = W - L
  float d_;
  uint64_t edge_index_;
};

static_assert(sizeof(LeanNode) == 24, "Unexpected size of LeanNode");

class NodeView {};

class NodeHashShard {
 public:
  NodeView GetNode(uint64_t hash);

 private:
  // TODO(crem) Replace unordered_map with flat_hash_map.
  std::unordered_map<uint64_t, LeanNode> nodes_;
  ArrayArena<LeanEdge, 65536> edges_;
};

}  // namespace lczero