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

class NodeHashShard {
 public:
  NodeView GetNode(uint64_t hash);

 private:
  // TODO(crem) Replace unordered_map with flat_hash_map.
  std::unordered_map<uint64_t, LeanNode> nodes_;
  ArrayArena<LeanEdge, 65536> edges_;
};

class FatEdge;
class FatNode {
  FatNode* parent_;
  FatEdge* parent_edge_;
  float visited_policy_;
  LeanEdge* edges_;
  LeanNode* node_;
  uint64_t material_key_;
  uint64_t zobrist_hash_;
  uint64_t history_hash_;

  uint16_t best_edge_;  // Atomic_together
  uint16_t best_edge_remaining_n_;
};

class FatEdge {
  Move move_;
  atomic n_in_flight_;
  FatNode* node;
};

class FatTree {};

void AddNodesToCompute(Node* node, int count) {
  // Atomically:
  if (node->best_edge_remaining_n_ = 0) {
    ComputeNewBestEdge();
  }
  // node->best_child_remaining_n_ > 0
  if (best_edge_->node == nullptr) {
    // Atomically.
    FatNode* new_node = AllocateNewNode();
    new_node.prepare();
    best_edge_->node = new_node;
  }
  ++best_edge_->refcount;  // Check after that node still exists.

  int count_to_visit = min(count, node->best_child_remaining_n_);
  best_child_remaining_n_ -= count_to_visit;

  best_child_->n_in_flight_ += count_to_visit;
  AddNodesToCompute()
}

}  // namespace lczero