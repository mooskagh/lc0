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

#pragma once

#include <limits>

#include "lc2/chess/position-key.h"
#include "lc2/mcts/node.h"
#include "lc2/storage/storage.h"

namespace lc2 {

using NodeStorage = Storage<PositionKey, NodeHead, NodeTail>;

struct BatchStats {
  size_t nn_evals{};
  size_t collisions{};

  void Reset() { *this = BatchStats{}; }
};

class Batch {
 public:
  void EnqueuePosition(const lczero::ChessBoard& board, const PositionKey& key,
                       size_t visit_count,
                       size_t parent_idx = std::numeric_limits<size_t>::max());
  void Gather(NodeStorage* node_storage);
  size_t size() const { return positions_keys_.size(); }
  size_t fetched_size() const { return node_heads_.size(); }

 private:
  void FetchNodes(NodeStorage* node_storage, size_t begin_idx, size_t end_idx);
  void ProcessNodes(size_t begin_idx, size_t end_idx);
  void CommitNodes(NodeStorage* node_storage, size_t begin_idx, size_t end_idx);
  void ProcessSingleNode(size_t idx);
  void ForwardVisit(size_t parent_idx, const lczero::ChessBoard& parent_board,
                    const PositionKey& parent_key, lczero::Move move,
                    size_t visits);

  // Not sure whether having that as parallel (for performance reasons) arrays
  // worth it.
  std::vector<lczero::ChessBoard> boards_;
  std::vector<PositionKey> positions_keys_;
  std::vector<NodeHead> node_heads_;
  std::vector<UnpackedNode> unpacked_nodes_;
  std::vector<size_t> visit_counts_;
  std::vector<NodeStorage::Status> fetch_status_;
  std::vector<size_t> parent_idx_;
  // Batch for NN evaluation.
  std::vector<size_t> idx_to_eval_;
  BatchStats stats_;
};

}  // namespace lc2