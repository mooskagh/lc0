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

#include <functional>
#include <limits>
#include <vector>

#include "chess/board.h"
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
                       size_t visit_limit,
                       size_t parent_idx = std::numeric_limits<size_t>::max());
  void Gather(NodeStorage* node_storage);
  size_t queue_size() const { return queue_.position_keys.size(); }
  size_t fetched_size() const { return nodes_.heads.size(); }
  size_t leaf_count() const { return leafs_.leaf_indices.size(); }

  struct BackPropData {
    float wl;
    float d;
    float ml;
    uint32_t backproping_visits;
    uint32_t visits_to_rollback;
    // is_termninal blah-blah
  };
  using LeafVisitFunc =
      std::function<void(const lczero::ChessBoard&, std::vector<float>* p,
                         std::vector<lczero::Move>* moves, BackPropData* data)>;
  void VisitLeafs(LeafVisitFunc);
  // TODO NOTE ABOUT WHERE I WRITE
  // Visit three times:
  // 1. To expand leafs with movegen
  // 2. To encode positions for NN
  // 3. To populate NN results back.

 private:
  void FetchNodes(NodeStorage* node_storage, size_t begin_idx, size_t end_idx);
  void ProcessNodes(size_t begin_idx, size_t end_idx);
  void CommitNodes(NodeStorage* node_storage, size_t begin_idx, size_t end_idx);
  void ProcessSingleNode(size_t idx);
  void ForwardVisit(size_t parent_idx, const lczero::ChessBoard& parent_board,
                    const PositionKey& parent_key, lczero::Move move,
                    size_t visits);

  void UnpackEdgesFromHead(const NodeHead& head);
  void UnpackEdgesFromHeadAndTail(const NodeHead& head, const NodeTail& tail);
  void UnpackEdgesFromHeadAndBuffer(const NodeHead& head,
                                    std::string_view* tail_buffer,
                                    size_t to_fetch, size_t to_pad);

  // All vectors in FetchQueue must be of the same length. An element is added
  // when a node is enqueued, either root node initially, or a child node during
  // the visit.
  struct FetchQueue {
    std::vector<PositionKey> position_keys;
    std::vector<lczero::ChessBoard> boards;
    std::vector<size_t> visit_limit;
    std::vector<size_t> parent_idx;
    enum class Status { kQueued, kNew, kFetched, kBusy };
    std::vector<Status> node_status;
  };
  // All vectors in ForwardPassNodeData are the same size, but the size may be
  // behind FetchQueue. An element is added when a node is fetched from the
  // storage.
  struct NodeData {
    // Forward pass.
    std::vector<NodeHead> heads;
    std::vector<size_t> last_edge_idx;
  };
  // All edge data of ForwardPassNodeData node, linearized for parallel
  // processing.
  struct EdgeData {
    std::vector<float> p;
    std::vector<uint16_t> moves;
    std::vector<float> q_wl;
    std::vector<float> q_d;
    std::vector<float> q_ml;
    std::vector<uint32_t> n;
  };
  // Indices of leaf nodes (in FetchQueue/NodeData).
  struct LeafData {
    std::vector<size_t> leaf_indices;
  };

  FetchQueue queue_;
  NodeData nodes_;
  EdgeData edges_;
  LeafData leafs_;
  BatchStats stats_;
};

}  // namespace lc2