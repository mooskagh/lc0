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

#include "lc2/search/nodes-worker.h"

#include "lc2/search/search.h"
#include "utils/exception.h"

namespace lczero {
namespace lc2 {

NodesWorker::NodesWorker(Search* search, NodeShard* shard)
    : search_(search), shard_(shard) {}

void NodesWorker::RunBlocking() {
  while (true) {
    auto token = channel_.Dequeue();

    switch (token.message()->type) {
      case Message::kNodeGather:
        GatherNode(std::move(token));

        break;
      default:
        throw Exception("Unexpected message type " +
                        std::to_string(token.message()->type) +
                        " in node worker.");
    }
  }
}

void NodesWorker::GatherNode(Token token) {
  auto* msg = token.message();

  // A node can be in following states:
  // 1. Not created yet (will be sent for eval).
  // 2. Already sent for eval but not returned (collision).
  // 3. It was already evaluated (have to visit deeper).
  // 4. (TODO) The node is in external storage.
  auto hash = msg->position_history.Last().Hash();

  auto [found, node] = shard_->GetNode(hash);
  if (!found) {
    // This is the first time this node is visited, so send it for eval.
    const bool will_collide = msg->arity > 1;
    auto eval_token = will_collide ? token.SplitOff(1) : std::move(token);
    eval_token.message()->type = Message::kEvalEval;
    search_->DispatchToEval(std::move(token));

    // It was a single visit which was sent to evals, no need to handle
    // collisions.
    if (!will_collide) return;
  }

  if (!node->eval_completed) {
    // Collision.
    msg->type = Message::kRootCollision;
    search_->DispatchToRoot(std::move(token));
    return;
  }

  // DO NOT SUBMIT TODO(crem) Send out-of-order here if parent N is less than
  // own N.

  // This is a node which was evaluated earlier, forward the visit further
  // down the tree.
  ForwardVisit(node, std::move(token));
}

void NodesWorker::ForwardVisit(Node* node, Token token) {
  auto* msg = token.message();
  // The message down the tree will not be a root node anymore.
  msg->is_root_node = false;

  // A heap, Q+U to a move index.
  using Item = std::tuple<Node::NT::QPlusU, Node::NT::Q, size_t>;
  std::vector<Item> q_plus_u;
  const auto move_count = node->num_edges();
  q_plus_u.reserve(move_count);

  const auto cpuct = Node::NT::ComputeQFactor(1.23, node->n);
  const Node::NT::FPU fpu = -1;  // DO NOT SUBMIT  compute FPU properly.
  for (size_t i = 0; i < move_count; ++i) {
    auto q = Node::NT::ComputeQ(cpuct, node->q_edge[i]);
    auto u = Node::NT::ComputeU(node->p_edge[i], fpu, node->n_edge[i], node->n);
    q_plus_u.push_back({q + u, q, i});
  }
  std::make_heap(q_plus_u.begin(), q_plus_u.end());

  int num_visits = msg->arity;
  assert(num_visits > 0);

  std::vector<int> edge_to_visits(node->num_edges());

  // TODO: Instead of appending one visit per iteration, it's possible to do
  // several.
  while (true) {
    std::pop_heap(q_plus_u.begin(), q_plus_u.end());
    auto& qu = std::get<0>(q_plus_u.back());
    const auto& q = std::get<1>(q_plus_u.back());
    const auto& idx = std::get<2>(q_plus_u.back());
    auto new_n = ++edge_to_visits[idx];
    qu = q + Node::NT::ComputeU(node->p_edge[idx], fpu,
                                node->n_edge[idx] + new_n, node->n);
    --num_visits;
    if (num_visits == 0) break;
    std::push_heap(q_plus_u.begin(), q_plus_u.end());
  }

  for (size_t i = 0; i < move_count; ++i) {
    const auto count = edge_to_visits[i];
    if (count == 0) continue;
    const bool last_iteration = num_visits == msg->arity;
    auto send_token = last_iteration ? std::move(token) : token.SplitOff(count);
    send_token.message()->position_history.Append(node->edges[i]);
    search_->DispatchToNodes(std::move(send_token));
    if (last_iteration) break;
  }
}

}  // namespace lc2
}  // namespace lczero