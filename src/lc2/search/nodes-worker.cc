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
    auto msg = channel_.Dequeue();

    switch (msg->type) {
      case Message::kNodeGather:
        GatherNode(std::move(msg));
        break;
      case Message::kNodeBackProp:
        BackProp(std::move(msg));
        break;
      default:
        throw Exception("Unexpected message type " + std::to_string(msg->type) +
                        " in node worker.");
    }
  }
}

void NodesWorker::GatherNode(std::unique_ptr<Message> msg) {
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
    auto eval_msg = will_collide ? msg->SplitOff(1) : std::move(msg);
    eval_msg->type = Message::kEvalEval;
    search_->DispatchToEval(std::move(eval_msg));

    // It was a single visit which was sent to evals, no need to handle
    // collisions.
    if (!will_collide) return;
  }

  if (!node->eval_completed) {
    // Collision.
    msg->type = Message::kRootCollision;
    search_->DispatchToRoot(std::move(msg));
    return;
  }

  // DO NOT SUBMIT TODO(crem) Send out-of-order here if parent N is less than
  // own N.

  // This is a node which was evaluated earlier, forward the visit further
  // down the tree.
  ForwardVisit(node, std::move(msg));
}

void NodesWorker::ForwardVisit(Node* node, std::unique_ptr<Message> msg) {
  // A heap, Q+U to a move index.
  using Item = std::tuple<Node::NT::QPlusU, Node::NT::Q, size_t>;
  std::vector<Item> q_plus_u;
  const auto move_count = node->num_edges();
  q_plus_u.reserve(move_count);

  const auto cpuct_mult = Node::NT::ComputeUFactor(1.23, node->n);
  const Node::NT::Q fpu = Node::NT::ComputeFPU();
  for (size_t i = 0; i < move_count; ++i) {
    auto q = Node::NT::ComputeQ(node->n_edge[i], fpu, node->q_edge[i]);
    auto u = Node::NT::ComputeU(cpuct_mult, node->p_edge[i], node->n_edge[i]);
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
    qu = q + Node::NT::ComputeU(cpuct_mult, node->p_edge[idx],
                                node->n_edge[idx] + new_n);
    --num_visits;
    if (num_visits == 0) break;
    std::push_heap(q_plus_u.begin(), q_plus_u.end());
  }

  for (size_t i = 0; i < move_count; ++i) {
    const auto count = edge_to_visits[i];
    if (count == 0) continue;
    const bool last_iteration = num_visits == msg->arity;
    auto send_msg = last_iteration ? std::move(msg) : msg->SplitOff(count);
    send_msg->position_history.Append(node->edges[i]);
    send_msg->move_idx.push_back(i);
    search_->DispatchToNodes(std::move(send_msg));
    if (last_iteration) break;
  }
}

void NodesWorker::BackProp(std::unique_ptr<Message> msg) {
  auto hash = msg->position_history.Last().Hash();

  auto [found, node] = shard_->GetNode(hash);
  assert(found);
  // TODO(crem) Support higher arity updates.
  assert(msg->arity == 1);
  const auto arity = msg->arity;
  assert(msg->eval_result);
  auto& eval_result = *msg->eval_result;

  const bool is_leaf = !node->eval_completed;
  if (is_leaf) {
    msg->node_height_is_odd = false;
    node->eval_completed = true;
    node->wdl = msg->eval_result->wdl;
    node->n = arity;

    node->edges = std::move(eval_result.edges);
    node->p_edge = std::move(eval_result.p_edge);
    node->q_edge.resize(node->edges.size());
    node->n_edge.resize(node->edges.size());
  } else {
    node->n += arity;
    const auto idx = msg->move_idx.back();
    node->n_edge[idx] += arity;
    node->q_edge[idx] = msg->child_q;
    msg->move_idx.pop_back();
    Node::NT::UpdateWDL(&node->wdl, msg->eval_result->wdl, arity, node->n,
                        msg->node_height_is_odd);
  }

  const bool is_root = msg->move_idx.empty();
  if (is_root) {
    msg->type = Message::kRootBackPropDone;
    search_->DispatchToRoot(std::move(msg));
  } else {
    msg->node_height_is_odd = !msg->node_height_is_odd;
    msg->position_history.Pop();
    msg->child_q = Node::NT::WDLtoQ(node->wdl);
    search_->DispatchToNodes(std::move(msg));
  }
}

}  // namespace lc2
}  // namespace lczero