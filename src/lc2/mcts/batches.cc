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

#include "lc2/mcts/batches.h"

#include <absl/algorithm/container.h>

#include "lc2/mcts/node.h"

namespace lc2 {

void Batch::EnqueuePosition(const lczero::ChessBoard& board,
                            const PositionKey& key, size_t visit_count) {
  boards_.emplace_back(board);
  positions_keys_.emplace_back(key);
  visit_counts_.emplace_back(visit_count);
}

void Batch::Gather(NodeStorage* node_storage) {
  while (fetched_size() < size()) {
    const size_t begin_idx = fetched_size();
    const size_t end_idx = size();

    // Fetch nodes from the storage into the working arrays.
    FetchNodes(node_storage, begin_idx, end_idx);
    ProcessNodes(begin_idx, end_idx);
    // CommitNodes
  }
}

void Batch::FetchNodes(NodeStorage* node_storage, size_t begin_idx,
                       size_t end_idx) {
  node_heads_.resize(end_idx);
  unpacked_nodes_.resize(end_idx);
  fetch_status_.resize(end_idx);
  auto fetch_head_func = [&](size_t offset, NodeHead* head,
                             NodeStorage::Status status) -> bool {
    const size_t idx = begin_idx + offset;
    node_heads_[idx] = *head;
    fetch_status_[idx] = status;
    const bool is_busy = head->flags.is_being_processed;
    head->flags.is_being_processed = true;
    if (!is_busy && !head->flags.tail_is_valid) {
      // TODO(crem) this particular unpack doesn't have to happen under mutex,
      // it's possible to store index and then unpack later. Not sure whether it
      // helps, so not doing that for now.
      auto& unpacked_node = unpacked_nodes_[begin_idx + offset];
      unpacked_node.UnpackFromHead(*head);
    }
    return !is_busy && head->flags.tail_is_valid;
  };
  auto fetch_tail_func = [&](size_t offset, NodeHead* head, NodeTail* tail) {
    auto& unpacked_node = unpacked_nodes_[begin_idx + offset];
    unpacked_node.UnpackFromHeadAndTail(*head, *tail);
  };
  node_storage->FetchOrCreate(
      {&positions_keys_[begin_idx], end_idx - begin_idx}, fetch_head_func,
      fetch_tail_func);
}

void Batch::ProcessNodes(size_t begin_idx, size_t end_idx) {
  for (auto idx = begin_idx; idx != end_idx; ++idx) {
    if (node_heads_[idx].flags.is_being_processed) continue;
    ProcessSingleNode(idx);
  }
}

namespace {

constexpr float kCpuct = 6.0f;

std::vector<uint16_t> DistributeVisits(const absl::Span<const float> p,
                                       const absl::Span<const uint32_t> n,
                                       const absl::Span<const float> edge_q,
                                       float fpu_q, uint32_t total_n,
                                       size_t total_visits) {
  auto get_n = [&](size_t idx) { return (idx >= n.size()) ? 0 : n[idx]; };
  using QUandIdx = std::pair<float, size_t>;
  const float sqrt_total_n = std::sqrt(total_n);
  std::vector<uint16_t> per_edge_visit(p.size());
  auto get_q_plus_u = [&](size_t i) {
    const auto n = get_n(i);
    const float q = (n == 0) ? fpu_q : edge_q[i];
    const float u = kCpuct * p[i] * sqrt_total_n / (1 + n + per_edge_visit[i]);
    return q + u;
  };
  std::vector<QUandIdx> qu_i;
  qu_i.reserve(p.size());
  for (size_t i = 0; i < p.size(); ++i) qu_i.emplace_back(get_q_plus_u(i), i);
  absl::c_make_heap(qu_i);

  while (true) {
    assert(total_visits > 0);
    absl::c_pop_heap(qu_i);
    do {
      // TODO(crem) instead of a loop, it's possible to do some math.
      auto& [qu, idx] = qu_i.back();
      ++per_edge_visit[idx];
      qu = get_q_plus_u(idx);
    } while (total_visits > 0 && qu_i.back().first >= qu_i.front().first);
    if (total_visits == 0) break;
    absl::c_push_heap(qu_i);
  }
  return per_edge_visit;
}

}  // namespace

void Batch::ProcessSingleNode(size_t idx) {
  auto& head = node_heads_[idx];
  auto& node = unpacked_nodes_[idx];
  const size_t visits = visit_counts_[idx];

  if (fetch_status_[idx] == NodeStorage::Status::kCreated) {
    // Fresh node, send to NN for eval.
    ++stats_.nn_evals;
    stats_.collisions += visits - 1;
    idx_to_eval_.push_back(idx);
    return;
  }
  if (head.n == 0) {
    // N=0 meaning NN is running somewhere else, that's a collision.
    stats_.collisions += visits;
    return;
  }

  auto edge_visits =
      DistributeVisits(node.p, node.n, node.q, GetNodeQ(head), head.n, visits);
}

}  // namespace lc2