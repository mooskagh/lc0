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

#include <locale>

#include "chess/board.h"
#include "lc2/chess/position-key.h"
#include "lc2/mcts/node.h"

namespace lc2 {

void Batch::EnqueuePosition(const lczero::ChessBoard& board,
                            const PositionKey& key, size_t visit_count,
                            size_t parent_idx) {
  boards_.emplace_back(board);
  positions_keys_.emplace_back(key);
  visit_counts_.emplace_back(visit_count);
  parent_idx_.emplace_back(parent_idx);
}

void Batch::Gather(NodeStorage* node_storage) {
  while (fetched_size() < size()) {
    const size_t begin_idx = fetched_size();
    const size_t end_idx = size();

    // Fetch nodes from the storage into the working arrays.
    FetchNodes(node_storage, begin_idx, end_idx);
    ProcessNodes(begin_idx, end_idx);
    CommitNodes(node_storage, begin_idx, end_idx);
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

void Batch::CommitNodes(NodeStorage* node_storage, size_t begin_idx,
                        size_t end_idx) {
  auto fetch_head_func = [&](size_t offset, NodeHead* head,
                             NodeStorage::Status status) -> bool {
    assert(status == NodeStorage::Status::kFetched);
    const size_t idx = begin_idx + offset;
    // The node was busy, don't update.
    auto& local_head = node_heads_[idx];
    if (local_head.flags.is_being_processed) return false;
    // If tail is valid, will handle that in the tail func.
    if (head->flags.tail_is_valid) return true;
    // This may happen ourside of mutex, if causes problems.
    unpacked_nodes_[begin_idx + offset].UpdateNIntoHead(&local_head);
    *head = local_head;
    return false;
  };
  auto fetch_tail_func = [&](size_t offset, NodeHead* head, NodeTail* tail) {
    auto& unpacked_node = unpacked_nodes_[begin_idx + offset];
    const size_t idx = begin_idx + offset;
    auto& local_head = node_heads_[idx];
    unpacked_node.UpdateNIntoHeadAndTail(&local_head, tail);
    *head = local_head;
  };
  node_storage->FetchOrCreate(
      {&positions_keys_[begin_idx], end_idx - begin_idx}, fetch_head_func,
      fetch_tail_func);
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

PositionKey UpdatePositionKey(const PositionKey& /* previous_key */,
                              const lczero::ChessBoard& /* previous_board */,
                              lczero::Move /* move */,
                              const lczero::ChessBoard& new_board) {
  return PositionKey(new_board.Hash());
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

  auto fpu_q = GetNodeQ(head);
  auto edge_visits =
      DistributeVisits(node.p, node.n, node.q, fpu_q, head.n, visits);
  const auto& board = boards_[idx];
  const auto& key = positions_keys_[idx];
  for (size_t i = 0; i < edge_visits.size(); ++i) {
    size_t visits = edge_visits[i];
    if (visits == 0) continue;
    head.n += visits;
    while (node.n.size() <= i) {
      node.n.emplace_back();
      node.q.emplace_back(fpu_q);
    }
    node.n[i] += visits;

    auto move = lczero::Move::from_packed_int(node.moves[i]);
    auto new_board = board;
    new_board.ApplyMove(move);
    auto new_key = UpdatePositionKey(key, board, move, new_board);
    EnqueuePosition(new_board, new_key, visits, idx);
  }
  if (node.n.size() > head.edge_n.size()) head.flags.tail_is_valid = true;
}

}  // namespace lc2