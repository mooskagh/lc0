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
#include <absl/types/span.h>

#include <cstring>
#include <iterator>
#include <locale>
#include <string_view>

#include "chess/board.h"
#include "lc2/chess/position-key.h"
#include "lc2/mcts/formulas.h"
#include "lc2/mcts/node.h"

namespace lc2 {

void Batch::EnqueuePosition(const lczero::ChessBoard& board,
                            const PositionKey& key, size_t visit_limit,
                            size_t parent_node_idx, size_t parent_edge_idx) {
  queue_.boards.emplace_back(board);
  queue_.position_keys.emplace_back(key);
  queue_.visit_limit.emplace_back(visit_limit);
  queue_.parent_node_idx.emplace_back(parent_node_idx);
  queue_.parent_edge_idx.emplace_back(parent_edge_idx);
  queue_.node_status.emplace_back(FetchQueue::Status::kQueued);
}

void Batch::Gather(NodeStorage* node_storage) {
  while (fetched_size() < queue_size()) {
    const size_t begin_idx = fetched_size();
    const size_t begin_edge_idx = edges_size();

    // Fetch nodes from the storage into the working arrays.
    FetchNodes(node_storage, begin_idx);
    ComputeVisitedPolicy(begin_edge_idx);
    ComputeNodeVals(begin_edge_idx);
    ComputeQU(begin_edge_idx);
    ProcessNodes(begin_idx);
  }
}

void Batch::FetchNodes(NodeStorage* node_storage, size_t from) {
  nodes_.heads.resize(queue_size());
  auto fetch_head_func = [&](size_t offset, NodeHead* head,
                             NodeStorage::Status fetch_status) -> bool {
    const size_t idx = from + offset;
    auto& status = queue_.node_status[idx];
    nodes_.heads.push_back(*head);
    switch (fetch_status) {
      case NodeStorage::Status::kCreated:
        status = FetchQueue::Status::kNew;
        return false;
      case NodeStorage::Status::kFetched:
        if (head->flags.is_being_processed) {
          status = FetchQueue::Status::kBusy;
          return false;
        }
        status = FetchQueue::Status::kFetched;
    };
    head->flags.is_being_processed = true;
    if (head->TailIsValid()) {
      return true;
    } else {
      UnpackEdgesFromHead(idx, *head);
      return false;
    }
  };
  auto fetch_tail_func = [&](size_t offset, NodeHead* head, NodeTail* tail) {
    UnpackEdgesFromHeadAndTail(from + offset, *head, *tail);
  };
  node_storage->FetchOrCreate(
      {&queue_.position_keys[from], queue_size() - from}, fetch_head_func,
      fetch_tail_func);
}

void Batch::ComputeVisitedPolicy(size_t from_edge) {
  nodes_.visited_policy.resize(fetched_size());
  for (size_t i = from_edge; i < edges_size(); ++i) {
    const size_t node_idx = edges_.node_idx[i];
    if (edges_.n[i] != 0) nodes_.visited_policy[node_idx] += edges_.p[i];
  }
}

void Batch::ComputeNodeVals(size_t from_edge) {
  nodes_.fpu.resize(fetched_size());
  nodes_.u_factor.resize(fetched_size());
  const size_t from_node = edges_.node_idx[from_edge];
  const float draw_score = params_.GetDrawScore();
  const float fpu_value = params_.GetFpuValue();
  const float cpuct_init = params_.GetCPuct();
  const float cpuct_f = params_.GetCPuctFactor();
  const float cpuct_base = params_.GetCPuctBase();
  for (size_t i = from_node; i < fetched_size(); ++i) {
    const auto& head = nodes_.heads[i];
    auto q = ComputeQ(head.q_wl, head.q_d, head.q_ml, draw_score);
    nodes_.fpu[i] = ComputeFPUReduction(q, nodes_.visited_policy[i], fpu_value);
    nodes_.u_factor[i] =
        ComputeUFactor(head.n, cpuct_init, cpuct_f, cpuct_base);
  }
}

void Batch::ComputeQU(size_t from_edge) {
  edges_.qu.resize(edges_size());
  edges_.q.resize(edges_size());
  const float draw_score = params_.GetDrawScore();
  for (size_t i = from_edge; i < edges_size(); ++i) {
    const float q =
        ComputeQ(edges_.q_wl[i], edges_.q_d[i], edges_.q_ml[i], draw_score);
    edges_.q[i] = q;
    const size_t node_idx = edges_.node_idx[i];
    const float u =
        ComputeU(nodes_.u_factor[node_idx], edges_.p[i], edges_.n[i]);
    edges_.qu[i] = q + u;
  }
}

namespace {
template <typename T, size_t Size>
void SerializeIntoHeadAndTail(absl::Span<T> data, std::array<T, Size>* head,
                              std::string* tail) {
  std::copy(data.begin(), data.begin() + std::min(data.size(), head->size()),
            head->begin());
  if (head->size() < data.size()) {
    assert(tail != nullptr);
    tail->append(reinterpret_cast<const char*>(&data[head->size()]),
                 sizeof(T) * (data.size() - head->size()));
  }
}
}  // namespace

void Batch::PackEdgesIntoHeadAndTail(size_t node_idx, NodeHead* head,
                                     NodeTail* tail) {
  const size_t header_bytes_in_tail =
      (sizeof(decltype(head->edge_p)::value_type) +
       sizeof(decltype(head->moves)::value_type)) *
      (head->num_edges - head->edge_p.size());
  tail->resize(header_bytes_in_tail);
  const size_t edge_idx = nodes_.start_edge_idx[node_idx];
  const size_t edge_count = nodes_.start_edge_idx[node_idx + 1] - edge_idx;
  SerializeIntoHeadAndTail(absl::MakeSpan(&edges_.n[edge_idx], edge_count),
                           &head->edge_n, tail);
  SerializeIntoHeadAndTail(absl::MakeSpan(&edges_.q_wl[edge_idx], edge_count),
                           &head->edge_q_wl, tail);
  SerializeIntoHeadAndTail(absl::MakeSpan(&edges_.q_d[edge_idx], edge_count),
                           &head->edge_q_d, tail);
  SerializeIntoHeadAndTail(absl::MakeSpan(&edges_.q_ml[edge_idx], edge_count),
                           &head->edge_q_ml, tail);
}

namespace {
template <typename T, size_t InSize>
void AppendFromHeadAndTail(const std::array<T, InSize>& head_in,
                           std::string_view* tail_buffer, size_t copy_count,
                           size_t extra_count, std::vector<T>* out,
                           T default_val = T{}) {
  // Copy head.
  const size_t head_copy_count = std::min(copy_count, InSize);
  std::copy(head_in.begin(), head_in.begin() + head_copy_count,
            std::back_inserter(*out));
  copy_count -= head_copy_count;
  // Copy tail.
  if (copy_count > 0 && tail_buffer != nullptr) {
    std::copy(reinterpret_cast<const T*>(tail_buffer->data()),
              reinterpret_cast<const T*>(tail_buffer->data()) + copy_count,
              std::back_inserter(*out));
    tail_buffer->remove_prefix(copy_count * sizeof(T));
    copy_count = 0;
  }
  // Fill with the default val;
  out->insert(out->end(), copy_count + extra_count, default_val);
}
}  // namespace

void Batch::UnpackEdgesFromHeadAndBuffer(const NodeHead& head,
                                         std::string_view* tail,
                                         size_t to_fetch, size_t to_pad) {
  AppendFromHeadAndTail(head.edge_p, tail, to_fetch + to_pad, 0, &edges_.p);
  AppendFromHeadAndTail(head.moves, tail, to_fetch + to_pad, 0, &edges_.moves);
  AppendFromHeadAndTail(head.edge_n, tail, to_fetch, to_pad, &edges_.n);
  AppendFromHeadAndTail(head.edge_q_wl, tail, to_fetch, to_pad, &edges_.q_wl);
  AppendFromHeadAndTail(head.edge_q_d, tail, to_fetch, to_pad, &edges_.q_d);
  AppendFromHeadAndTail(head.edge_q_ml, tail, to_fetch, to_pad, &edges_.q_ml);
}

void Batch::UnpackEdgesFromHeadAndTail(size_t node_idx, const NodeHead& head,
                                       const NodeTail& tail) {
  std::string_view tail_buffer(tail.data(), tail.size());
  const size_t num_filled_edges = head.num_filled_edges;
  const size_t to_pad = head.num_edges == num_filled_edges ? 0 : 1;
  UnpackEdgesFromHeadAndBuffer(head, &tail_buffer, num_filled_edges, to_pad);
  edges_.node_idx.insert(edges_.node_idx.end(), num_filled_edges + to_pad,
                         node_idx);
  nodes_.start_edge_idx.push_back(edges_size());
}

void Batch::UnpackEdgesFromHead(size_t node_idx, const NodeHead& head) {
  const size_t to_fetch =
      std::min(head.edge_p.size(), static_cast<size_t>(head.num_edges));
  UnpackEdgesFromHeadAndBuffer(head, nullptr, to_fetch, 0);
  edges_.node_idx.insert(edges_.node_idx.end(), to_fetch, node_idx);
  nodes_.start_edge_idx.push_back(edges_size());
}

void Batch::ProcessNodes(size_t begin_idx) {
  for (auto idx = begin_idx; idx != fetched_size(); ++idx) {
    if (queue_.node_status[idx] == FetchQueue::Status::kBusy) continue;
    ProcessSingleNode(idx);
  }
}

void Batch::CommitNodes(NodeStorage* node_storage, size_t begin_idx) {
  auto fetch_head_func = [&](size_t offset, NodeHead* head,
                             NodeStorage::Status status) -> bool {
    assert(status == NodeStorage::Status::kFetched);
    const size_t idx = begin_idx + offset;
    // The node was busy, don't update.
    if (queue_.node_status[idx] != FetchQueue::Status::kFetched) return false;
    auto& local_head = nodes_.heads[idx];
    // If it was a collision, don't update.
    if (local_head.n == 0) return false;
    if (local_head.flags.is_being_processed) return false;
    // If tail is valid, will handle that in the tail func.
    if (head->TailIsValid()) return true;
    // This may happen ourside of mutex, if causes problems.
    PackEdgesIntoHeadAndTail(idx, &local_head, nullptr);
    *head = local_head;
    return false;
  };
  auto fetch_tail_func = [&](size_t offset, NodeHead* head, NodeTail* tail) {
    const size_t idx = begin_idx + offset;
    auto& local_head = nodes_.heads[idx];
    PackEdgesIntoHeadAndTail(idx, &local_head, tail);
    *head = local_head;
  };
  const size_t count = queue_size() - begin_idx;
  node_storage->FetchOrCreate({&queue_.position_keys[begin_idx], count},
                              fetch_head_func, fetch_tail_func);
}

std::vector<uint16_t> Batch::DistributeVisits(size_t from_edge,
                                              size_t edge_count,
                                              size_t visit_count) {
  assert(edge_count > 0);
  if (edge_count == 1) return std::vector<uint16_t>(1, visit_count);
  const float u_factor = nodes_.u_factor[edges_.node_idx[from_edge]];
  std::vector<uint16_t> per_edge_visits(edge_count);
  std::vector<size_t> indices(edge_count);
  std::iota(indices.begin(), indices.end(), from_edge);
  auto qu_less = [&](size_t a, size_t b) {
    if (edges_.qu[a] != edges_.qu[b]) return edges_.qu[a] < edges_.qu[b];
    // Among edges with equal Q+U, prefer ones with smaller index.
    return a > b;
  };

  absl::c_make_heap(indices, qu_less);
  while (true) {
    assert(visit_count > 0);
    absl::c_pop_heap(indices, qu_less);
    const size_t i1 = indices.front();
    const size_t i2 = indices.back();
    const size_t visits = std::min(
        visit_count, ComputeVisitsToReach(
                         edges_.p[i1], edges_.n[i1], edges_.q[i1], edges_.p[i2],
                         edges_.n[i2], edges_.q[i2], u_factor));
    edges_.n[i2] += visits;
    edges_.qu[i2] = ComputeU(u_factor, edges_.p[i2], edges_.n[i2]);
    visit_count -= visits;
    per_edge_visits[i2 - from_edge] += visits;
    if (visit_count == 0) break;
    absl::c_push_heap(indices, qu_less);
  }
  return per_edge_visits;
}

void Batch::ProcessSingleNode(size_t idx) {
  const size_t visits = queue_.visit_limit[idx];
  auto& head = nodes_.heads[idx];

  if (queue_.node_status[idx] == FetchQueue::Status::kNew) {
    // Fresh node, send to NN for eval.
    ++stats_.nn_evals;
    stats_.collisions += visits - 1;
    leafs_.node_indices.push_back(idx);
    return;
  }
  if (head.n == 0) {
    // N=0 meaning NN is running somewhere else, that's a collision.
    stats_.collisions += visits;
    return;
  }

  const size_t from_edge = nodes_.start_edge_idx[idx];
  const size_t to_edge = nodes_.start_edge_idx[idx + 1];
  auto edge_visits = DistributeVisits(from_edge, to_edge - from_edge, visits);
  const auto& board = queue_.boards[idx];
  const auto& key = queue_.position_keys[idx];
  for (size_t i = 0; i < edge_visits.size(); ++i) {
    size_t visits = edge_visits[i];
    if (visits == 0) continue;
    head.n += visits;
    // edges_.n[i] += visits; already done inside DeistributeVisits().
    const size_t move_idx = from_edge + i;
    auto move = edges_.moves[move_idx];
    auto new_board = board;
    new_board.ApplyMove(lczero::Move::from_packed_int(move));
    auto new_key = ComputePositionKey(key, board, move, new_board);
    EnqueuePosition(new_board, new_key, visits, idx, move_idx);
  }
  if (head.num_filled_edges < edge_visits.size() &&
      edge_visits[head.num_filled_edges] > 0) {
    head.num_filled_edges = edge_visits.size();
  }
}

}  // namespace lc2