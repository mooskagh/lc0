/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

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

#include <absl/container/flat_hash_map.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "search/lc5/key.h"
#include "search/lc5/node_store.h"
#include "search/lc5/settings.h"

namespace lczero::lc5 {

using MaterializationTicketId = uint64_t;

enum class NodeLifecycle : uint8_t { kMaterializing, kExpanded };

// The machinery owns an edge's move, prior and reservations; `stats` belongs
// to the policy P.
template <typename P>
struct EdgeState {
  Move move;
  uint16_t in_flight = 0;  // Virtual loss; bounded by kMaxActiveVisitsLimit.
  float prior = 0.0f;
  typename P::EdgeStats stats{};
  static_assert(kMaxActiveVisitsLimit <=
                std::numeric_limits<decltype(in_flight)>::max());
};

// Deterministic tie-break between equally ranked edges: higher prior first,
// then the lower move encoding. A node's edges are sorted in this order, and
// selection relies on it.
template <typename Edge>
bool PreferOnTie(const Edge& a, const Edge& b) {
  if (a.prior != b.prior) return a.prior > b.prior;
  return a.move.raw_data() < b.move.raw_data();
}

template <typename P>
struct NodeState {
  // Unique per (re)creation, so updates for an erased node are rejected.
  uint64_t generation = 0;
  NodeLifecycle lifecycle = NodeLifecycle::kMaterializing;
  MaterializationTicketId ticket = 0;  // Zero once expanded.
  TerminalKind terminal = TerminalKind::kNonTerminal;
  typename P::NodeStats stats{};
  std::vector<EdgeState<P>> edges{};
  uint32_t started_end = 0;  // Edges from here on have never been started.
};

struct FindOrCreateResult {
  uint64_t generation;
  bool created;
  NodeLifecycle lifecycle;
  MaterializationTicketId ticket;
};

struct InstallResult {
  uint64_t generation;
  bool accepted;
};

enum class SelectStatus { kSelected, kMissing, kMaterializing, kTerminal };

struct SelectResult {
  SelectStatus status = SelectStatus::kMissing;
  Move move{};
  uint64_t generation = 0;
  TerminalKind terminal = TerminalKind::kNonTerminal;
};

enum class UpdateResult {
  kApplied,
  kMissing,
  kStale,
  kEdgeMissing,
  kUnderflow
};

template <typename P>
struct BackupResult {
  UpdateResult node;
  UpdateResult edge;
  // What the parent receives; set only when the node update was applied.
  std::optional<typename P::Message> to_parent;
};

// Sharded map from NodeKey to node search state and outgoing edges. Edges do
// not point to children; callers derive child keys from their own history.
// Every mutation names the node generation it expects, so updates computed
// against an erased and recreated node are rejected instead of misapplied.
// The policy P owns the node and edge statistics, how a message updates them
// and how an edge is selected.
template <typename P>
class GameGraph {
 public:
  using Node = NodeState<P>;
  using Edge = EdgeState<P>;
  using Message = typename P::Message;

  static constexpr size_t kShardCount = 1024;

  // Finds an existing node or creates a kMaterializing one owned by `ticket`.
  FindOrCreateResult FindOrCreateMaterializing(NodeKey key,
                                               MaterializationTicketId ticket);

  // Erases only the matching materializing placeholder, never an expansion.
  bool CancelMaterialization(NodeKey key, MaterializationTicketId ticket);

  // Expands the node with `payload` if it is still materializing under
  // `ticket`, creating it if absent. Publication is one-shot per generation.
  // An accepted install also updates the node with `payload.leaf` as its
  // first visit.
  InstallResult InstallPayload(NodeKey key, MaterializationTicketId ticket,
                               const ExpansionPayload& payload);

  // Selects an edge by the policy and reserves it (in_flight += 1); otherwise
  // reports why the node can't be selected from. Every result but kMissing
  // carries the node's generation, and kTerminal also its terminal kind.
  SelectResult SelectAndReserve(NodeKey key, const typename P::Params& params);

  // Applies the policy's update to the node and, if `move` is given,
  // completes that edge's reservation. On success `to_parent` is the message
  // the parent receives.
  BackupResult<P> BackupNode(NodeKey key, uint64_t expected_generation,
                             Message message, std::optional<Move> move);

  // Releases an edge reservation without committing a visit.
  UpdateResult CancelEdge(NodeKey key, uint64_t expected_generation, Move move);

  std::optional<Node> SnapshotNode(NodeKey key) const;

  bool Erase(NodeKey key);
  void Clear();
  size_t Size() const { return size_.load(std::memory_order_relaxed); }
  std::vector<std::pair<NodeKey, Node>> SnapshotAllForTesting() const;

 private:
  struct Shard {
    mutable std::mutex mutex;
    absl::flat_hash_map<NodeKey, Node> nodes;
  };

  static size_t ShardIndex(NodeKey key) {
    return (key.hash * 11400714819323198485ull) >> 54;
  }
  Shard& ShardFor(NodeKey key) { return shards_[ShardIndex(key)]; }
  const Shard& ShardFor(NodeKey key) const { return shards_[ShardIndex(key)]; }
  // Inserts a fresh materializing node; the shard must be locked.
  Node& InsertLocked(Shard& shard, NodeKey key, MaterializationTicketId ticket);

  template <typename Map>
  static auto* FindNode(Map& nodes, NodeKey key) {
    const auto it = nodes.find(key);
    return it == nodes.end() ? nullptr : &it->second;
  }

  static UpdateResult CheckGeneration(const Node* node, uint64_t generation) {
    if (!node) return UpdateResult::kMissing;
    if (node->generation != generation) return UpdateResult::kStale;
    return UpdateResult::kApplied;
  }

  static Edge* FindEdge(Node& node, Move move) {
    const auto it = std::ranges::find(node.edges, move, &Edge::move);
    return it == node.edges.end() ? nullptr : &*it;
  }

  // Releases one reservation of the edge.
  static UpdateResult ReleaseEdge(Edge* edge) {
    if (!edge) return UpdateResult::kEdgeMissing;
    if (edge->in_flight == 0) return UpdateResult::kUnderflow;
    --edge->in_flight;
    return UpdateResult::kApplied;
  }

  std::array<Shard, kShardCount> shards_;
  std::atomic<size_t> size_{0};
  std::atomic<uint64_t> next_generation_{1};
};

template <typename P>
NodeState<P>& GameGraph<P>::InsertLocked(Shard& shard, NodeKey key,
                                         MaterializationTicketId ticket) {
  const uint64_t generation =
      next_generation_.fetch_add(1, std::memory_order_relaxed);
  assert(generation != 0);
  size_.fetch_add(1, std::memory_order_relaxed);
  return shard.nodes
      .emplace(key, Node{.generation = generation, .ticket = ticket})
      .first->second;
}

template <typename P>
FindOrCreateResult GameGraph<P>::FindOrCreateMaterializing(
    NodeKey key, MaterializationTicketId ticket) {
  auto& shard = ShardFor(key);
  std::lock_guard lock(shard.mutex);
  if (const Node* node = FindNode(shard.nodes, key)) {
    return {node->generation, false, node->lifecycle, node->ticket};
  }
  const Node& node = InsertLocked(shard, key, ticket);
  return {node.generation, true, node.lifecycle, ticket};
}

template <typename P>
bool GameGraph<P>::CancelMaterialization(NodeKey key,
                                         MaterializationTicketId ticket) {
  auto& shard = ShardFor(key);
  std::lock_guard lock(shard.mutex);
  const Node* node = FindNode(shard.nodes, key);
  if (!node || node->lifecycle != NodeLifecycle::kMaterializing ||
      node->ticket != ticket) {
    return false;
  }
  shard.nodes.erase(key);
  size_.fetch_sub(1, std::memory_order_relaxed);
  return true;
}

template <typename P>
InstallResult GameGraph<P>::InstallPayload(NodeKey key,
                                           MaterializationTicketId ticket,
                                           const ExpansionPayload& payload) {
  assert(payload.moves.size() == payload.priors.size());
  assert(payload.terminal == TerminalKind::kNonTerminal ||
         payload.moves.empty());
  auto& shard = ShardFor(key);
  std::lock_guard lock(shard.mutex);
  Node* existing = FindNode(shard.nodes, key);
  Node& node = existing ? *existing : InsertLocked(shard, key, ticket);
  // Late or duplicate completions must not reset expanded edges, including
  // their reservations and statistics.
  if (node.lifecycle == NodeLifecycle::kExpanded || node.ticket != ticket) {
    return {node.generation, false};
  }
  node.lifecycle = NodeLifecycle::kExpanded;
  node.ticket = 0;
  node.terminal = payload.terminal;
  node.edges.clear();
  node.edges.reserve(payload.moves.size());
  for (size_t i = 0; i < payload.moves.size(); ++i) {
    node.edges.push_back(
        {.move = payload.moves[i], .prior = payload.priors[i]});
  }
  assert(std::ranges::is_sorted(node.edges, PreferOnTie<Edge>));
  // The leaf value counts as the node's first visit, so a node keeps it even
  // if the visit that materialized it is cancelled before backing up.
  P::Update(node.stats, nullptr, P::Leaf(payload.leaf));
  return {node.generation, true};
}

template <typename P>
SelectResult GameGraph<P>::SelectAndReserve(NodeKey key,
                                            const typename P::Params& params) {
  auto& shard = ShardFor(key);
  std::lock_guard lock(shard.mutex);
  Node* node = FindNode(shard.nodes, key);
  if (!node) return {.status = SelectStatus::kMissing};
  if (node->lifecycle == NodeLifecycle::kMaterializing) {
    return {.status = SelectStatus::kMaterializing,
            .generation = node->generation};
  }
  if (node->terminal != TerminalKind::kNonTerminal || node->edges.empty()) {
    return {.status = SelectStatus::kTerminal,
            .generation = node->generation,
            .terminal = node->terminal};
  }
  const size_t best_index = P::Select(*node, params);
  if (best_index == node->started_end) ++node->started_end;
  Edge* best = &node->edges[best_index];
  assert(best->in_flight < kMaxActiveVisitsLimit);
  ++best->in_flight;
  return {.status = SelectStatus::kSelected,
          .move = best->move,
          .generation = node->generation};
}

template <typename P>
BackupResult<P> GameGraph<P>::BackupNode(NodeKey key,
                                         uint64_t expected_generation,
                                         Message message,
                                         std::optional<Move> move) {
  auto& shard = ShardFor(key);
  std::lock_guard lock(shard.mutex);
  Node* node = FindNode(shard.nodes, key);
  if (const auto check = CheckGeneration(node, expected_generation);
      check != UpdateResult::kApplied) {
    return {check, move ? check : UpdateResult::kApplied, std::nullopt};
  }
  Edge* edge = nullptr;
  UpdateResult edge_result = UpdateResult::kApplied;
  if (move) {
    edge = FindEdge(*node, *move);
    edge_result = ReleaseEdge(edge);
  }
  // The node is updated even if the edge reservation could not be released.
  Edge* updated = edge_result == UpdateResult::kApplied ? edge : nullptr;
  const Message to_parent =
      P::Update(node->stats, updated ? &updated->stats : nullptr, message);
  return {UpdateResult::kApplied, edge_result, to_parent};
}

template <typename P>
UpdateResult GameGraph<P>::CancelEdge(NodeKey key, uint64_t expected_generation,
                                      Move move) {
  auto& shard = ShardFor(key);
  std::lock_guard lock(shard.mutex);
  Node* node = FindNode(shard.nodes, key);
  if (const auto check = CheckGeneration(node, expected_generation);
      check != UpdateResult::kApplied) {
    return check;
  }
  return ReleaseEdge(FindEdge(*node, move));
}

template <typename P>
std::optional<NodeState<P>> GameGraph<P>::SnapshotNode(NodeKey key) const {
  const auto& shard = ShardFor(key);
  std::lock_guard lock(shard.mutex);
  const Node* node = FindNode(shard.nodes, key);
  if (!node) return std::nullopt;
  return *node;
}

template <typename P>
bool GameGraph<P>::Erase(NodeKey key) {
  auto& shard = ShardFor(key);
  std::lock_guard lock(shard.mutex);
  if (shard.nodes.erase(key) == 0) return false;
  size_.fetch_sub(1, std::memory_order_relaxed);
  return true;
}

template <typename P>
void GameGraph<P>::Clear() {
  for (auto& shard : shards_) {
    std::lock_guard lock(shard.mutex);
    // Subtract per shard to preserve concurrent insertions into shards that
    // have already been cleared.
    size_.fetch_sub(shard.nodes.size(), std::memory_order_relaxed);
    shard.nodes.clear();
  }
}

template <typename P>
std::vector<std::pair<NodeKey, NodeState<P>>>
GameGraph<P>::SnapshotAllForTesting() const {
  std::vector<std::pair<NodeKey, Node>> result;
  for (const auto& shard : shards_) {
    std::lock_guard lock(shard.mutex);
    result.insert(result.end(), shard.nodes.begin(), shard.nodes.end());
  }
  return result;
}

}  // namespace lczero::lc5
