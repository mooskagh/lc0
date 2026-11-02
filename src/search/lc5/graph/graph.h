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
#include <optional>
#include <utility>
#include <vector>

#include "search/lc5/graph/key.h"
#include "search/lc5/graph/node.h"
#include "search/lc5/sync.h"

namespace lczero::lc5 {

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

enum class SelectStatus {
  kSelected,
  kMissing,
  kMaterializing,
  kTerminal,
  kProven
};

struct SelectResult {
  SelectStatus status = SelectStatus::kMissing;
  Move move{};
  uint64_t generation = 0;
  TerminalKind terminal = TerminalKind::kNonTerminal;
  SearchValue value{};  // The proven value, for kProven.
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
  using Tried = TriedEdge<P>;
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

  // Like InstallPayload, but the node takes the statistics, edges and terminal
  // kind of `restored` (an evicted node) instead of counting a leaf. The node
  // keeps the generation of its materializing placeholder, which is new, so
  // updates for the evicted node stay rejected.
  InstallResult RestoreNode(NodeKey key, MaterializationTicketId ticket,
                            Node restored);

  // Selects an edge by the policy and reserves it (in_flight += 1); otherwise
  // reports why the node can't be selected from. Every result but kMissing
  // carries the node's generation, and kTerminal also its terminal kind. With
  // `stop_at_proven`, a node the policy has proven returns kProven with its
  // value instead.
  SelectResult SelectAndReserve(NodeKey key, const typename P::Params& params,
                                bool stop_at_proven = false);

  // Releases one reservation of `move` and reserves the edge the policy
  // reselects as if that visit had completed with `from_child`. kMissing if
  // the node is not `expected_generation` or `move` holds no reservation;
  // nothing is changed then.
  SelectResult ReselectAndReserve(NodeKey key, uint64_t expected_generation,
                                  Move move, const Message& from_child,
                                  const typename P::Params& params);

  // Applies the policy's update to the node and, if `move` is given,
  // completes that edge's reservation. On success `to_parent` is the message
  // the parent receives: the update's, or with `send_current` the node's
  // statistics after the update (P::Current).
  BackupResult<P> BackupNode(NodeKey key, uint64_t expected_generation,
                             Message message, std::optional<Move> move,
                             bool send_current = false);

  // Releases an edge reservation without committing a visit.
  UpdateResult CancelEdge(NodeKey key, uint64_t expected_generation, Move move);

  std::optional<Node> SnapshotNode(NodeKey key) const;

  // Copies of up to `max` evictable nodes: expanded, not `keep`, with no edge
  // reserved. Scans shards from `*cursor` on, at most all of them once,
  // and leaves the cursor after the last one scanned. From each shard it takes
  // the `fraction` of its nodes with the fewest visits (at least one node).
  std::vector<std::pair<NodeKey, Node>> EvictCandidates(size_t* cursor,
                                                        size_t max,
                                                        double fraction,
                                                        NodeKey keep) const;
  // Erases the node if it still is `generation` at `version` and has no
  // reservations, as when it was copied.
  bool EraseIfUnchanged(NodeKey key, uint64_t generation, uint32_t version);
  // Calls fn(key, node) for every expanded node, with its shard locked.
  template <typename F>
  void ForEachExpanded(F&& fn) const;

  bool Erase(NodeKey key);
  void Clear();
  size_t Size() const { return size_.load(std::memory_order_relaxed); }
  std::vector<std::pair<NodeKey, Node>> SnapshotAllForTesting() const;

 private:
  struct Shard {
    mutable absl::Mutex mutex;
    absl::flat_hash_map<NodeKey, Node> nodes ABSL_GUARDED_BY(mutex);
  };

  static size_t ShardIndex(NodeKey key) {
    return (key.hash * 11400714819323198485ull) >> 54;
  }
  Shard& ShardFor(NodeKey key) { return shards_[ShardIndex(key)]; }
  const Shard& ShardFor(NodeKey key) const { return shards_[ShardIndex(key)]; }
  // Inserts a fresh materializing node.
  Node& InsertLocked(Shard& shard, NodeKey key, MaterializationTicketId ticket)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(shard.mutex);
  // Reserves the edge at `index` (in_flight += 1), which is tried or the first
  // untried one, and returns its move; the shard must be locked.
  static Move ReserveLocked(Node& node, size_t index);

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

  // The index of the edge with `move`, or edges.size() if there is none.
  static size_t FindEdge(const Node& node, Move move) {
    return std::ranges::find(node.edges, move, &LegalEdge::move) -
           node.edges.begin();
  }

  // Releases one reservation of the edge at `index`. An edge that was never
  // tried has no reservations.
  static UpdateResult ReleaseEdge(Node& node, size_t index) {
    if (index >= node.edges.size()) return UpdateResult::kEdgeMissing;
    if (index >= node.tried.size() || node.tried[index].in_flight == 0) {
      return UpdateResult::kUnderflow;
    }
    --node.tried[index].in_flight;
    ++node.version;
    return UpdateResult::kApplied;
  }

  // Whether the node can leave the graph: expanded, and no visit holds one
  // of its edges.
  static bool Evictable(const Node& node) {
    return node.lifecycle == NodeLifecycle::kExpanded &&
           std::ranges::none_of(node.tried, [](const Tried& tried) {
             return tried.in_flight > 0;
           });
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
  MutexLock lock(shard.mutex);
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
  MutexLock lock(shard.mutex);
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
  MutexLock lock(shard.mutex);
  Node* existing = FindNode(shard.nodes, key);
  Node& node = existing ? *existing : InsertLocked(shard, key, ticket);
  // Late or duplicate completions must not reset expanded edges, including
  // their reservations and statistics.
  if (node.lifecycle == NodeLifecycle::kExpanded || node.ticket != ticket) {
    return {node.generation, false};
  }
  node.lifecycle = NodeLifecycle::kExpanded;
  node.ticket = 0;
  ++node.version;
  node.terminal = payload.terminal;
  node.edges.clear();
  node.edges.reserve(payload.moves.size());
  for (size_t i = 0; i < payload.moves.size(); ++i) {
    node.edges.push_back(
        {.move = payload.moves[i], .prior = payload.priors[i]});
  }
  assert(std::ranges::is_sorted(node.edges, PreferOnTie));
  node.tried.clear();
  // The leaf value counts as the node's first visit, so a node keeps it even
  // if the visit that materialized it is cancelled before backing up.
  P::Update(node, nullptr, P::Leaf(payload.leaf, IsExact(payload.terminal)));
  return {node.generation, true};
}

template <typename P>
InstallResult GameGraph<P>::RestoreNode(NodeKey key,
                                        MaterializationTicketId ticket,
                                        Node restored) {
  auto& shard = ShardFor(key);
  MutexLock lock(shard.mutex);
  Node* existing = FindNode(shard.nodes, key);
  Node& node = existing ? *existing : InsertLocked(shard, key, ticket);
  if (node.lifecycle == NodeLifecycle::kExpanded || node.ticket != ticket) {
    return {node.generation, false};
  }
  const uint64_t generation = node.generation;
  const uint32_t version = node.version + 1;
  node = std::move(restored);
  node.generation = generation;
  node.version = version;
  node.lifecycle = NodeLifecycle::kExpanded;
  node.ticket = 0;
  for (auto& tried : node.tried) tried.in_flight = 0;
  return {generation, true};
}

template <typename P>
SelectResult GameGraph<P>::SelectAndReserve(NodeKey key,
                                            const typename P::Params& params,
                                            bool stop_at_proven) {
  auto& shard = ShardFor(key);
  MutexLock lock(shard.mutex);
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
  if (stop_at_proven) {
    if (const auto value = P::ProvenValue(*node, params)) {
      return {.status = SelectStatus::kProven,
              .generation = node->generation,
              .value = *value};
    }
  }
  return {.status = SelectStatus::kSelected,
          .move = ReserveLocked(*node, P::Select(*node, params)),
          .generation = node->generation};
}

template <typename P>
SelectResult GameGraph<P>::ReselectAndReserve(
    NodeKey key, uint64_t expected_generation, Move move,
    const Message& from_child, const typename P::Params& params) {
  auto& shard = ShardFor(key);
  MutexLock lock(shard.mutex);
  Node* node = FindNode(shard.nodes, key);
  if (CheckGeneration(node, expected_generation) != UpdateResult::kApplied) {
    return {.status = SelectStatus::kMissing};
  }
  const size_t index = FindEdge(*node, move);
  if (ReleaseEdge(*node, index) != UpdateResult::kApplied) {
    return {.status = SelectStatus::kMissing};
  }
  return {.status = SelectStatus::kSelected,
          .move = ReserveLocked(*node,
                                P::Reselect(*node, params, index, from_child)),
          .generation = node->generation};
}

template <typename P>
Move GameGraph<P>::ReserveLocked(Node& node, size_t index) {
  if (index == node.tried.size()) {
    // Grows 2 -> 4 -> 8...; most nodes only ever try a few edges.
    if (node.tried.empty()) node.tried.reserve(2);
    node.tried.emplace_back();
  }
  Tried& tried = node.tried[index];
  assert(tried.in_flight < kMaxActiveVisitsLimit);
  ++tried.in_flight;
  ++node.version;
  return node.edges[index].move;
}

template <typename P>
BackupResult<P> GameGraph<P>::BackupNode(NodeKey key,
                                         uint64_t expected_generation,
                                         Message message,
                                         std::optional<Move> move,
                                         bool send_current) {
  auto& shard = ShardFor(key);
  MutexLock lock(shard.mutex);
  Node* node = FindNode(shard.nodes, key);
  if (const auto check = CheckGeneration(node, expected_generation);
      check != UpdateResult::kApplied) {
    return {check, move ? check : UpdateResult::kApplied, std::nullopt};
  }
  Tried* edge = nullptr;
  UpdateResult edge_result = UpdateResult::kApplied;
  if (move) {
    const size_t index = FindEdge(*node, *move);
    edge_result = ReleaseEdge(*node, index);
    if (edge_result == UpdateResult::kApplied) edge = &node->tried[index];
  }
  // The node is updated even if the edge reservation could not be released.
  ++node->version;
  Message to_parent = P::Update(*node, edge, message);
  if (send_current) to_parent = P::Current(*node);
  return {UpdateResult::kApplied, edge_result, to_parent};
}

template <typename P>
UpdateResult GameGraph<P>::CancelEdge(NodeKey key, uint64_t expected_generation,
                                      Move move) {
  auto& shard = ShardFor(key);
  MutexLock lock(shard.mutex);
  Node* node = FindNode(shard.nodes, key);
  if (const auto check = CheckGeneration(node, expected_generation);
      check != UpdateResult::kApplied) {
    return check;
  }
  return ReleaseEdge(*node, FindEdge(*node, move));
}

template <typename P>
std::optional<NodeState<P>> GameGraph<P>::SnapshotNode(NodeKey key) const {
  const auto& shard = ShardFor(key);
  MutexLock lock(shard.mutex);
  const Node* node = FindNode(shard.nodes, key);
  if (!node) return std::nullopt;
  return *node;
}

template <typename P>
std::vector<std::pair<NodeKey, NodeState<P>>> GameGraph<P>::EvictCandidates(
    size_t* cursor, size_t max, double fraction, NodeKey keep) const {
  std::vector<std::pair<NodeKey, Node>> result;
  std::vector<std::pair<uint64_t, const std::pair<const NodeKey, Node>*>>
      evictable;
  for (size_t scanned = 0; scanned < kShardCount && result.size() < max;
       ++scanned) {
    const Shard& shard = shards_[*cursor];
    *cursor = (*cursor + 1) % kShardCount;
    MutexLock lock(shard.mutex);
    evictable.clear();
    for (const auto& entry : shard.nodes) {
      if (entry.first == keep || !Evictable(entry.second)) continue;
      evictable.emplace_back(P::Visits(entry.second.stats), &entry);
    }
    const size_t wanted = std::max<size_t>(
        1, static_cast<size_t>(
               fraction * static_cast<double>(shard.nodes.size()) + 0.5));
    const size_t take =
        std::min({wanted, evictable.size(), max - result.size()});
    std::ranges::nth_element(evictable, evictable.begin() + take, {},
                             &decltype(evictable)::value_type::first);
    for (size_t i = 0; i < take; ++i) {
      result.push_back(*evictable[i].second);
    }
  }
  return result;
}

template <typename P>
bool GameGraph<P>::EraseIfUnchanged(NodeKey key, uint64_t generation,
                                    uint32_t version) {
  auto& shard = ShardFor(key);
  MutexLock lock(shard.mutex);
  const Node* node = FindNode(shard.nodes, key);
  if (!node || node->generation != generation || node->version != version ||
      !Evictable(*node)) {
    return false;
  }
  shard.nodes.erase(key);
  size_.fetch_sub(1, std::memory_order_relaxed);
  return true;
}

template <typename P>
template <typename F>
void GameGraph<P>::ForEachExpanded(F&& fn) const {
  for (const auto& shard : shards_) {
    MutexLock lock(shard.mutex);
    for (const auto& [key, node] : shard.nodes) {
      if (node.lifecycle == NodeLifecycle::kExpanded) fn(key, node);
    }
  }
}

template <typename P>
bool GameGraph<P>::Erase(NodeKey key) {
  auto& shard = ShardFor(key);
  MutexLock lock(shard.mutex);
  if (shard.nodes.erase(key) == 0) return false;
  size_.fetch_sub(1, std::memory_order_relaxed);
  return true;
}

template <typename P>
void GameGraph<P>::Clear() {
  for (auto& shard : shards_) {
    MutexLock lock(shard.mutex);
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
    MutexLock lock(shard.mutex);
    result.insert(result.end(), shard.nodes.begin(), shard.nodes.end());
  }
  return result;
}

}  // namespace lczero::lc5
