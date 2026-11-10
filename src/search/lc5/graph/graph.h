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
#include <absl/synchronization/mutex.h>

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

namespace lczero::lc5 {

struct SelectResult {
  enum class Status {
    kEdgeSelected,
    kNodeMissing,
    kNodeTerminal,
    kNodeProven,
  };
  // `using enum Status;` triggers a GCC 11 ICE (tsubst_copy) in templates.
  static constexpr Status kEdgeSelected = Status::kEdgeSelected;
  static constexpr Status kNodeMissing = Status::kNodeMissing;
  static constexpr Status kNodeTerminal = Status::kNodeTerminal;
  static constexpr Status kNodeProven = Status::kNodeProven;

  Status status = kNodeMissing;
  Move move{};
  TerminalKind terminal = TerminalKind::kNonTerminal;
  VisitValue value{};  // The proven value, for kNodeProven.
};

template <typename P>
struct UpdateResult {
  // What the parent receives; set only when the node update was applied.
  std::optional<typename P::Message> to_parent;
  // Whether the node's bestmove changed; checked only when `params` is given.
  bool best_move_changed = false;
};

// Sharded map from NodeKey to materialized node search state and outgoing
// edges. Edges do not point to children; callers derive child keys from their
// own history. The policy P owns the node and edge statistics, how a message
// updates them and how an edge is selected.
template <typename P>
class GameGraph {
 public:
  using Node = lc5::Node<P>;
  using Tried = TriedEdge<P>;
  using Message = typename P::Message;

  static constexpr size_t kShardCount = 1024;

  // Selects and reserves an edge (in_flight += 1), or reports why the node
  // cannot be selected from (missing, terminal, or proven if `stop_at_proven`).
  SelectResult SelectAndReserveEdge(NodeKey key,
                                    const typename P::Params& params,
                                    bool stop_at_proven = false);

  // Releases `move` (in_flight -= 1) and selects and reserves an edge
  // (in_flight += 1); returns the selected move, or nullopt if `move` could not
  // be released.
  std::optional<Move> ReselectAndReserveEdge(NodeKey key, Move move,
                                             const typename P::Params& params);

  // Releases an edge reservation (in_flight -= 1) without committing a visit;
  // returns true if a reservation was released.
  bool ReleaseEdge(NodeKey key, Move move);

  bool ContainsNode(NodeKey key) const;

  // Inserts a new node initialized from `eval` (counting `eval.leaf` as its
  // first visit) if absent; returns true if inserted, false if already present.
  bool InsertNode(NodeKey key, const NodeEval& eval);

  // Inserts `node` (e.g. restored from the store, with in_flight reset to 0) if
  // absent; returns true if inserted, false if already present.
  bool InsertNode(NodeKey key, Node node);

  // Updates the node (and `move`'s edge with in_flight -= 1, if given), sending
  // `P::Current` when `send_current` and checking bestmove when `params`.
  UpdateResult<P> UpdateNode(NodeKey key, Message message,
                             std::optional<Move> move,
                             bool send_current = false,
                             const typename P::Params* params = nullptr);

  // Up to `max` unreserved nodes (all in_flight == 0, excluding `keep`) with
  // the fewest visits per shard (`fraction` of each), advancing `*cursor`.
  std::vector<std::pair<NodeKey, Node>> FindEvictableNodes(size_t* cursor,
                                                           size_t max,
                                                           double fraction,
                                                           NodeKey keep) const;

  // Erases the node if still at `version` with no reservations (in_flight == 0
  // on all edges); returns true if erased.
  bool EraseNode(NodeKey key, uint32_t version);

  std::optional<Node> SnapshotNode(NodeKey key) const;
  size_t Size() const { return size_.load(std::memory_order_relaxed); }
  void Clear();

 private:
  struct Shard {
    mutable absl::Mutex mutex;
    absl::flat_hash_map<NodeKey, Node> nodes ABSL_GUARDED_BY(mutex);
    uint32_t version ABSL_GUARDED_BY(mutex) = 0;
  };

  static size_t ShardIndex(NodeKey key) {
    return (key.hash * 11400714819323198485ull) >> 54;
  }
  Shard& ShardFor(NodeKey key) { return shards_[ShardIndex(key)]; }
  const Shard& ShardFor(NodeKey key) const { return shards_[ShardIndex(key)]; }
  // Reserves the edge at `index` (in_flight += 1) and returns its move.
  static Move ReserveEdgeLocked(Shard& shard, Node& node, size_t index)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(shard.mutex);

  template <typename Map>
  static auto* FindNode(Map& nodes, NodeKey key) {
    const auto it = nodes.find(key);
    return it == nodes.end() ? nullptr : &it->second;
  }

  // The index of the edge with `move`, or edges.size() if not found.
  static size_t FindEdge(const Node& node, Move move) {
    return std::ranges::find(node.edges, move, &LegalEdge::move) -
           node.edges.begin();
  }

  // Releases one reservation of the edge at `index` (in_flight -= 1); returns
  // true if released.
  static bool ReleaseEdgeLocked(Shard& shard, Node& node, size_t index)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(shard.mutex) {
    if (index >= node.tried.size() || node.tried[index].in_flight == 0) {
      return false;
    }
    --node.tried[index].in_flight;
    node.version = ++shard.version;
    return true;
  }

  // Returns true if no edge on `node` holds an in-flight reservation.
  static bool Evictable(const Node& node) {
    return std::ranges::none_of(
        node.tried, [](const Tried& tried) { return tried.in_flight > 0; });
  }

  std::array<Shard, kShardCount> shards_;
  std::atomic<size_t> size_{0};
};

template <typename P>
SelectResult GameGraph<P>::SelectAndReserveEdge(
    NodeKey key, const typename P::Params& params, bool stop_at_proven) {
  auto& shard = ShardFor(key);
  absl::MutexLock lock(shard.mutex);
  Node* node = FindNode(shard.nodes, key);
  if (!node) return {.status = SelectResult::kNodeMissing};
  if (node->terminal != TerminalKind::kNonTerminal || node->edges.empty()) {
    return {.status = SelectResult::kNodeTerminal, .terminal = node->terminal};
  }
  if (stop_at_proven) {
    if (const auto value = P::ProvenValue(*node, params)) {
      return {.status = SelectResult::kNodeProven, .value = *value};
    }
  }
  return {.status = SelectResult::kEdgeSelected,
          .move = ReserveEdgeLocked(shard, *node, P::Select(*node, params))};
}

template <typename P>
std::optional<Move> GameGraph<P>::ReselectAndReserveEdge(
    NodeKey key, Move move, const typename P::Params& params) {
  auto& shard = ShardFor(key);
  absl::MutexLock lock(shard.mutex);
  Node* node = FindNode(shard.nodes, key);
  if (!node) return std::nullopt;
  if (!ReleaseEdgeLocked(shard, *node, FindEdge(*node, move))) {
    return std::nullopt;
  }
  return ReserveEdgeLocked(shard, *node, P::Select(*node, params));
}

template <typename P>
Move GameGraph<P>::ReserveEdgeLocked(Shard& shard, Node& node, size_t index) {
  if (index == node.tried.size()) {
    node.tried.emplace_back();
  }
  Tried& tried = node.tried[index];
  assert(tried.in_flight < kMaxActiveVisitsLimit);
  ++tried.in_flight;
  node.version = ++shard.version;
  return node.edges[index].move;
}

template <typename P>
bool GameGraph<P>::ReleaseEdge(NodeKey key, Move move) {
  auto& shard = ShardFor(key);
  absl::MutexLock lock(shard.mutex);
  Node* node = FindNode(shard.nodes, key);
  return node && ReleaseEdgeLocked(shard, *node, FindEdge(*node, move));
}

template <typename P>
bool GameGraph<P>::ContainsNode(NodeKey key) const {
  const auto& shard = ShardFor(key);
  absl::MutexLock lock(shard.mutex);
  return shard.nodes.contains(key);
}

template <typename P>
bool GameGraph<P>::InsertNode(NodeKey key, const NodeEval& eval) {
  assert(eval.moves.size() == eval.priors.size());
  assert(eval.terminal == TerminalKind::kNonTerminal || eval.moves.empty());
  auto& shard = ShardFor(key);
  absl::MutexLock lock(shard.mutex);
  // Late or duplicate completions must not reset materialized edges, including
  // their reservations and statistics.
  auto [it, inserted] = shard.nodes.try_emplace(key);
  if (!inserted) return false;
  size_.fetch_add(1, std::memory_order_relaxed);
  Node& node = it->second;
  node.version = ++shard.version;
  node.terminal = eval.terminal;
  node.edges.reserve(eval.moves.size());
  for (size_t i = 0; i < eval.moves.size(); ++i) {
    node.edges.push_back({.move = eval.moves[i], .prior = eval.priors[i]});
  }
  assert(std::ranges::is_sorted(node.edges, PreferOnTie));
  // The leaf value counts as the node's first visit, so a node keeps it even
  // if the visit that materialized it is cancelled before backing up.
  P::Update(node, nullptr, P::Leaf(eval.leaf, IsExact(eval.terminal)));
  return true;
}

template <typename P>
bool GameGraph<P>::InsertNode(NodeKey key, Node restored) {
  auto& shard = ShardFor(key);
  absl::MutexLock lock(shard.mutex);
  if (shard.nodes.contains(key)) return false;
  restored.version = ++shard.version;
  for (auto& tried : restored.tried) tried.in_flight = 0;
  shard.nodes.emplace(key, std::move(restored));
  size_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

template <typename P>
UpdateResult<P> GameGraph<P>::UpdateNode(NodeKey key, Message message,
                                         std::optional<Move> move,
                                         bool send_current,
                                         const typename P::Params* params) {
  auto& shard = ShardFor(key);
  absl::MutexLock lock(shard.mutex);
  Node* node = FindNode(shard.nodes, key);
  if (!node) return {};
  const auto best_before =
      params ? P::ChooseMove(*node, *params) : std::nullopt;
  Tried* edge = nullptr;
  if (move) {
    const size_t index = FindEdge(*node, *move);
    if (ReleaseEdgeLocked(shard, *node, index)) edge = &node->tried[index];
  }
  // The node is updated even if the edge reservation could not be released.
  node->version = ++shard.version;
  Message to_parent = P::Update(*node, edge, message);
  if (send_current) to_parent = P::Current(*node);
  const bool best_changed =
      params && P::ChooseMove(*node, *params) != best_before;
  return {to_parent, best_changed};
}

template <typename P>
std::vector<std::pair<NodeKey, Node<P>>> GameGraph<P>::FindEvictableNodes(
    size_t* cursor, size_t max, double fraction, NodeKey keep) const {
  std::vector<std::pair<NodeKey, Node>> result;
  std::vector<std::pair<uint64_t, const std::pair<const NodeKey, Node>*>>
      evictable;
  for (size_t scanned = 0; scanned < kShardCount && result.size() < max;
       ++scanned) {
    const Shard& shard = shards_[*cursor];
    *cursor = (*cursor + 1) % kShardCount;
    absl::MutexLock lock(shard.mutex);
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
bool GameGraph<P>::EraseNode(NodeKey key, uint32_t version) {
  auto& shard = ShardFor(key);
  absl::MutexLock lock(shard.mutex);
  const Node* node = FindNode(shard.nodes, key);
  if (!node || node->version != version || !Evictable(*node)) {
    return false;
  }
  shard.nodes.erase(key);
  size_.fetch_sub(1, std::memory_order_relaxed);
  return true;
}

template <typename P>
std::optional<Node<P>> GameGraph<P>::SnapshotNode(NodeKey key) const {
  const auto& shard = ShardFor(key);
  absl::MutexLock lock(shard.mutex);
  const Node* node = FindNode(shard.nodes, key);
  if (!node) return std::nullopt;
  return *node;
}

template <typename P>
void GameGraph<P>::Clear() {
  for (auto& shard : shards_) {
    absl::MutexLock lock(shard.mutex);
    // Subtract per shard to preserve concurrent insertions into shards that
    // have already been cleared.
    size_.fetch_sub(shard.nodes.size(), std::memory_order_relaxed);
    shard.nodes.clear();
  }
}

}  // namespace lczero::lc5
