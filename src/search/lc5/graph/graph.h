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

#include <absl/algorithm/container.h>
#include <absl/container/flat_hash_map.h>
#include <absl/synchronization/mutex.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "search/lc5/graph/key.h"
#include "search/lc5/graph/node.h"
#include "utils/map_util.h"

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

// Sharded map from NodeKey to materialized node search state and outgoing
// edges. Edges do not point to children; callers derive child keys from their
// own history. The policy P owns the node and edge statistics, how a backup
// updates them and how an edge is selected.
template <typename P>
class GameGraph {
 public:
  using Node = lc5::Node<P>;
  using Tried = TriedEdge<P>;
  using BackupValue = typename P::BackupValue;

  static constexpr size_t kShardCount = 1024;
  static constexpr size_t kEvictWindow = 4096;
  static_assert(kShardCount > 1 && std::has_single_bit(kShardCount));

  // Position in a rolling eviction scan: the next shard to visit, and for each
  // shard the key where its last window stopped (a key from another shard means
  // begin()).
  struct EvictCursor {
    size_t shard = 0;
    std::array<NodeKey, kShardCount> next_keys = {NodeKey{1}};
  };

  size_t Size() const { return size_.load(std::memory_order_relaxed); }
  bool ContainsNode(NodeKey key) const;
  std::optional<Node> SnapshotNode(NodeKey key) const;
  void Clear();

  // Inserts a new node initialized from `eval` (counting `eval.leaf` as its
  // first visit) if absent; returns true if inserted, false if already present.
  bool InsertNode(NodeKey key, const NodeEval& eval);

  // Inserts `node` (e.g. restored from the store, with in_flight reset to 0) if
  // absent; returns true if inserted, false if already present.
  bool InsertNode(NodeKey key, Node node);

  // Selects and reserves an edge (in_flight += 1), or reports why the node
  // cannot be selected from (missing, terminal, or proven if `stop_at_proven`).
  SelectResult SelectAndReserveEdge(NodeKey key,
                                    const typename P::Params& params,
                                    bool stop_at_proven = false);

  // Releases `move` (in_flight -= 1) and selects and reserves an edge
  // (in_flight += 1); returns the selected move.
  Move ReselectAndReserveEdge(NodeKey key, Move move,
                              const typename P::Params& params);

  // Updates the node (and `move`'s edge with in_flight -= 1, if given); returns
  // the backup value for the parent, or nullopt if the node is absent.
  std::optional<BackupValue> BackupThroughNode(NodeKey key, BackupValue backup,
                                               std::optional<Move> move);

  // Releases an edge reservation (in_flight -= 1) without committing a visit.
  void ReleaseEdge(NodeKey key, Move move);

  // Up to `max` unreserved nodes (in_flight == 0) with the fewest visits per
  // shard window (`fraction` of each), advancing `*cursor`.
  std::vector<std::pair<NodeKey, Node>> FindEvictableNodes(
      EvictCursor* cursor, size_t max, double fraction) const;

  // Erases the node if still at `version` with no reservations (in_flight ==
  // 0); returns true if erased.
  bool EraseNode(NodeKey key, uint32_t version);

 private:
  using Map = absl::flat_hash_map<NodeKey, Node>;

  struct Shard {
    mutable absl::Mutex mutex;
    Map nodes ABSL_GUARDED_BY(mutex);
    uint32_t version ABSL_GUARDED_BY(mutex) = 0;
  };

  static constexpr size_t ShardIndex(NodeKey key) {
    return (key.hash * 11400714819323198485ull) >>
           (64 - std::countr_zero(kShardCount));
  }

  // A key that never hashes to `shard`, marking the start of its map.
  static constexpr NodeKey StartOfShard(size_t shard) {
    return NodeKey{shard == 0 ? 1u : 0u};
  }
  static_assert(ShardIndex(NodeKey{0}) == 0 && ShardIndex(NodeKey{1}) != 0);

  Shard& ShardFor(NodeKey key) { return shards_[ShardIndex(key)]; }
  const Shard& ShardFor(NodeKey key) const { return shards_[ShardIndex(key)]; }

  // The index of the edge with `move`, or edges.size() if not found.
  static size_t GetEdgeIndex(const Node& node, Move move) {
    return absl::c_find_if(
               node.edges,
               [move](const LegalEdge& edge) { return edge.move == move; }) -
           node.edges.begin();
  }

  // Reserves the edge at `index` (in_flight += 1) and returns its move.
  static Move ReserveEdgeLocked(Shard& shard, Node& node, size_t index)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(shard.mutex) {
    if (index == node.tried.size()) node.tried.emplace_back();
    Tried& tried = node.tried[index];
    assert(tried.in_flight < kMaxActiveVisitsLimit);
    ++tried.in_flight;
    ++node.in_flight;
    node.version = ++shard.version;
    return node.edges[index].move;
  }

  // Releases one reservation of the edge at `index` (in_flight -= 1).
  static void ReleaseEdgeLocked(Shard& shard, Node& node, size_t index)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(shard.mutex) {
    assert(index < node.tried.size());
    assert(node.tried[index].in_flight > 0);
    assert(node.in_flight > 0);
    --node.tried[index].in_flight;
    --node.in_flight;
    node.version = ++shard.version;
  }

  std::array<Shard, kShardCount> shards_;
  std::atomic<size_t> size_{0};
};

template <typename P>
bool GameGraph<P>::ContainsNode(NodeKey key) const {
  const auto& shard = ShardFor(key);
  absl::MutexLock lock(shard.mutex);
  return shard.nodes.contains(key);
}

template <typename P>
std::optional<Node<P>> GameGraph<P>::SnapshotNode(NodeKey key) const {
  const auto& shard = ShardFor(key);
  absl::MutexLock lock(shard.mutex);
  const Node* node = FindOrNull(shard.nodes, key);
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

template <typename P>
bool GameGraph<P>::InsertNode(NodeKey key, const NodeEval& eval) {
  assert(eval.moves.size() == eval.priors.size());
  assert(eval.terminal == TerminalKind::kNonTerminal || eval.moves.empty());
  auto& shard = ShardFor(key);
  absl::MutexLock lock(shard.mutex);
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
  assert(absl::c_is_sorted(node.edges, PreferOnTie));
  P::InitNodeStats(node, eval);
  return true;
}

template <typename P>
bool GameGraph<P>::InsertNode(NodeKey key, Node restored) {
  auto& shard = ShardFor(key);
  absl::MutexLock lock(shard.mutex);
  if (shard.nodes.contains(key)) return false;
  restored.version = ++shard.version;
  restored.in_flight = 0;
  for (auto& tried : restored.tried) tried.in_flight = 0;
  shard.nodes.emplace(key, std::move(restored));
  size_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

template <typename P>
SelectResult GameGraph<P>::SelectAndReserveEdge(
    NodeKey key, const typename P::Params& params, bool stop_at_proven) {
  auto& shard = ShardFor(key);
  absl::MutexLock lock(shard.mutex);
  Node* node = FindOrNull(shard.nodes, key);
  if (!node) return {.status = SelectResult::kNodeMissing};
  if (node->terminal != TerminalKind::kNonTerminal || node->edges.empty()) {
    return {.status = SelectResult::kNodeTerminal, .terminal = node->terminal};
  }
  if (stop_at_proven) {
    if (const auto value = P::GetProvenNodeValue(*node, params)) {
      return {.status = SelectResult::kNodeProven, .value = *value};
    }
  }
  return {.status = SelectResult::kEdgeSelected,
          .move = ReserveEdgeLocked(shard, *node,
                                    P::PickEdgeToVisit(*node, params))};
}

template <typename P>
Move GameGraph<P>::ReselectAndReserveEdge(NodeKey key, Move move,
                                          const typename P::Params& params) {
  auto& shard = ShardFor(key);
  absl::MutexLock lock(shard.mutex);
  Node& node = FindOrAssert(shard.nodes, key);
  ReleaseEdgeLocked(shard, node, GetEdgeIndex(node, move));
  return ReserveEdgeLocked(shard, node, P::PickEdgeToVisit(node, params));
}

template <typename P>
std::optional<typename P::BackupValue> GameGraph<P>::BackupThroughNode(
    NodeKey key, BackupValue backup, std::optional<Move> move) {
  auto& shard = ShardFor(key);
  absl::MutexLock lock(shard.mutex);
  Node* node = FindOrNull(shard.nodes, key);
  if (!node) {
    assert(!move);
    return std::nullopt;
  }
  Tried* edge = nullptr;
  if (move) {
    const size_t index = GetEdgeIndex(*node, *move);
    ReleaseEdgeLocked(shard, *node, index);
    edge = &node->tried[index];
  }
  node->version = ++shard.version;
  return P::BackupThroughNode(*node, edge, backup);
}

template <typename P>
void GameGraph<P>::ReleaseEdge(NodeKey key, Move move) {
  auto& shard = ShardFor(key);
  absl::MutexLock lock(shard.mutex);
  Node& node = FindOrAssert(shard.nodes, key);
  ReleaseEdgeLocked(shard, node, GetEdgeIndex(node, move));
}

template <typename P>
std::vector<std::pair<NodeKey, Node<P>>> GameGraph<P>::FindEvictableNodes(
    EvictCursor* cursor, size_t max, double fraction) const {
  std::vector<std::pair<NodeKey, Node>> result;
  std::vector<std::pair<uint64_t, const std::pair<const NodeKey, Node>*>>
      evictable;
  evictable.reserve(kEvictWindow);
  for (size_t scanned = 0; scanned < kShardCount && result.size() < max;
       ++scanned) {
    const size_t shard_index = cursor->shard;
    cursor->shard = (shard_index + 1) % kShardCount;
    const Shard& shard = shards_[shard_index];
    absl::MutexLock lock(shard.mutex);
    if (shard.nodes.empty()) continue;

    const NodeKey resume = cursor->next_keys[shard_index];
    auto it = ShardIndex(resume) == shard_index ? shard.nodes.find(resume)
                                                : shard.nodes.end();
    if (it == shard.nodes.end()) it = shard.nodes.begin();

    const size_t remaining = max - result.size();
    evictable.clear();
    size_t window = 0;
    while (window < kEvictWindow && it != shard.nodes.end() &&
           (evictable.size() < remaining ||
            fraction * static_cast<double>(window) <
                static_cast<double>(remaining))) {
      if (it->second.in_flight == 0) {
        evictable.emplace_back(it->second.stats.visits, &*it);
      }
      ++window;
      ++it;
    }
    cursor->next_keys[shard_index] =
        it == shard.nodes.end() ? StartOfShard(shard_index) : it->first;
    if (evictable.empty()) continue;

    const size_t wanted = std::max<size_t>(
        1, static_cast<size_t>(fraction * static_cast<double>(window) + 0.5));
    const size_t take = std::min({wanted, evictable.size(), remaining});
    absl::c_nth_element(
        evictable, evictable.begin() + take,
        [](const auto& a, const auto& b) { return a.first < b.first; });
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
  const Node* node = FindOrNull(shard.nodes, key);
  if (!node || node->version != version || node->in_flight > 0) return false;
  shard.nodes.erase(key);
  size_.fetch_sub(1, std::memory_order_relaxed);
  return true;
}

}  // namespace lczero::lc5
