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

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "search/lc5/key.h"
#include "search/lc5/node_store.h"
#include "search/lc5/settings.h"
#include "search/lc5/value.h"

namespace lczero::lc5 {

using MaterializationTicketId = uint64_t;

enum class NodeLifecycle : uint8_t { kMaterializing, kExpanded };

struct EdgeState {
  Move move;
  float prior = 0.0f;
  uint32_t in_flight = 0;  // Virtual loss.
  ValueStats value{};
  float Q() const { return value.Q(); }
};

// Deterministic tie-break between equally ranked edges: higher prior first,
// then the lower move encoding.
inline bool PreferOnTie(const EdgeState& a, const EdgeState& b) {
  if (a.prior != b.prior) return a.prior > b.prior;
  return a.move.raw_data() < b.move.raw_data();
}

struct NodeState {
  // Unique per (re)creation, so updates for an erased node are rejected.
  uint64_t generation = 0;
  NodeLifecycle lifecycle = NodeLifecycle::kMaterializing;
  MaterializationTicketId ticket = 0;  // Zero once expanded.
  TerminalKind terminal = TerminalKind::kNonTerminal;
  ValueStats value{};
  std::vector<EdgeState> edges{};
};

// Lightweight snapshot without edges or value stats, used during selection.
struct NodeMetadata {
  uint64_t generation;
  NodeLifecycle lifecycle;
  TerminalKind terminal;
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

enum class SelectStatus {
  kSelected,
  kMissing,
  kStale,
  kMaterializing,
  kTerminal
};

struct SelectResult {
  SelectStatus status = SelectStatus::kMissing;
  Move move{};
  uint64_t generation = 0;
};

enum class UpdateResult {
  kApplied,
  kMissing,
  kStale,
  kEdgeMissing,
  kUnderflow
};

struct BackupResult {
  UpdateResult node;
  UpdateResult edge;
};

// Sharded map from NodeKey to node search state and outgoing edges. Edges do
// not point to children; callers derive child keys from their own history.
// Every mutation names the node generation it expects, so updates computed
// against an erased and recreated node are rejected instead of misapplied.
class GameGraph {
 public:
  static constexpr size_t kShardCount = 1024;

  // Finds an existing node or creates a kMaterializing one owned by `ticket`.
  FindOrCreateResult FindOrCreateMaterializing(NodeKey key,
                                               MaterializationTicketId ticket);

  // Erases only the matching materializing placeholder, never an expansion.
  bool CancelMaterialization(NodeKey key, MaterializationTicketId ticket);

  // Expands the node with `payload` if it is still materializing under
  // `ticket`, creating it if absent. Publication is one-shot per generation.
  InstallResult InstallPayload(NodeKey key, MaterializationTicketId ticket,
                               const ExpansionPayload& payload);

  // Selects the best edge by PUCT and reserves it (in_flight += 1).
  SelectResult SelectAndReserve(NodeKey key, uint64_t expected_generation,
                                const Settings::Resolved& settings);

  // Adds `value` to the node and, if `move` is given, completes that edge's
  // reservation with the same value.
  BackupResult BackupNode(NodeKey key, uint64_t expected_generation,
                          SearchValue value, std::optional<Move> move);

  // Releases an edge reservation without committing a visit.
  UpdateResult CancelEdge(NodeKey key, uint64_t expected_generation, Move move);

  std::optional<NodeMetadata> SnapshotNodeMetadata(NodeKey key) const;
  std::optional<NodeState> SnapshotNode(NodeKey key) const;

  bool Erase(NodeKey key);
  void Clear();
  size_t Size() const { return size_.load(std::memory_order_relaxed); }
  std::vector<std::pair<NodeKey, NodeState>> SnapshotAllForTesting() const;

 private:
  struct Shard {
    mutable std::mutex mutex;
    absl::flat_hash_map<NodeKey, NodeState> nodes;
  };

  static size_t ShardIndex(NodeKey key) {
    return (key.hash * 11400714819323198485ull) >> 54;
  }
  Shard& ShardFor(NodeKey key) { return shards_[ShardIndex(key)]; }
  const Shard& ShardFor(NodeKey key) const { return shards_[ShardIndex(key)]; }
  // Inserts a fresh materializing node; the shard must be locked.
  NodeState& InsertLocked(Shard& shard, NodeKey key,
                          MaterializationTicketId ticket);

  std::array<Shard, kShardCount> shards_;
  std::atomic<size_t> size_{0};
  std::atomic<uint64_t> next_generation_{1};
};

}  // namespace lczero::lc5
