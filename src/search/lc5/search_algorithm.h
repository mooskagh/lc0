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

#include <absl/types/span.h>

#include <cstddef>
#include <cstdint>

#include "chess/callbacks.h"
#include "neural/backend.h"
#include "search/lc5/graph/key.h"
#include "search/lc5/graph/node_eval.h"
#include "search/lc5/metrics.h"
#include "search/lc5/store/node_store.h"
#include "search/lc5/visit.h"
#include "utils/optionsdict.h"

namespace lczero::lc5 {

// Root numbers the controller needs for limits, info output and early stopping.
struct RootStats {
  bool materialized = false;
  uint64_t visits = 0;  // Node visits; 0 if the root is absent.
  size_t edges = 0;
  uint64_t best_edge_visits = 0;
  uint64_t second_edge_visits = 0;
  Move best_move{};
  // The root's result is proven, so the search can stop.
  bool decided = false;
};

// The search-algorithm side of a visit: everything that reads or writes node
// and edge statistics, and owns the graph. SearchSession owns visits,
// materializations, batching, store threads, and evaluators. Implementations
// must be thread-safe; each call locks at most one graph shard at a time.
class SearchAlgorithm {
 public:
  virtual ~SearchAlgorithm() = default;

  // Reads the policy's options and what the backend provides; called only
  // while no search runs.
  virtual void Configure(const OptionsDict& options,
                         const BackendAttributes& backend) = 0;
  // Drops the whole graph; called only while no search runs.
  virtual void Clear() = 0;

  // Runs `visit` all the way through the graph (reserving one outgoing edge
  // per node) until it either finishes completely or hits a missing node:
  // - kCompleted: the visit is completely finished — it hit an in-graph
  //   terminal, proven node, or repetition, backed up all the way to the root,
  //   and released all edge reservations.
  // - kNeedsMaterialization: paused at `visit.current_key` because that node
  //   is not in the graph yet. Edge reservations are still held and no backup
  //   has happened; the caller must materialize the node and then finish the
  //   visit via BackupVisit, another RunVisit, or CancelVisit.
  //
  // If `reselect_at_parent` is true (and `visit` is below the root), re-runs
  // move selection at the parent node before descending.
  enum class RunResult { kCompleted, kNeedsMaterialization };
  virtual RunResult RunVisit(Visit& visit, Metrics& metrics,
                             bool reselect_at_parent) = 0;
  // Backs up `value` all the way from `visit.current_key` to the root for a
  // visit paused at kNeedsMaterialization whose leaf was just inserted by
  // InsertNode (so `value` is already counted at the leaf), releasing all edge
  // reservations.
  virtual void BackupVisit(const Visit& visit, const VisitValue& value,
                           Metrics& metrics, TerminalKind terminal) = 0;
  // Releases every edge reservation on visit.path without committing a visit.
  virtual void CancelVisit(const Visit& visit) = 0;

  // Fills the score, WDL and PV of `info`; the caller fills the rest.
  virtual void FillThinkingInfo(const VisitOrigin& root,
                                ThinkingInfo& info) const = 0;
  virtual RootStats GetRootStats(NodeKey root) const = 0;

  // Graph node lifecycle and cold-tier serialization.
  virtual size_t GetGraphSize() const = 0;
  virtual bool ContainsNode(NodeKey key) const = 0;
  // Drops a draw-by-rule terminal stub at `root`, if present, so a search at
  // `root` expands its legal moves; called only while no search runs.
  virtual void PrepareRoot(NodeKey root) = 0;
  // Installs a node from `eval` and counts `eval.leaf` as its first visit;
  // returns false if the node was already in the graph.
  virtual bool InsertNode(NodeKey key, const NodeEval& eval) = 0;
  // Installs a node serialized by EvictToStore with its old statistics; returns
  // false if rejected (already present, or a draw-by-rule stub when `is_root`),
  // in which case evaluate it instead.
  virtual bool RestoreNode(NodeKey key, absl::Span<const uint8_t> record,
                           bool is_root) = 0;
  // Moves up to `max` unreserved nodes (fewest visits first, never `keep`) from
  // the graph to `store`; returns how many were evicted.
  virtual size_t EvictToStore(NodeStore& store, size_t max, NodeKey keep,
                              Metrics& metrics) = 0;
};

}  // namespace lczero::lc5
