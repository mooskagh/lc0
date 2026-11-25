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
#include <string_view>

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

  // The policy name, as in the Lc5Policy option.
  virtual std::string_view GetName() const = 0;
  // Reads the policy's options and what the backend provides; called only
  // while no search runs.
  virtual void Configure(const OptionsDict& options,
                         const BackendAttributes& backend) = 0;
  // Drops the whole graph; called only while no search runs.
  virtual void Clear() = 0;

  // Visit lifecycle:
  // - DescendVisit steps down the graph until it backs up an in-graph or
  //   repetition leaf (kCompleted, releasing all reservations) or pauses at a
  //   missing node (kUnmaterializedLeaf -> BackupLeaf, CancelVisit, or another
  //   DescendVisit once the node is in the graph).
  // - With `reselect_parent` set (a waiter waking after the owner backed up a
  //   newly evaluated leaf), DescendVisit first re-runs selection at the parent
  //   edge before descending.
  enum class DescendResult { kUnmaterializedLeaf, kCompleted };
  virtual DescendResult DescendVisit(Visit& visit, Metrics& metrics,
                                     bool reselect_parent) = 0;
  // Backs up `value` for a visit paused at kUnmaterializedLeaf and releases its
  // reservations. `count_at_leaf` is false when InsertNode already counted
  // `value` as the leaf's first visit (the owner), and true for terminal
  // waiters.
  virtual void BackupLeaf(const Visit& visit, const VisitValue& value,
                          Metrics& metrics, TerminalKind terminal,
                          bool count_at_leaf) = 0;
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
