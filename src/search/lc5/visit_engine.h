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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "chess/callbacks.h"
#include "neural/backend.h"
#include "search/lc5/graph/graph.h"
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
// materializations, batching and threads, and calls only this. Implementations
// must be thread-safe; each call locks at most one graph shard at a time.
class VisitEngine {
 public:
  virtual ~VisitEngine() = default;

  // The policy name, as in the Lc5Policy option.
  virtual std::string_view GetName() const = 0;
  // Reads the policy's options and what the backend provides; called only
  // while no search runs.
  virtual void Configure(const OptionsDict& options,
                         const BackendAttributes& backend) = 0;
  // Drops the whole graph and its store; called only while no search runs.
  virtual void Clear() = 0;

  // Selects and reserves edges down to a node to materialize
  // (kUnmaterializedLeaf) or a leaf whose value is in visit.result
  // (kValuedLeaf).
  enum class DescendResult { kUnmaterializedLeaf, kValuedLeaf };
  virtual DescendResult DescendVisit(Visit& visit, Metrics& metrics) = 0;
  // Backs visit.result up visit.path.
  virtual void BackupVisit(const Visit& visit, Metrics& metrics) = 0;
  // Releases every edge reservation on visit.path without committing a visit.
  virtual void CancelVisit(const Visit& visit) = 0;
  // A waiter's node finished materializing and its owner has backed up.
  // Selects on (kSelect, maybe from another edge at the parent), restarts from
  // the root (kRestart), or backs up visit.result (kBackup: a repetition).
  enum class ResumeAction { kSelect, kRestart, kBackup };
  virtual ResumeAction ResumeVisit(Visit& visit, Metrics& metrics) = 0;
  // Fills the score, WDL and PV of `info`; the caller fills the rest.
  virtual void FillThinkingInfo(const VisitOrigin& root,
                                ThinkingInfo& info) const = 0;
  virtual RootStats GetRootStats(NodeKey root) const = 0;

  // Node lifecycle for the materialization machinery.
  virtual bool ContainsNode(NodeKey key) const = 0;
  // Materializes the node from `eval` and counts `eval.leaf` as its first
  // visit; returns whether `eval` was installed.
  virtual bool CompleteMaterialization(NodeKey key, const NodeEval& eval) = 0;
  virtual size_t GetGraphSize() const = 0;

  // The NodeStore, the graph's cold tier: nodes evicted from the graph, kept
  // with their statistics. Set only while no search runs; Clear() clears it.
  virtual void SetStore(std::unique_ptr<NodeStore> store) = 0;
  virtual bool HasStore() const = 0;
  // The stored records of `keys`, nullopt where there is none. Needs a store.
  virtual std::vector<std::optional<std::vector<uint8_t>>> ReadFromStore(
      std::span<const NodeKey> keys) = 0;
  // Materializes the node from a stored record with its old statistics; returns
  // whether the record was installed (false if it is not this policy's, in
  // which case evaluate the node instead).
  virtual bool RestoreFromRecord(NodeKey key,
                                 std::span<const uint8_t> record) = 0;
  // Moves up to `max` nodes from the graph to the store: fewest visits first,
  // never `keep` or a node with a reserved edge. Needs a store.
  virtual void EvictToStore(size_t max, NodeKey keep, Metrics& metrics) = 0;
};

}  // namespace lczero::lc5
