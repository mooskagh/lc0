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

#include <absl/container/flat_hash_set.h>
#include <absl/synchronization/mutex.h>
#include <absl/types/span.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "search/lc5/graph/graph.h"
#include "search/lc5/policies/node_serialization.h"
#include "search/lc5/search_algorithm.h"
#include "search/lc5/settings.h"

namespace lczero::lc5 {

// SearchAlgorithm backed by a GameGraph<P>, delegating node/edge statistics,
// selection, and backup to policy P (see PuctPolicy in puct.h).
template <typename P>
class PolicyAlgorithm final : public SearchAlgorithm {
 public:
  void Configure(const OptionsDict& options,
                 const BackendAttributes& backend) override {
    params_ = P::Params::FromOptions(options, backend);
    history_key_length_ = Settings(options).history_key_length();
  }
  void Clear() override {
    graph_.Clear();
    absl::MutexLock lock(evict_mutex_);
    evict_cursor_ = {};
  }

  RunResult RunVisit(Visit& visit, Metrics& metrics,
                     bool reselect_at_parent) override;
  void BackupVisit(const Visit& visit, const VisitValue& value,
                   Metrics& metrics, TerminalKind terminal,
                   bool count_at_leaf) override;
  void CancelVisit(const Visit& visit) override;
  void FillThinkingInfo(const VisitOrigin& root,
                        ThinkingInfo& info) const override;
  RootStats GetRootStats(NodeKey root) const override;

  size_t GetGraphSize() const override { return graph_.Size(); }
  bool ContainsNode(NodeKey key) const override {
    return graph_.ContainsNode(key);
  }
  void PrepareRoot(NodeKey root) override { graph_.EraseDrawByRuleNode(root); }
  bool InsertNode(NodeKey key, const NodeEval& eval) override {
    return graph_.InsertNode(key, eval);
  }
  bool RestoreNode(NodeKey key, absl::Span<const uint8_t> record,
                   bool is_root) override {
    Node<P> node = DeserializeNode<P>(record);
    if (is_root && IsDrawByRule(node.terminal)) return false;
    return graph_.InsertNode(key, std::move(node));
  }
  size_t EvictToStore(NodeStore& store, size_t max, NodeKey keep,
                      Metrics& metrics) override;

 private:
  static constexpr size_t kMaxPvLength = 256;
  static constexpr double kMinEvictFraction = 0.1;

  static bool IsBelowOrigin(const Visit& visit) { return !visit.path.empty(); }

  // Records `move` on the path, advances `visit` across it, and returns true if
  // that reaches and backs up an in-search two-fold repetition.
  bool FollowEdge(Visit& visit, Move move, Metrics& metrics);
  void Backup(const Visit& visit, const VisitValue& value, bool exact,
              bool update_leaf, Metrics& metrics);

  GameGraph<P> graph_;
  typename P::Params params_{};
  int history_key_length_ = 7;  // Preceding positions hashed into a NodeKey.
  absl::Mutex evict_mutex_;
  // Rolling cursor so eviction scans spread across all shards.
  typename GameGraph<P>::EvictCursor evict_cursor_
      ABSL_GUARDED_BY(evict_mutex_);
};

template <typename P>
SearchAlgorithm::RunResult PolicyAlgorithm<P>::RunVisit(
    Visit& visit, Metrics& metrics, bool reselect_at_parent) {
  // At the origin there is no parent edge to reselect.
  if (reselect_at_parent && IsBelowOrigin(visit)) {
    const PathStep parent = visit.path.back();
    // The leaf's owner has already backed up, so the parent edge holds its
    // value.
    const Move reselected =
        graph_.ReselectAndReserveEdge(parent.key, parent.move, params_);
    if (reselected != parent.move) {
      metrics.visits_reselected.fetch_add(1);
      visit.current_key = parent.key;
      visit.path.pop_back();
      visit.history.Pop();
      if (FollowEdge(visit, reselected, metrics)) {
        return RunResult::kCompleted;
      }
    }
  }
  for (;;) {
    metrics.selection_node_steps.fetch_add(1);
    RaiseHighWater(metrics.max_selected_depth, visit.path.size() + 1);
    // Keep selecting at the origin even when proven so its PV still deepens.
    const SelectResult selection = graph_.SelectAndReserveEdge(
        visit.current_key, params_, /*stop_at_proven=*/IsBelowOrigin(visit));
    switch (selection.status) {
      case SelectResult::kNodeMissing:
        return RunResult::kNeedsMaterialization;
      case SelectResult::kNodeTerminal:
        metrics.terminal_visits.fetch_add(1);
        Backup(visit, TerminalValue(selection.terminal),
               IsExact(selection.terminal), /*update_leaf=*/true, metrics);
        return RunResult::kCompleted;
      case SelectResult::kNodeProven:
        metrics.proven_visits.fetch_add(1);
        Backup(visit, selection.value, /*exact=*/true, /*update_leaf=*/true,
               metrics);
        return RunResult::kCompleted;
      case SelectResult::kEdgeSelected:
        break;
    }
    if (FollowEdge(visit, selection.move, metrics)) {
      return RunResult::kCompleted;
    }
  }
}

template <typename P>
void PolicyAlgorithm<P>::BackupVisit(const Visit& visit,
                                     const VisitValue& value, Metrics& metrics,
                                     TerminalKind terminal,
                                     bool count_at_leaf) {
  if (terminal != TerminalKind::kNonTerminal) {
    metrics.terminal_visits.fetch_add(1);
  }
  Backup(visit, value, IsExact(terminal), /*update_leaf=*/count_at_leaf,
         metrics);
}

template <typename P>
void PolicyAlgorithm<P>::CancelVisit(const Visit& visit) {
  for (const PathStep& step : visit.path) {
    graph_.ReleaseEdge(step.key, step.move);
  }
}

template <typename P>
void PolicyAlgorithm<P>::FillThinkingInfo(const VisitOrigin& root,
                                          ThinkingInfo& info) const {
  info.pv.clear();
  auto node = graph_.SnapshotNode(root.key);
  if (!node) return;
  P::FillThinkingInfo(*node, params_, info);
  PositionHistory history = root.history;
  absl::flat_hash_set<NodeKey> seen{root.key};
  while (node && info.pv.size() < kMaxPvLength &&
         node->terminal == TerminalKind::kNonTerminal && !node->edges.empty()) {
    const Move move = P::GetBestMove(*node, params_);
    Move uci_move = move;
    // Graph moves are from the side-to-move's perspective; UCI is not.
    if (history.IsBlackToMove()) uci_move.Flip();
    info.pv.push_back(uci_move);
    history.Append(move);
    const NodeKey key = MakeNodeKey(history, history_key_length_);
    if (!seen.insert(key).second) break;
    node = graph_.SnapshotNode(key);
  }
}

template <typename P>
RootStats PolicyAlgorithm<P>::GetRootStats(NodeKey root) const {
  const auto node = graph_.SnapshotNode(root);
  if (!node) return {};
  RootStats stats{
      .materialized = true,
      .visits = node->stats.visits,
      .edges = node->edges.size(),
      .best_move =
          node->edges.empty() ? Move{} : P::GetBestMove(*node, params_),
      .decided = P::IsNodeDecided(*node, params_),
  };
  for (const auto& tried : node->tried) {
    const uint64_t visits = tried.stats.visits;
    if (visits > stats.best_edge_visits) {
      stats.second_edge_visits = std::exchange(stats.best_edge_visits, visits);
    } else {
      stats.second_edge_visits = std::max(stats.second_edge_visits, visits);
    }
  }
  return stats;
}

template <typename P>
size_t PolicyAlgorithm<P>::EvictToStore(NodeStore& store, size_t max,
                                        NodeKey keep, Metrics& metrics) {
  const size_t size = graph_.Size();
  if (size == 0 || max == 0) return 0;
  const double fraction =
      std::clamp(static_cast<double>(max) / size, kMinEvictFraction, 1.0);
  std::vector<std::pair<NodeKey, Node<P>>> candidates;
  {
    absl::MutexLock lock(evict_mutex_);
    candidates = graph_.FindEvictableNodes(&evict_cursor_, max, fraction);
  }
  std::erase_if(candidates,
                [keep](const auto& entry) { return entry.first == keep; });
  if (candidates.empty()) return 0;

  // Store before erasing so a concurrent miss always finds the record.
  size_t total_bytes = 0;
  for (const auto& [key, node] : candidates) {
    total_bytes += SerializedNodeSize(node);
  }
  std::vector<uint8_t> buffer(total_bytes);
  std::vector<StoredNode> records;
  records.reserve(candidates.size());
  size_t offset = 0;
  for (const auto& [key, node] : candidates) {
    const size_t record_size = SerializedNodeSize(node);
    const absl::Span<uint8_t> record =
        absl::MakeSpan(buffer).subspan(offset, record_size);
    SerializeNode(node, record);
    records.push_back({.key = key, .record = record});
    offset += record_size;
  }
  store.StoreBatch(records);
  metrics.node_store_evict_batches.fetch_add(1);

  // Modified nodes stay in the graph; stale store records are harmless because
  // the graph is checked first.
  size_t erased = 0;
  for (const auto& [key, node] : candidates) {
    if (graph_.EraseNode(key, node.version)) ++erased;
  }
  metrics.nodes_evicted.fetch_add(erased);
  metrics.evictions_skipped.fetch_add(candidates.size() - erased);
  return erased;
}

template <typename P>
bool PolicyAlgorithm<P>::FollowEdge(Visit& visit, Move move, Metrics& metrics) {
  visit.path.push_back({.key = visit.current_key, .move = move});
  visit.history.Append(move);
  visit.current_key = MakeNodeKey(visit.history, history_key_length_);
  // Score an in-search two-fold repetition as a draw without touching its graph
  // node, since repetition depends on the search path rather than the node.
  const Position& pos = visit.history.Last();
  const int search_plies = static_cast<int>(visit.path.size());
  if (pos.GetRepetitions() == 1 && search_plies >= 4 &&
      search_plies >= pos.GetPliesSincePrevRepetition()) {
    RaiseHighWater(metrics.max_selected_depth, visit.path.size() + 1);
    metrics.terminal_visits.fetch_add(1);
    metrics.two_fold_visits.fetch_add(1);
    Backup(visit, TerminalValue(TerminalKind::kRepetition), /*exact=*/false,
           /*update_leaf=*/false, metrics);
    return true;
  }
  return false;
}

template <typename P>
void PolicyAlgorithm<P>::Backup(const Visit& visit, const VisitValue& value,
                                bool exact, bool update_leaf,
                                Metrics& metrics) {
  RaiseHighWater(metrics.max_depth, visit.path.size() + 1);
  auto backup = P::MakeLeafBackup(value, exact);
  if (update_leaf) {
    const auto parent_backup = graph_.BackupLeaf(visit.current_key, backup);
    metrics.backup_node_steps.fetch_add(1);
    backup = parent_backup.value_or(P::MakeStaleBackup(backup));
  } else {
    // Skip repetition leaves and leaves already counted at installation.
    backup = P::MakeSkipBackup(backup);
  }
  for (auto it = visit.path.rbegin(); it != visit.path.rend(); ++it) {
    backup = graph_.BackupThroughEdge(it->key, it->move, backup);
    metrics.backup_node_steps.fetch_add(1);
  }
}

}  // namespace lczero::lc5
