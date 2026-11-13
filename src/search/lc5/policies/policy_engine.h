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

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "search/lc5/graph/graph.h"
#include "search/lc5/policies/node_serialization.h"
#include "search/lc5/settings.h"
#include "search/lc5/visit_engine.h"

namespace lczero::lc5 {

// VisitEngine backed by a GameGraph<P>, delegating node/edge statistics,
// selection, and backup to policy P (see PuctPolicy in puct.h).
template <typename P>
class PolicyEngine final : public VisitEngine {
 public:
  std::string_view GetName() const override { return P::kName; }
  void Configure(const OptionsDict& options,
                 const BackendAttributes& backend) override {
    params_ = P::Params::FromOptions(options, backend);
    history_key_length_ = Settings(options).history_key_length();
  }
  void Clear() override {
    graph_.Clear();
    if (store_) store_->Clear();
    root_remaining_visits_.store(std::numeric_limits<uint64_t>::max(),
                                 std::memory_order_relaxed);
    absl::MutexLock lock(evict_mutex_);
    evict_cursor_ = {};
  }
  void SetRootRemainingVisits(uint64_t remaining_visits) override {
    root_remaining_visits_.store(remaining_visits, std::memory_order_relaxed);
  }

  DescendResult DescendVisit(Visit& visit, Metrics& metrics) override;
  void BackupVisit(const Visit& visit, Metrics& metrics) override;
  void CancelVisit(const Visit& visit) override;
  ResumeAction ResumeVisit(Visit& visit, Metrics& metrics) override;
  void FillThinkingInfo(const VisitOrigin& root,
                        ThinkingInfo& info) const override;
  RootStats GetRootStats(NodeKey root) const override;

  bool ContainsNode(NodeKey key) const override {
    return graph_.ContainsNode(key);
  }
  bool CompleteMaterialization(NodeKey key, const NodeEval& eval) override {
    return graph_.InsertNode(key, eval);
  }
  size_t GetGraphSize() const override { return graph_.Size(); }

  void SetStore(std::unique_ptr<NodeStore> store) override {
    store_ = std::move(store);
  }
  bool HasStore() const override { return store_ != nullptr; }
  std::vector<std::optional<std::vector<uint8_t>>> ReadFromStore(
      std::span<const NodeKey> keys) override {
    assert(store_);
    return store_->LoadBatch(keys);
  }
  bool RestoreFromRecord(NodeKey key,
                         std::span<const uint8_t> record) override {
    return graph_.InsertNode(key, DeserializeNode<P>(record));
  }
  size_t EvictToStore(size_t max, NodeKey keep, Metrics& metrics) override;

 private:
  static constexpr size_t kMaxPvLength = 256;
  static constexpr double kMinEvictFraction = 0.1;

  static bool IsBelowOrigin(const Visit& visit) {
    return visit.path.size() > visit.origin.backup_prefix.size() + 1;
  }

  // Follows `move` from the current node; returns kBackup if that reaches an
  // in-search two-fold repetition (setting `visit.result`), else kDescend.
  ResumeAction FollowEdge(Visit& visit, Move move, Metrics& metrics);

  GameGraph<P> graph_;
  typename P::Params params_{};
  int history_key_length_ = 7;  // Preceding positions hashed into a NodeKey.
  std::atomic<uint64_t> root_remaining_visits_{
      std::numeric_limits<uint64_t>::max()};
  std::unique_ptr<NodeStore> store_;
  absl::Mutex evict_mutex_;
  // Rolling cursor so eviction scans spread across all shards.
  typename GameGraph<P>::EvictCursor evict_cursor_
      ABSL_GUARDED_BY(evict_mutex_);
};

template <typename P>
VisitEngine::DescendResult PolicyEngine<P>::DescendVisit(Visit& visit,
                                                         Metrics& metrics) {
  for (;;) {
    if (visit.path.empty() || visit.path.back().key != visit.current_key) {
      visit.path.push_back({.key = visit.current_key});
    }
    PathStep& step = visit.path.back();
    metrics.selection_node_steps.fetch_add(1);
    RaiseHighWater(metrics.max_selected_depth, visit.path.size());
    // Keep selecting at the origin even when proven so its PV still deepens.
    const bool below_origin = IsBelowOrigin(visit);
    const uint64_t remaining_root_visits =
        below_origin
            ? std::numeric_limits<uint64_t>::max()
            : root_remaining_visits_.load(std::memory_order_relaxed);
    const SelectResult selection = graph_.SelectAndReserveEdge(
        visit.current_key, params_, /*stop_at_proven=*/below_origin,
        remaining_root_visits);
    step.update_node = selection.status != SelectResult::kNodeMissing;
    switch (selection.status) {
      case SelectResult::kNodeMissing:
        return DescendResult::kUnmaterializedLeaf;
      case SelectResult::kNodeTerminal:
        visit.result = TerminalValue(selection.terminal);
        visit.result_exact = IsExact(selection.terminal);
        metrics.terminal_visits.fetch_add(1);
        return DescendResult::kValuedLeaf;
      case SelectResult::kNodeProven:
        visit.result = selection.value;
        visit.result_exact = true;
        metrics.proven_visits.fetch_add(1);
        return DescendResult::kValuedLeaf;
      case SelectResult::kEdgeSelected:
        break;
    }
    if (FollowEdge(visit, selection.move, metrics) == ResumeAction::kBackup) {
      return DescendResult::kValuedLeaf;
    }
  }
}

template <typename P>
void PolicyEngine<P>::BackupVisit(const Visit& visit, Metrics& metrics) {
  assert(visit.path.empty() || !visit.path.back().selected_move);
  auto backup = P::MakeLeafBackup(visit.result, visit.result_exact);
  for (auto it = visit.path.rbegin(); it != visit.path.rend(); ++it) {
    // Skip repetition leaves and leaves already counted at installation.
    if (!it->update_node) {
      backup = P::MakeSkipBackup(backup);
      continue;
    }
    const auto parent_backup =
        graph_.BackupThroughNode(it->key, backup, it->selected_move);
    metrics.backup_node_steps.fetch_add(1);
    backup = parent_backup.value_or(P::MakeStaleBackup(backup));
  }
}

template <typename P>
void PolicyEngine<P>::CancelVisit(const Visit& visit) {
  for (const PathStep& step : visit.path) {
    if (step.selected_move) graph_.ReleaseEdge(step.key, *step.selected_move);
  }
}

template <typename P>
VisitEngine::ResumeAction PolicyEngine<P>::ResumeVisit(Visit& visit,
                                                       Metrics& metrics) {
  // At the origin there is no parent edge to reselect.
  if (!IsBelowOrigin(visit)) return ResumeAction::kDescend;
  const PathStep& parent = visit.path[visit.path.size() - 2];
  const bool parent_is_origin =
      visit.path.size() == visit.origin.backup_prefix.size() + 2;
  const uint64_t remaining_root_visits =
      parent_is_origin
          ? root_remaining_visits_.load(std::memory_order_relaxed)
          : std::numeric_limits<uint64_t>::max();
  // The leaf's owner has already backed up, so the parent edge holds its value.
  const Move reselected = graph_.ReselectAndReserveEdge(
      parent.key, *parent.selected_move, params_, remaining_root_visits);
  if (reselected == *parent.selected_move) return ResumeAction::kDescend;
  metrics.visits_reselected.fetch_add(1);
  visit.path.pop_back();
  visit.history.Pop();
  return FollowEdge(visit, reselected, metrics);
}

template <typename P>
void PolicyEngine<P>::FillThinkingInfo(const VisitOrigin& root,
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
RootStats PolicyEngine<P>::GetRootStats(NodeKey root) const {
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
size_t PolicyEngine<P>::EvictToStore(size_t max, NodeKey keep,
                                     Metrics& metrics) {
  assert(store_);
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
  std::vector<StoredNode> records;
  records.reserve(candidates.size());
  for (const auto& [key, node] : candidates) {
    records.push_back({key, SerializeNode(node)});
  }
  store_->StoreBatch(records);
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
VisitEngine::ResumeAction PolicyEngine<P>::FollowEdge(Visit& visit, Move move,
                                                      Metrics& metrics) {
  visit.path.back().selected_move = move;
  visit.history.Append(move);
  visit.current_key = MakeNodeKey(visit.history, history_key_length_);
  // Score an in-search two-fold repetition as a draw without touching its graph
  // node, since repetition depends on the search path rather than the node.
  const Position& pos = visit.history.Last();
  const int search_plies = static_cast<int>(visit.path.size());
  if (pos.GetRepetitions() == 1 && search_plies >= 4 &&
      search_plies >= pos.GetPliesSincePrevRepetition()) {
    visit.path.push_back({.key = visit.current_key});
    RaiseHighWater(metrics.max_selected_depth, visit.path.size());
    visit.result = TerminalValue(TerminalKind::kRepetition);
    visit.result_exact = false;
    metrics.terminal_visits.fetch_add(1);
    metrics.two_fold_visits.fetch_add(1);
    return ResumeAction::kBackup;
  }
  return ResumeAction::kDescend;
}

}  // namespace lczero::lc5
