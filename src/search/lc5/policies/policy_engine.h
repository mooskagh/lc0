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

#include <absl/synchronization/mutex.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "search/lc5/graph/graph.h"
#include "search/lc5/graph/node_record.h"
#include "search/lc5/settings.h"
#include "search/lc5/visit_engine.h"

namespace lczero::lc5 {

// What a policy asks for when a waiting visit's node finishes materializing:
// continue selecting below that node, reselect at its parent with the node's
// value known (possibly switching edges), or restart from the root.
enum class WaiterResume { kSelect, kReselect, kRestart };

// Search over a GameGraph it owns, with node and edge statistics, selection
// and backup supplied by the policy P (see PuctPolicy and PuctAveraging in
// puct.h for the interface).
template <typename P>
class PolicyEngine final : public VisitEngine {
 public:
  explicit PolicyEngine(typename P::Params params = {},
                        int history_key_length = 7)
      : params_(params), history_key_length_(history_key_length) {}

  std::string_view GetName() const override { return P::kName; }
  void Configure(const OptionsDict& options,
                 const BackendAttributes& backend) override {
    params_ = P::Params::FromOptions(options, backend);
    history_key_length_ = Settings(options).history_key_length();
  }
  void Clear() override {
    graph_.Clear();
    if (store_) store_->Clear();
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
    auto node = DecodeNode<P>(record);
    if (!node) return false;
    return graph_.InsertNode(key, std::move(*node));
  }
  void EvictToStore(size_t max, NodeKey keep, Metrics& metrics) override {
    assert(store_);
    const size_t size = graph_.Size();
    if (size == 0 || max == 0) return;
    const double fraction =
        std::min(1.0, static_cast<double>(max) / static_cast<double>(size));
    std::vector<std::pair<NodeKey, Node<P>>> nodes;
    {
      absl::MutexLock lock(evict_mutex_);
      nodes = graph_.FindEvictableNodes(&evict_cursor_, max, fraction);
    }
    // Store first: once a node leaves the graph, a miss must find its record.
    std::vector<StoredNode> records;
    records.reserve(nodes.size());
    for (const auto& [key, node] : nodes) {
      records.push_back({key, EncodeNode(node)});
    }
    store_->StoreBatch(records);
    metrics.node_store_evict_batches.fetch_add(1);
    // A node that changed since its record was taken stays in the graph; its
    // stale record is harmless, as the graph is always checked first.
    size_t erased = 0;
    for (const auto& [key, node] : nodes) {
      if (key != keep) erased += graph_.EraseNode(key, node.version);
    }
    metrics.nodes_evicted.fetch_add(erased);
    metrics.evictions_skipped.fetch_add(nodes.size() - erased);
  }

  GameGraph<P>& graph() { return graph_; }

 private:
  static constexpr size_t kMaxPvLength = 256;

  // Moves in the graph are from the side to move's perspective; UCI is not.
  static Move ToUci(Move move, const Position& position) {
    if (position.IsBlackToMove()) move.Flip();
    return move;
  }

  // Takes `move` from `step`, the last path step, extending visit.history and
  // visit.current_key. Returns true if that reaches an in-search repetition,
  // whose value is then in visit.result.
  bool Advance(Visit& visit, PathStep& step, Move move, Metrics& metrics);

  GameGraph<P> graph_;
  typename P::Params params_;
  // Earlier positions hashed into a NodeKey.
  int history_key_length_;
  std::unique_ptr<NodeStore> store_;
  // Where the next eviction scan starts, so evictions spread over all shards.
  absl::Mutex evict_mutex_;
  size_t evict_cursor_ ABSL_GUARDED_BY(evict_mutex_) = 0;
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
    // Below the search root, a proven node backs up its exact value. The root
    // keeps selecting, so its PV still deepens.
    const bool stop_at_proven =
        visit.path.size() > visit.origin.backup_prefix.size() + 1;
    const SelectResult selected =
        graph_.SelectAndReserveEdge(visit.current_key, params_, stop_at_proven);
    switch (selected.status) {
      case SelectResult::kNodeMissing:
        return DescendResult::kUnmaterializedLeaf;
      case SelectResult::kNodeTerminal:
        step.update_node = true;
        visit.result = TerminalValue(selected.terminal);
        visit.result_exact = IsExact(selected.terminal);
        metrics.terminal_visits.fetch_add(1);
        return DescendResult::kValuedLeaf;
      case SelectResult::kNodeProven:
        step.update_node = true;
        visit.result = selected.value;
        visit.result_exact = true;
        metrics.proven_visits.fetch_add(1);
        return DescendResult::kValuedLeaf;
      case SelectResult::kEdgeSelected:
        step.update_node = true;
        break;
    }
    if (Advance(visit, step, selected.move, metrics)) {
      return DescendResult::kValuedLeaf;
    }
  }
}

template <typename P>
bool PolicyEngine<P>::Advance(Visit& visit, PathStep& step, Move move,
                              Metrics& metrics) {
  step.selected_move = move;
  visit.history.Append(move);
  visit.current_key = MakeNodeKey(visit.history, history_key_length_);
  // A two-fold repetition whose cycle lies within this search is scored as
  // a draw. Whether that holds depends on the root, so the graph node is
  // left untouched and the leaf is backed up from a step with
  // update_node = false.
  const Position& position = visit.history.Last();
  const int ply = static_cast<int>(visit.path.size());
  const int cycle = position.GetPliesSincePrevRepetition();
  if (position.GetRepetitions() == 1 && ply >= 4 && ply >= cycle) {
    visit.path.push_back({.key = visit.current_key});
    RaiseHighWater(metrics.max_selected_depth, visit.path.size());
    visit.result = TerminalValue(TerminalKind::kRepetition);
    visit.result_exact = false;
    metrics.terminal_visits.fetch_add(1);
    metrics.two_fold_visits.fetch_add(1);
    return true;
  }
  return false;
}

template <typename P>
VisitEngine::ResumeAction PolicyEngine<P>::ResumeVisit(Visit& visit,
                                                       Metrics& metrics) {
  switch (P::OnResume(params_)) {
    case WaiterResume::kSelect:
      return ResumeAction::kSelect;
    case WaiterResume::kRestart:
      return ResumeAction::kRestart;
    case WaiterResume::kReselect:
      break;
  }
  // The waiter's node X is the search root: it has no parent in this search.
  if (visit.path.size() < visit.origin.backup_prefix.size() + 2) {
    return ResumeAction::kSelect;
  }
  PathStep& parent = visit.path[visit.path.size() - 2];
  // X's owner has already backed up, so the parent's edge holds X's value.
  const Move reselected =
      graph_.ReselectAndReserveEdge(parent.key, *parent.selected_move, params_);
  if (reselected == *parent.selected_move) {
    return ResumeAction::kSelect;
  }
  metrics.visits_reselected.fetch_add(1);
  // pop_back leaves `parent` valid.
  visit.path.pop_back();
  visit.history.Pop();
  return Advance(visit, parent, reselected, metrics) ? ResumeAction::kBackup
                                                     : ResumeAction::kSelect;
}

template <typename P>
void PolicyEngine<P>::BackupVisit(const Visit& visit, Metrics& metrics) {
  typename P::Message message = P::Leaf(visit.result, visit.result_exact);
  for (size_t i = visit.path.size(); i-- > 0;) {
    const PathStep& step = visit.path[i];
    // An in-search repetition leaf has no graph node, and an installed leaf
    // took its value at install; the parent receives the skipped-node message.
    if (!step.update_node) {
      message = P::Skip(message);
      continue;
    }
    // The leaf commits only its node value; ancestors complete their edges.
    const auto move =
        i + 1 < visit.path.size() ? step.selected_move : std::nullopt;
    const auto to_parent = graph_.UpdateNode(step.key, message, move);
    metrics.backup_node_steps.fetch_add(1);
    // A missing node is skipped.
    message = to_parent ? *to_parent : P::Stale(message);
  }
}

template <typename P>
void PolicyEngine<P>::CancelVisit(const Visit& visit) {
  for (const PathStep& step : visit.path) {
    if (step.selected_move) {
      graph_.ReleaseEdge(step.key, *step.selected_move);
    }
  }
}

template <typename P>
void PolicyEngine<P>::FillThinkingInfo(const VisitOrigin& root,
                                       ThinkingInfo& info) const {
  auto node = graph_.SnapshotNode(root.key);
  if (node) P::Describe(*node, params_, info);
  std::vector<Move>& pv = info.pv;
  pv.clear();
  PositionHistory history = root.history;
  std::unordered_set<uint64_t> seen{root.key.hash};
  while (node && pv.size() < kMaxPvLength &&
         node->terminal == TerminalKind::kNonTerminal) {
    // The root's move is the bestmove, so it is chosen even before any of its
    // edges has a visit.
    const auto index = pv.empty() ? P::ChooseMove(*node, params_)
                                  : P::ChooseEdge(*node, params_);
    if (!index) break;
    const Move move = node->edges[*index].move;
    pv.push_back(ToUci(move, history.Last()));
    history.Append(move);
    const NodeKey key = MakeNodeKey(history, history_key_length_);
    if (!seen.insert(key.hash).second) break;
    node = graph_.SnapshotNode(key);
  }
}

template <typename P>
RootStats PolicyEngine<P>::GetRootStats(NodeKey root) const {
  RootStats stats;
  const auto node = graph_.SnapshotNode(root);
  if (!node) return stats;
  stats.materialized = true;
  stats.visits = P::Visits(node->stats);
  stats.edges = node->edges.size();
  if (const auto index = P::ChooseMove(*node, params_)) {
    stats.best_move = node->edges[*index].move;
  }
  stats.decided = P::Decided(*node, params_);
  for (const auto& tried : node->tried) {
    const uint64_t visits = P::Visits(tried.stats);
    if (visits > stats.best_edge_visits) {
      stats.second_edge_visits = stats.best_edge_visits;
      stats.best_edge_visits = visits;
    } else {
      stats.second_edge_visits = std::max(stats.second_edge_visits, visits);
    }
  }
  return stats;
}

}  // namespace lczero::lc5
