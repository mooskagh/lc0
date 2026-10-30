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

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "search/lc5/graph.h"
#include "search/lc5/visit_engine.h"

namespace lczero::lc5 {

// Search over a GameGraph it owns, with node and edge statistics, selection
// and backup supplied by the policy P (see PuctAveraging for the interface).
template <typename P>
class PolicyEngine final : public VisitEngine {
 public:
  explicit PolicyEngine(typename P::Params params = {}) : params_(params) {}

  std::string_view name() const override { return P::kName; }
  void Configure(const OptionsDict& options) override {
    params_ = P::Params::FromOptions(options);
  }
  void Clear() override { graph_.Clear(); }

  DescendResult Descend(Visit& visit, const VisitContext& ctx) override;
  void Backup(const Visit& visit, const VisitContext& ctx) override;
  void Cancel(const Visit& visit, const VisitContext& ctx) override;
  ResumeAction OnResume(const Visit&, const VisitContext&) override {
    return P::OnResume(params_);
  }
  void Report(const VisitOrigin& root, const VisitContext& ctx,
              ThinkingInfo& info) const override;
  RootStats Root(NodeKey root) const override;

  FindOrCreateResult FindOrCreateMaterializing(
      NodeKey key, MaterializationTicketId ticket) override {
    return graph_.FindOrCreateMaterializing(key, ticket);
  }
  InstallResult Install(NodeKey key, MaterializationTicketId ticket,
                        const ExpansionPayload& payload) override {
    return graph_.InstallPayload(key, ticket, payload);
  }
  void CancelMaterialization(NodeKey key,
                             MaterializationTicketId ticket) override {
    graph_.CancelMaterialization(key, ticket);
  }
  size_t GraphSize() const override { return graph_.Size(); }

  GameGraph<P>& graph() { return graph_; }

 private:
  static constexpr size_t kMaxPvLength = 256;

  // Moves in the graph are from the side to move's perspective; UCI is not.
  static Move ToUci(Move move, const Position& position) {
    if (position.IsBlackToMove()) move.Flip();
    return move;
  }

  GameGraph<P> graph_;
  typename P::Params params_;
};

template <typename P>
VisitEngine::DescendResult PolicyEngine<P>::Descend(Visit& visit,
                                                    const VisitContext& ctx) {
  for (;;) {
    if (visit.path.empty() || visit.path.back().key != visit.current_key) {
      visit.path.push_back({.key = visit.current_key});
    }
    PathStep& step = visit.path.back();
    ctx.metrics.selection_node_steps.fetch_add(1);
    RaiseHighWater(ctx.metrics.max_selected_depth, visit.path.size());
    const SelectResult selected =
        graph_.SelectAndReserve(visit.current_key, params_);
    switch (selected.status) {
      case SelectStatus::kMissing:
      case SelectStatus::kMaterializing:
        return DescendResult::kMaterialize;
      case SelectStatus::kTerminal:
        step.generation = selected.generation;
        visit.result = TerminalValue(selected.terminal);
        ctx.metrics.terminal_visits.fetch_add(1);
        return DescendResult::kBackup;
      case SelectStatus::kSelected:
        step.generation = selected.generation;
        break;
    }
    step.selected_move = selected.move;
    visit.history.Append(selected.move);
    visit.current_key =
        MakeNodeKey(visit.history, ctx.settings.history_key_length);
    // A two-fold repetition whose cycle lies within this search is scored as
    // a draw. Whether that holds depends on the root, so the graph node is
    // left untouched and the leaf is backed up from a step that is not in the
    // graph (generation 0).
    const Position& position = visit.history.Last();
    const int ply = static_cast<int>(visit.path.size());
    if (position.GetRepetitions() == 1 && ply >= 4 &&
        ply >= position.GetPliesSincePrevRepetition()) {
      visit.path.push_back({.key = visit.current_key});
      visit.result = TerminalValue(TerminalKind::kRepetition);
      ctx.metrics.terminal_visits.fetch_add(1);
      return DescendResult::kBackup;
    }
  }
}

template <typename P>
void PolicyEngine<P>::Backup(const Visit& visit, const VisitContext& ctx) {
  typename P::Message message = P::Leaf(visit.result);
  for (size_t i = visit.path.size(); i-- > 0;) {
    const PathStep& step = visit.path[i];
    const bool leaf = i + 1 == visit.path.size();
    // An in-search repetition leaf has no graph node, and a counted leaf took
    // its value at install; the parent receives the skipped-node message.
    if (step.generation == 0 || (leaf && visit.leaf_counted)) {
      message = P::Skip(message);
      continue;
    }
    // The leaf commits only its node value; ancestors complete their edges.
    const auto move =
        i + 1 < visit.path.size() ? step.selected_move : std::nullopt;
    const BackupResult<P> result =
        graph_.BackupNode(step.key, step.generation, message, move);
    ctx.metrics.backup_node_steps.fetch_add(1);
    if (result.node != UpdateResult::kApplied) {
      ctx.metrics.stale_generation_node_updates.fetch_add(1);
    }
    if (result.edge == UpdateResult::kUnderflow) {
      ctx.metrics.invariant_underflow_prevented.fetch_add(1);
    } else if (result.edge != UpdateResult::kApplied) {
      ctx.metrics.stale_generation_edge_updates.fetch_add(1);
    }
    // A stale or missing node is skipped.
    message = result.to_parent ? *result.to_parent : P::Skip(message);
  }
}

template <typename P>
void PolicyEngine<P>::Cancel(const Visit& visit, const VisitContext& ctx) {
  for (const PathStep& step : visit.path) {
    if (step.selected_move &&
        graph_.CancelEdge(step.key, step.generation, *step.selected_move) ==
            UpdateResult::kUnderflow) {
      ctx.metrics.invariant_underflow_prevented.fetch_add(1);
    }
  }
}

template <typename P>
void PolicyEngine<P>::Report(const VisitOrigin& root, const VisitContext& ctx,
                             ThinkingInfo& info) const {
  auto node = graph_.SnapshotNode(root.key);
  if (node) P::Describe(node->stats, info);
  std::vector<Move>& pv = info.pv;
  pv.clear();
  PositionHistory history = root.history;
  std::unordered_set<uint64_t> seen{root.key.hash};
  while (node && pv.size() < kMaxPvLength &&
         node->lifecycle == NodeLifecycle::kExpanded &&
         node->terminal == TerminalKind::kNonTerminal) {
    // The root's move is the bestmove, so it is chosen even before any of its
    // edges has a visit.
    const auto index = pv.empty() ? P::ChooseMove(*node) : P::ChooseEdge(*node);
    if (!index) break;
    const Move move = node->edges[*index].move;
    pv.push_back(ToUci(move, history.Last()));
    history.Append(move);
    const NodeKey key = MakeNodeKey(history, ctx.settings.history_key_length);
    if (!seen.insert(key.hash).second) break;
    node = graph_.SnapshotNode(key);
  }
}

template <typename P>
RootStats PolicyEngine<P>::Root(NodeKey root) const {
  RootStats stats;
  const auto node = graph_.SnapshotNode(root);
  if (!node) return stats;
  stats.expanded = node->lifecycle == NodeLifecycle::kExpanded;
  stats.visits = P::Visits(node->stats);
  stats.edges = node->edges.size();
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
