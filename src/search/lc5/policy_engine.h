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
  void Configure(const OptionsDict& options,
                 const BackendAttributes& backend) override {
    params_ = P::Params::FromOptions(options, backend);
  }
  void Clear() override { graph_.Clear(); }

  DescendResult Descend(Visit& visit, const VisitContext& ctx) override;
  void Backup(const Visit& visit, const VisitContext& ctx) override;
  void Cancel(const Visit& visit, const VisitContext& ctx) override;
  ResumeAction OnResume(Visit& visit, const VisitContext& ctx) override;
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

  // Takes `move` from `step`, the last path step, extending visit.history and
  // visit.current_key. Returns true if that reaches an in-search repetition,
  // whose value is then in visit.result.
  bool Advance(Visit& visit, PathStep& step, Move move,
               const VisitContext& ctx);

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
    // Below the search root, a proven node backs up its exact value. The root
    // keeps selecting, so its PV still deepens.
    const bool stop_at_proven =
        visit.path.size() > visit.origin.backup_prefix.size() + 1;
    const SelectResult selected =
        graph_.SelectAndReserve(visit.current_key, params_, stop_at_proven);
    switch (selected.status) {
      case SelectStatus::kMissing:
      case SelectStatus::kMaterializing:
        return DescendResult::kMaterialize;
      case SelectStatus::kTerminal:
        step.generation = selected.generation;
        visit.result = TerminalValue(selected.terminal);
        visit.result_exact = IsExact(selected.terminal);
        ctx.metrics.terminal_visits.fetch_add(1);
        return DescendResult::kBackup;
      case SelectStatus::kProven:
        step.generation = selected.generation;
        visit.result = selected.value;
        visit.result_exact = true;
        ctx.metrics.proven_visits.fetch_add(1);
        return DescendResult::kBackup;
      case SelectStatus::kSelected:
        step.generation = selected.generation;
        break;
    }
    if (Advance(visit, step, selected.move, ctx)) {
      return DescendResult::kBackup;
    }
  }
}

template <typename P>
bool PolicyEngine<P>::Advance(Visit& visit, PathStep& step, Move move,
                              const VisitContext& ctx) {
  step.selected_move = move;
  visit.history.Append(move);
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
    visit.result_exact = false;
    ctx.metrics.terminal_visits.fetch_add(1);
    return true;
  }
  return false;
}

template <typename P>
VisitEngine::ResumeAction PolicyEngine<P>::OnResume(Visit& visit,
                                                    const VisitContext& ctx) {
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
  // X's owner has not backed up yet, so the parent's edge does not show X's
  // value; it is what the parent receives from X, whose own update was
  // counted at install.
  const auto from_child = P::Skip(P::Leaf(visit.result, visit.result_exact));
  const SelectResult reselected =
      graph_.ReselectAndReserve(parent.key, parent.generation,
                                *parent.selected_move, from_child, params_);
  if (reselected.status != SelectStatus::kSelected ||
      reselected.move == *parent.selected_move) {
    return ResumeAction::kSelect;
  }
  ctx.metrics.visits_reselected.fetch_add(1);
  // pop_back leaves `parent` valid.
  visit.path.pop_back();
  visit.history.Pop();
  return Advance(visit, parent, reselected.move, ctx) ? ResumeAction::kBackup
                                                      : ResumeAction::kSelect;
}

template <typename P>
void PolicyEngine<P>::Backup(const Visit& visit, const VisitContext& ctx) {
  typename P::Message message = P::Leaf(visit.result, visit.result_exact);
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
    message = result.to_parent ? *result.to_parent : P::Stale(message);
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
  if (node) P::Describe(*node, params_, info);
  std::vector<Move>& pv = info.pv;
  pv.clear();
  PositionHistory history = root.history;
  std::unordered_set<uint64_t> seen{root.key.hash};
  while (node && pv.size() < kMaxPvLength &&
         node->lifecycle == NodeLifecycle::kExpanded &&
         node->terminal == TerminalKind::kNonTerminal) {
    // The root's move is the bestmove, so it is chosen even before any of its
    // edges has a visit.
    const auto index = pv.empty() ? P::ChooseMove(*node, params_)
                                  : P::ChooseEdge(*node, params_);
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
