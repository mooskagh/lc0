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

#include "search/lc5/policy_engine.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

namespace lczero::lc5 {
namespace {

constexpr size_t kMaxPvLength = 256;

// Moves in the graph are from the side to move's perspective; UCI is not.
Move ToUci(Move move, const Position& position) {
  if (position.IsBlackToMove()) move.Flip();
  return move;
}

// The PV edge: most visits, then best Q, then the usual tie-break.
const EdgeState* PvEdge(const NodeState& node) {
  const EdgeState* best = nullptr;
  for (const auto& edge : node.edges) {
    if (edge.value.visits == 0) continue;
    const auto rank = [](const EdgeState& e) {
      return std::pair(e.value.visits, e.Q());
    };
    if (!best || rank(edge) > rank(*best) ||
        (rank(edge) == rank(*best) && PreferOnTie(edge, *best))) {
      best = &edge;
    }
  }
  return best;
}

ThinkingInfo::WDL WdlFromValue(float q, float d) {
  d = std::clamp(d, 0.0f, 1.0f);
  const int draw = static_cast<int>(d * 1000.0f);
  const int win = static_cast<int>((1.0f - d + q) * 500.0f);
  return {std::clamp(win, 0, 1000), draw,
          std::clamp(1000 - win - draw, 0, 1000)};
}

int CentipawnsFromQ(float q) {
  return static_cast<int>(90 * std::tan(1.5637541897 * q));
}

}  // namespace

VisitEngine::DescendResult PolicyEngine::Descend(Visit& visit,
                                                 const VisitContext& ctx) {
  for (;;) {
    if (visit.path.empty() || visit.path.back().key != visit.current_key) {
      visit.path.push_back({.key = visit.current_key});
    }
    PathStep& step = visit.path.back();
    ctx.metrics.selection_node_steps.fetch_add(1);
    RaiseHighWater(ctx.metrics.max_selected_depth, visit.path.size());
    const SelectResult selected =
        graph_->SelectAndReserve(visit.current_key, ctx.settings);
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

void PolicyEngine::Backup(const Visit& visit, const VisitContext& ctx) {
  SearchValue value = visit.result;
  for (size_t i = visit.path.size(); i-- > 0;) {
    const PathStep& step = visit.path[i];
    const bool leaf = i + 1 == visit.path.size();
    // An in-search repetition leaf has no graph node, and a counted leaf took
    // its value at install; the parent receives the plain conversion.
    if (step.generation == 0 || (leaf && visit.leaf_counted)) {
      value = value.Parent();
      continue;
    }
    // The leaf commits only its node value; ancestors complete their edges.
    const auto move =
        i + 1 < visit.path.size() ? step.selected_move : std::nullopt;
    const BackupResult result =
        graph_->BackupNode(step.key, step.generation, value, move);
    ctx.metrics.backup_node_steps.fetch_add(1);
    if (result.node != UpdateResult::kApplied) {
      ctx.metrics.stale_generation_node_updates.fetch_add(1);
    }
    if (result.edge == UpdateResult::kUnderflow) {
      ctx.metrics.invariant_underflow_prevented.fetch_add(1);
    } else if (result.edge != UpdateResult::kApplied) {
      ctx.metrics.stale_generation_edge_updates.fetch_add(1);
    }
    // A stale or missing node falls back to the plain conversion.
    value = result.to_parent ? *result.to_parent : value.Parent();
  }
}

void PolicyEngine::Cancel(const Visit& visit, const VisitContext& ctx) {
  for (const PathStep& step : visit.path) {
    if (step.selected_move &&
        graph_->CancelEdge(step.key, step.generation, *step.selected_move) ==
            UpdateResult::kUnderflow) {
      ctx.metrics.invariant_underflow_prevented.fetch_add(1);
    }
  }
}

VisitEngine::ResumeAction PolicyEngine::OnResume(const Visit&,
                                                 const VisitContext& ctx) {
  // Restarting is like a collision: select again from the root with the new
  // value known, rather than committing this visit to the subtree.
  return ctx.settings.restart_waiters ? ResumeAction::kRestart
                                      : ResumeAction::kSelect;
}

void PolicyEngine::Report(const VisitOrigin& root, const VisitContext& ctx,
                          ThinkingInfo& info) const {
  auto node = graph_->SnapshotNode(root.key);
  if (node && node->value.visits) {
    info.score = CentipawnsFromQ(node->value.Q());
    info.wdl = WdlFromValue(node->value.Q(), node->value.D());
  }
  std::vector<Move>& pv = info.pv;
  pv.clear();
  PositionHistory history = root.history;
  std::unordered_set<uint64_t> seen{root.key.hash};
  while (node && pv.size() < kMaxPvLength &&
         node->lifecycle == NodeLifecycle::kExpanded &&
         node->terminal == TerminalKind::kNonTerminal) {
    const EdgeState* edge = PvEdge(*node);
    if (!edge) break;
    pv.push_back(ToUci(edge->move, history.Last()));
    history.Append(edge->move);
    const NodeKey key = MakeNodeKey(history, ctx.settings.history_key_length);
    if (!seen.insert(key.hash).second) break;
    node = graph_->SnapshotNode(key);
  }
}

RootStats PolicyEngine::Root(NodeKey root) const {
  RootStats stats;
  const auto node = graph_->SnapshotNode(root);
  if (!node) return stats;
  stats.expanded = node->lifecycle == NodeLifecycle::kExpanded;
  stats.visits = node->value.visits;
  stats.edges = node->edges.size();
  for (const auto& edge : node->edges) {
    if (edge.value.visits > stats.best_edge_visits) {
      stats.second_edge_visits = stats.best_edge_visits;
      stats.best_edge_visits = edge.value.visits;
    } else {
      stats.second_edge_visits =
          std::max(stats.second_edge_visits, uint64_t{edge.value.visits});
    }
  }
  return stats;
}

FindOrCreateResult PolicyEngine::FindOrCreateMaterializing(
    NodeKey key, MaterializationTicketId ticket) {
  return graph_->FindOrCreateMaterializing(key, ticket);
}

InstallResult PolicyEngine::Install(NodeKey key, MaterializationTicketId ticket,
                                    const ExpansionPayload& payload) {
  return graph_->InstallPayload(key, ticket, payload);
}

void PolicyEngine::CancelMaterialization(NodeKey key,
                                         MaterializationTicketId ticket) {
  graph_->CancelMaterialization(key, ticket);
}

size_t PolicyEngine::GraphSize() const { return graph_->Size(); }

}  // namespace lczero::lc5
