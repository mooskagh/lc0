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
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <tuple>

#include "chess/callbacks.h"
#include "search/lc5/graph/graph.h"
#include "search/lc5/graph/value.h"
#include "search/lc5/policies/policy_engine.h"
#include "utils/optionsdict.h"
#include "utils/optionsparser.h"

namespace lczero::lc5 {

enum class FpuStrategy { kAbsolute, kReduction };

// The parameters of PUCT selection, shared by all PUCT policies (which
// therefore share their options).
struct PuctParams {
  float cpuct = 1.745f;
  float cpuct_base = 38739.0f;
  float cpuct_factor = 3.894f;
  FpuStrategy fpu_strategy = FpuStrategy::kReduction;
  float fpu_value = 0.33f;
  // Moves-left utility: in clearly decided nodes, prefer edges that shorten
  // a win or lengthen a loss. Needs a network with a moves-left head.
  struct MovesLeft {
    bool enabled = true;
    float max_effect = 0.0345f;
    float threshold = 0.8f;
    float slope = 0.0027f;
    float constant_factor = 0.0f;
    float scaled_factor = 1.6521f;
    float quadratic_factor = -0.6521f;

    bool ActiveAt(float node_q) const {
      return enabled && std::abs(node_q) > threshold;
    }
    // A bonus for a winning edge whose child has fewer moves left than the
    // node, or a losing one whose child has more; scaled by the edge's |Q|
    // above `threshold`.
    float Utility(float child_m, float node_m, float q) const;
  };
  MovesLeft moves_left = {};
  // Use proven bounds in selection, the move choice, early stopping and
  // mate scores. Bounds are tracked either way.
  bool sticky_endgames = false;

  static void Populate(OptionsParser* options);
  // Moves-left stays off if the backend has no moves-left head.
  static PuctParams FromOptions(const OptionsDict& options,
                                const BackendAttributes& backend);
};

// The WDL and centipawn score that FillThinkingInfo reports for a value.
ThinkingInfo::WDL WdlFromValue(float q, float d);
int CentipawnsFromQ(float q);

// PUCT selection, move choice, bounds and reporting over ValueStats nodes and
// edges. `Self` supplies kName and ApplyMessage: how a message changes the node
// and the edge it came through, and what the parent receives. Nodes and edges
// keep value means and bounds, and the message is a value and bounds from the
// side to move at the node.
template <typename Self>
struct PuctPolicy {
  using Params = PuctParams;

  // Node stats are from the node's side to move, edge stats from the edge's
  // parent's.
  using NodeStats = ValueStats;
  using EdgeStats = ValueStats;
  struct Message {
    VisitValue value;
    GameResultBounds bounds = {};
  };

  // Sent to seed a newly installed node and to start backprop from a leaf;
  // `exact` (non-repetition terminals, proven nodes) sets proven bounds.
  static Message MakeLeafMessage(const VisitValue& value, bool exact) {
    return {value, exact ? GameResultBounds::Exact(value.q, value.m)
                         : GameResultBounds{}};
  }
  // Sent to the parent during backprop when skipping a node update (an
  // in-search repetition leaf, or a leaf already counted at install).
  static Message MakeSkipMessage(const Message& message) {
    return {message.value.Parent(), message.bounds.Parent()};
  }
  // Sent to the parent during backprop when a node is missing from the graph
  // (evicted before backup): forwards the flipped value without bounds.
  static Message MakeStaleMessage(const Message& message) {
    return {message.value.Parent(), GameResultBounds{}};
  }

  // The edge index to visit next; `node` is materialized, non-terminal, with
  // edges.
  static size_t PickEdgeToVisit(const Node<Self>& node, const Params& params);
  // The exact value of a proven node, which a visit backs up instead of
  // descending; nullopt if the node is not proven or StickyEndgames is off.
  static std::optional<VisitValue> GetProvenNodeValue(const Node<Self>& node,
                                                      const Params& params);
  // Whether the node's result is proven and StickyEndgames is on.
  static bool IsNodeDecided(const Node<Self>& node, const Params& params) {
    return params.sticky_endgames && node.stats.bounds.Proven();
  }
  // The PV and bestmove move: most visits, then best Q, then PreferOnTie;
  // if no edge has a visit, the highest-prior edge when `allow_unvisited` is
  // true, or nullopt. With StickyEndgames, a proven win comes first (shortest
  // first) and a proven loss last (longest first).
  static std::optional<Move> GetBestMove(const Node<Self>& node,
                                         const Params& params,
                                         bool allow_unvisited = false);
  // Fills the score and WDL of `info` if the node has visits; with
  // StickyEndgames, a proven win or loss as a mate score instead.
  static void FillThinkingInfo(const Node<Self>& node, const Params& params,
                               ThinkingInfo& info);

  // The node's bounds from its edges' bounds, which are in the node's frame.
  // An untried edge is unknown.
  static GameResultBounds ComputeBoundsFromEdges(const Node<Self>& node);
  // Sets the bounds after an update with `bounds` from the edge (or, without
  // an edge, the node's own leaf): a terminal's bounds are its leaf's, and a
  // non-terminal node's follow from its edges'.
  static void UpdateBounds(Node<Self>& node, TriedEdge<Self>* edge,
                           const GameResultBounds& bounds) {
    if (node.edges.empty()) {
      // A terminal's bounds are its own leaf's.
      node.stats.bounds = bounds;
    } else if (edge && edge->stats.bounds != bounds) {
      // Bounds rarely change, so the node is recomputed only then. A
      // node-only update (the leaf's own value) leaves them alone.
      edge->stats.bounds = bounds;
      node.stats.bounds = ComputeBoundsFromEdges(node);
    }
  }
};

// PUCT over running averages: an edge averages the values of the visits that
// went through it, and a node those of all visits to it.
struct PuctAveraging : PuctPolicy<PuctAveraging> {
  static constexpr std::string_view kName = "puct";

  // Folds `message` into the node and, if given, the edge the visit took, and
  // returns what the parent receives: the value and the node's own bounds.
  static Message ApplyMessage(Node<PuctAveraging>& node,
                              TriedEdge<PuctAveraging>* edge,
                              const Message& message) {
    node.stats.Add(message.value);
    if (edge) edge->stats.Add(message.value);
    UpdateBounds(node, edge, message.bounds);
    return {message.value.Parent(), node.stats.bounds.Parent()};
  }
};
static_assert(sizeof(TriedEdge<PuctAveraging>) <= 24);

template <typename Self>
size_t PuctPolicy<Self>::PickEdgeToVisit(const Node<Self>& node,
                                         const Params& params) {
  assert(!node.edges.empty());
  const NodeStats& stats = node.stats;

  // Started visits (completed plus in flight) count in the parent's totals,
  // sqrt(N) and the FPU visited policy, as well as in the edge's own U.
  // Only tried edges have been started, so the sums only need that prefix.
  uint64_t children_visits_started = 0;
  float visited_policy = 0.0f;
  for (size_t i = 0; i < node.tried.size(); ++i) {
    const uint32_t n = node.tried[i].VisitsStarted();
    children_visits_started += n;
    if (n > 0) visited_policy += node.edges[i].prior;
  }

  const float cpuct =
      params.cpuct +
      params.cpuct_factor *
          std::log((stats.visits + params.cpuct_base) / params.cpuct_base);
  const float puct_mult =
      cpuct * std::sqrt(static_cast<float>(
                  std::max<uint64_t>(1, children_visits_started)));
  const float fpu =
      params.fpu_strategy == FpuStrategy::kAbsolute
          ? params.fpu_value
          : stats.Q() - params.fpu_value * std::sqrt(visited_policy);
  const bool use_m = params.moves_left.ActiveAt(stats.Q());

  // With StickyEndgames, a proven edge scores its exact value without a
  // moves-left term, and an unproven one stays within its bounds.
  const auto utility = [&](const TriedEdge<Self>& tried) -> float {
    if (tried.stats.visits == 0) return fpu;
    const GameResultBounds& bounds = tried.stats.bounds;
    if (params.sticky_endgames && bounds.Proven()) return bounds.lower;
    const float q =
        params.sticky_endgames
            ? std::clamp<float>(tried.stats.Q(), bounds.lower, bounds.upper)
            : tried.stats.Q();
    // The edge's M is one ply longer than the child's.
    return use_m ? q + params.moves_left.Utility(tried.stats.M() - 1.0f,
                                                 stats.M(), q)
                 : q;
  };

  // Untried edges get neither bounds nor a moves-left term, so they share one
  // score apart from the prior.
  const auto score = [&](size_t i) {
    const float prior = node.edges[i].prior;
    if (i >= node.tried.size()) return fpu + puct_mult * prior;
    const auto& tried = node.tried[i];
    return utility(tried) + puct_mult * prior / (1.0f + tried.VisitsStarted());
  };

  const auto pick_best = [&](size_t count) {
    size_t best = 0;
    float best_score = score(0);
    for (size_t i = 1; i < count; ++i) {
      // Ties keep the lower index, which PreferOnTie prefers.
      if (const float s = score(i); s > best_score) {
        best = i;
        best_score = s;
      }
    }
    return best;
  };

  // Untried edges share the FPU and are sorted by PreferOnTie, so the first
  // one beats or ties all later ones and wins the tie.
  const size_t best =
      pick_best(std::min(node.tried.size() + 1, node.edges.size()));
  assert(pick_best(node.edges.size()) == best);
  return best;
}

template <typename Self>
std::optional<VisitValue> PuctPolicy<Self>::GetProvenNodeValue(
    const Node<Self>& node, const Params& params) {
  const GameResultBounds& bounds = node.stats.bounds;
  if (!params.sticky_endgames || !bounds.Proven()) return std::nullopt;
  return VisitValue{.q = static_cast<float>(bounds.lower),
                    .d = bounds.lower == 0 ? 1.0f : 0.0f,
                    .m = static_cast<float>(bounds.plies)};
}

template <typename Self>
std::optional<Move> PuctPolicy<Self>::GetBestMove(const Node<Self>& node,
                                                  const Params& params,
                                                  bool allow_unvisited) {
  // Untried edges have no visits.
  const auto rank = [&](size_t i) {
    const auto& stats = node.tried[i].stats;
    // With StickyEndgames: proven wins, then the rest, then proven losses.
    int result = 0;
    float plies = 0.0f;
    if (params.sticky_endgames && stats.bounds.lower == 1) {
      result = 1;
      plies = -stats.bounds.plies;
    } else if (params.sticky_endgames && stats.bounds.upper == -1) {
      result = -1;
      plies = stats.bounds.plies;
    }
    return std::tuple(result, plies, stats.visits, stats.Q());
  };
  std::optional<size_t> best;
  for (size_t i = 0; i < node.tried.size(); ++i) {
    if (node.tried[i].stats.visits == 0) continue;
    // Ties keep the lower index, which PreferOnTie prefers.
    if (!best || rank(i) > rank(*best)) best = i;
  }
  if (best) return node.edges[*best].move;
  if (allow_unvisited && !node.edges.empty()) return node.edges[0].move;
  return std::nullopt;
}

template <typename Self>
void PuctPolicy<Self>::FillThinkingInfo(const Node<Self>& node,
                                        const Params& params,
                                        ThinkingInfo& info) {
  const NodeStats& stats = node.stats;
  if (!stats.visits) return;
  info.score = CentipawnsFromQ(stats.Q());
  info.wdl = WdlFromValue(stats.Q(), stats.D());
  const GameResultBounds& bounds = stats.bounds;
  // A terminal root (0 plies) has no move to show a mate for.
  if (params.sticky_endgames && bounds.Proven() && bounds.lower != 0 &&
      bounds.plies > 0) {
    // UCI counts mates in moves, negative when the side to move is mated.
    info.mate =
        bounds.lower == 1 ? (bounds.plies + 1) / 2 : -(bounds.plies / 2);
    info.score.reset();
  }
}

template <typename Self>
GameResultBounds PuctPolicy<Self>::ComputeBoundsFromEdges(
    const Node<Self>& node) {
  GameResultBounds bounds{.lower = -1,
                          .upper = static_cast<int8_t>(
                              node.tried.size() < node.edges.size() ? 1 : -1)};
  for (const auto& tried : node.tried) {
    bounds.lower = std::max(bounds.lower, tried.stats.bounds.lower);
    bounds.upper = std::max(bounds.upper, tried.stats.bounds.upper);
  }
  if (bounds.lower == 1) {
    // A win takes the shortest winning edge.
    bounds.plies = 255;
    for (const auto& tried : node.tried) {
      if (tried.stats.bounds.lower == 1) {
        bounds.plies = std::min(bounds.plies, tried.stats.bounds.plies);
      }
    }
  } else if (bounds.upper == -1) {
    // A loss takes the longest one.
    for (const auto& tried : node.tried) {
      bounds.plies = std::max(bounds.plies, tried.stats.bounds.plies);
    }
  }
  return bounds;
}

}  // namespace lczero::lc5
