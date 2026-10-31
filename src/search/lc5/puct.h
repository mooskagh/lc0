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
#include <optional>
#include <string_view>

#include "chess/callbacks.h"
#include "search/lc5/graph.h"
#include "search/lc5/value.h"
#include "search/lc5/visit_engine.h"
#include "utils/optionsdict.h"
#include "utils/optionsparser.h"

namespace lczero::lc5 {

enum class FpuStrategy { kAbsolute, kReduction };
// Which child visits count in the parent's totals: sqrt(N) and the FPU
// visited policy.
enum class PuctTotals { kStarted, kCompleted };
// How far an in-search two-fold draw is backed up: to the search root, or to
// the repeated position's first occurrence only.
enum class TwoFoldDraws { kRoot, kContained };

// PUCT selection over running averages, as in classic. Nodes and edges keep
// value means and bounds, and the message is a value and bounds from the side
// to move at the node.
struct PuctAveraging {
  static constexpr std::string_view kName = "puct";

  struct Params {
    float cpuct = 1.745f;
    float cpuct_base = 38739.0f;
    float cpuct_factor = 3.894f;
    FpuStrategy fpu_strategy = FpuStrategy::kReduction;
    float fpu_value = 0.33f;
    PuctTotals totals = PuctTotals::kStarted;
    WaiterResume waiter_resume = WaiterResume::kSelect;
    // Moves-left utility: in clearly decided nodes, prefer edges that shorten
    // a win or lengthen a loss. Needs a network with a moves-left head.
    struct MovesLeft {
      bool enabled = false;
      float max_effect = 0.0345f;
      float threshold = 0.8f;
      float slope = 0.0027f;
      float constant_factor = 0.0f;
      float scaled_factor = 1.6521f;
      float quadratic_factor = -0.6521f;
    };
    MovesLeft moves_left = {};
    // Use proven bounds in selection, the move choice, early stopping and
    // mate scores. Bounds are tracked either way.
    bool sticky_endgames = false;
    TwoFoldDraws two_fold_draws = TwoFoldDraws::kRoot;

    static void Populate(OptionsParser* options);
    // Moves-left stays off if the backend has no moves-left head.
    static Params FromOptions(const OptionsDict& options,
                              const BackendAttributes& backend);
  };

  // Node stats are from the node's side to move, edge stats from the edge's
  // parent's.
  using NodeStats = ValueStats;
  using EdgeStats = ValueStats;
  struct Message {
    SearchValue value;
    Bounds bounds = {};
  };

  // The message a leaf evaluation (or terminal value) starts the backup with.
  // An exact value (a terminal other than a repetition, or a proven node) is
  // proven.
  static Message Leaf(const SearchValue& value, bool exact) {
    return {value, exact ? Bounds::Exact(value.q, value.m) : Bounds{}};
  }
  // The message the parent receives when this node's update is skipped: an
  // in-search repetition leaf or a counted leaf.
  static Message Skip(const Message& message) {
    return {message.value.Parent(), message.bounds.Parent()};
  }
  // The message the parent receives past a stale or missing node: the value
  // without bounds, which belong to a node that is not the parent's child.
  static Message Stale(const Message& message) {
    return {message.value.Parent(), Bounds{}};
  }
  // Folds `message` into the node and, if given, the edge the visit took, and
  // returns what the parent receives: the value and the node's own bounds.
  static Message Update(NodeState<PuctAveraging>& node,
                        TriedEdge<PuctAveraging>* edge,
                        const Message& message) {
    node.stats.Add(message.value);
    if (edge) edge->stats.Add(message.value);
    if (node.edges.empty()) {
      // A terminal's bounds are its own leaf's.
      node.stats.bounds = message.bounds;
    } else if (edge && edge->stats.bounds != message.bounds) {
      // Bounds rarely change, so the node is recomputed only then. A
      // node-only update (the leaf's own value) leaves them alone.
      edge->stats.bounds = message.bounds;
      node.stats.bounds = BoundsFromEdges(node);
    }
    return {message.value.Parent(), node.stats.bounds.Parent()};
  }

  // What the parent receives from the node's current statistics instead of
  // the backed-up message: the node's mean and bounds.
  static Message Current(const NodeState<PuctAveraging>& node) {
    const ValueStats& stats = node.stats;
    return {SearchValue{stats.Q(), stats.D(), stats.M()}.Parent(),
            stats.bounds.Parent()};
  }
  // Whether an in-search two-fold draw stops at the repeated position's first
  // occurrence, which then sends its Current value further up.
  static bool ContainTwoFold(const Params& params) {
    return params.two_fold_draws == TwoFoldDraws::kContained;
  }

  static uint32_t Visits(const ValueStats& stats) { return stats.visits; }

  // The edge to visit next; `node` is expanded, non-terminal, with edges.
  static size_t Select(const NodeState<PuctAveraging>& node,
                       const Params& params);
  // The edge a waiter visits next after its visit through edge `index` found
  // the child's value `from_child`: selection is rerun as if that visit had
  // completed. The graph has already released its reservation; other visits
  // in flight on the edge still count as in flight.
  static size_t Reselect(const NodeState<PuctAveraging>& node,
                         const Params& params, size_t index,
                         const Message& from_child);
  // The exact value of a proven node, which a visit backs up instead of
  // descending; nullopt if the node is not proven or StickyEndgames is off.
  static std::optional<SearchValue> ProvenValue(
      const NodeState<PuctAveraging>& node, const Params& params);
  // Whether the node's result is proven and StickyEndgames is on.
  static bool Decided(const NodeState<PuctAveraging>& node,
                      const Params& params) {
    return params.sticky_endgames && node.stats.bounds.Proven();
  }
  // The index of the PV and bestmove edge: most visits, then best Q, then
  // PreferOnTie; nullopt if no edge has a visit. With StickyEndgames, a
  // proven win comes first (shortest first) and a proven loss last (longest
  // first).
  static std::optional<size_t> ChooseEdge(const NodeState<PuctAveraging>& node,
                                          const Params& params);
  // The index of the bestmove edge: ChooseEdge, or before any edge has a visit
  // the highest-prior one (edges are sorted by PreferOnTie); nullopt if no
  // edges.
  static std::optional<size_t> ChooseMove(const NodeState<PuctAveraging>& node,
                                          const Params& params) {
    const auto index = ChooseEdge(node, params);
    if (index || node.edges.empty()) return index;
    return 0;
  }
  // Fills the score and WDL of `info` if the node has visits; with
  // StickyEndgames, a proven win or loss as a mate score instead.
  static void Describe(const NodeState<PuctAveraging>& node,
                       const Params& params, ThinkingInfo& info);

  // The node's bounds from its edges' bounds, which are in the node's frame.
  // An untried edge is unknown.
  static Bounds BoundsFromEdges(const NodeState<PuctAveraging>& node);

  // Reselecting or restarting is like a collision: select again with the new
  // value known, rather than committing this visit to the subtree.
  static WaiterResume OnResume(const Params& params) {
    return params.waiter_resume;
  }
};
// The bounds grew the edge from 20 to 24 bytes.
static_assert(sizeof(TriedEdge<PuctAveraging>) <= 24);

}  // namespace lczero::lc5
