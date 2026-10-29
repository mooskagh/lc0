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
#include <string_view>

#include "chess/callbacks.h"
#include "search/lc5/graph.h"
#include "search/lc5/value.h"
#include "search/lc5/visit_engine.h"
#include "utils/optionsdict.h"
#include "utils/optionsparser.h"

namespace lczero::lc5 {

enum class FpuStrategy { kAbsolute, kReduction };

// PUCT selection over running averages, as in classic. Nodes and edges keep
// value means, and the message is a value from the side to move at the node.
struct PuctAveraging {
  static constexpr std::string_view kName = "puct";

  struct Params {
    float cpuct = 1.745f;
    float cpuct_base = 38739.0f;
    float cpuct_factor = 3.894f;
    FpuStrategy fpu_strategy = FpuStrategy::kReduction;
    float fpu_value = 0.33f;
    bool restart_waiters = false;

    static void Populate(OptionsParser* options);
    static Params FromOptions(const OptionsDict& options);
  };

  using NodeStats = ValueStats;
  using EdgeStats = ValueStats;
  using Message = SearchValue;

  // The message a leaf evaluation (or terminal value) starts the backup with.
  static Message Leaf(const SearchValue& value) { return value; }
  // The message the parent receives when this node's update is skipped: an
  // in-search repetition leaf, a counted leaf, or a stale or missing node.
  static Message Skip(const Message& message) { return message.Parent(); }
  // Folds `message` into the node and, if given, the edge the visit took, and
  // returns what the parent receives.
  static Message Update(NodeStats& node, EdgeStats* edge,
                        const Message& message) {
    node.Add(message);
    if (edge) edge->Add(message);
    return message.Parent();
  }

  static uint32_t Visits(const ValueStats& stats) { return stats.visits; }

  // The edge to visit next; `node` is expanded, non-terminal, with edges.
  static size_t Select(const NodeState<PuctAveraging>& node,
                       const Params& params);
  // The PV and bestmove edge: most visits, then best Q, then PreferOnTie;
  // nullptr if no edge has a visit.
  static const EdgeState<PuctAveraging>* ChooseEdge(
      const NodeState<PuctAveraging>& node);
  // The bestmove edge: ChooseEdge, or before any edge has a visit the
  // highest-prior one (edges are sorted by PreferOnTie); nullptr if no edges.
  static const EdgeState<PuctAveraging>* ChooseMove(
      const NodeState<PuctAveraging>& node) {
    const auto* edge = ChooseEdge(node);
    return edge || node.edges.empty() ? edge : &node.edges.front();
  }
  // Fills the score and WDL of `info` if the node has visits.
  static void Describe(const NodeStats& node, ThinkingInfo& info);

  static VisitEngine::ResumeAction OnResume(const Params& params) {
    // Restarting is like a collision: select again from the root with the new
    // value known, rather than committing this visit to the subtree.
    return params.restart_waiters ? VisitEngine::ResumeAction::kRestart
                                  : VisitEngine::ResumeAction::kSelect;
  }
};
static_assert(sizeof(EdgeState<PuctAveraging>) <= 24);

}  // namespace lczero::lc5
