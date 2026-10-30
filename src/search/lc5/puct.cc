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

#include "search/lc5/puct.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "utils/fastmath.h"

namespace lczero::lc5 {
namespace {

const OptionId kCpuct{"cpuct", "CPuct", "Lc5 PUCT initial constant."};
const OptionId kCpuctBase{"cpuct-base", "CPuctBase", "Lc5 PUCT growth base."};
const OptionId kCpuctFactor{"cpuct-factor", "CPuctFactor",
                            "Lc5 PUCT growth multiplier."};
const OptionId kFpuStrategy{"fpu-strategy", "FpuStrategy",
                            "Lc5 first-play urgency strategy."};
const OptionId kFpuValue{"fpu-value", "FpuValue",
                         "Lc5 absolute FPU value or FPU reduction."};
const OptionId kPuctTotals{
    "puct-totals", "PuctTotals",
    "Which child visits count in the parent's sqrt(N) and FPU visited policy: "
    "started (completed plus in flight) or completed only, as in classic. "
    "In-flight visits always count in the edge's own U denominator."};
const OptionId kWaiterResume{
    "waiter-resume", "WaiterResume",
    "What a visit does after waiting for a node's evaluation: select below "
    "that node, reselect at its parent with the new value (and switch edges if "
    "another one is now better), or restart from the root."};
const OptionId kMovesLeft{
    "moves-left", "MovesLeft",
    "Use the network's moves-left estimate to prefer shorter wins and longer "
    "losses in clearly decided positions. Needs a network with a moves-left "
    "head."};
const OptionId kMovesLeftMaxEffect{
    "moves-left-max-effect", "MovesLeftMaxEffect",
    "Largest moves-left bonus or penalty, in Q units, before Q scaling."};
const OptionId kMovesLeftThreshold{
    "moves-left-threshold", "MovesLeftThreshold",
    "Moves-left applies only in nodes whose |Q| is above this."};
const OptionId kMovesLeftSlope{
    "moves-left-slope", "MovesLeftSlope",
    "Moves-left bonus per ply of difference between the child's and the "
    "node's moves left."};
const OptionId kMovesLeftConstantFactor{
    "moves-left-constant-factor", "MovesLeftConstantFactor",
    "Constant term of the moves-left scaling by the edge's Q."};
const OptionId kMovesLeftScaledFactor{
    "moves-left-scaled-factor", "MovesLeftScaledFactor",
    "Linear term of the moves-left scaling by the edge's |Q| above the "
    "threshold."};
const OptionId kMovesLeftQuadraticFactor{
    "moves-left-quadratic-factor", "MovesLeftQuadraticFactor",
    "Quadratic term of the moves-left scaling by the edge's |Q| above the "
    "threshold."};

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

WaiterResume WaiterResumeFromString(const std::string& value) {
  if (value == "reselect") return WaiterResume::kReselect;
  if (value == "restart") return WaiterResume::kRestart;
  return WaiterResume::kSelect;
}

// PUCT selection with `tried_edges` in place of node.tried.
size_t SelectEdge(const NodeState<PuctAveraging>& node,
                  std::span<const TriedEdge<PuctAveraging>> tried_edges,
                  const PuctAveraging::Params& params) {
  const auto started = [](const TriedEdge<PuctAveraging>& tried) {
    return tried.stats.visits + tried.in_flight;
  };
  const bool completed_only = params.totals == PuctTotals::kCompleted;
  // Visits a child contributes to the parent's totals: sqrt(N) and the FPU
  // visited policy. In-flight visits always count in the edge's own U.
  const auto counted = [&](const TriedEdge<PuctAveraging>& tried) {
    return completed_only ? tried.stats.visits : started(tried);
  };
  // Only tried edges have been started, so the sums only need that prefix.
  uint64_t children_counted = 0;
  float visited_policy = 0.0f;
  for (size_t i = 0; i < tried_edges.size(); ++i) {
    const auto& tried = tried_edges[i];
    children_counted += counted(tried);
    if (counted(tried) > 0) visited_policy += node.edges[i].prior;
  }
  const float cpuct =
      params.cpuct +
      params.cpuct_factor *
          std::log((node.stats.visits + params.cpuct_base) / params.cpuct_base);
  const float scale =
      std::sqrt(static_cast<float>(std::max<uint64_t>(1, children_counted)));
  const float fpu =
      params.fpu_strategy == FpuStrategy::kAbsolute
          ? params.fpu_value
          : node.stats.Q() - params.fpu_value * std::sqrt(visited_policy);
  const auto& ml = params.moves_left;
  // The moves-left utility applies only in clearly decided nodes.
  const bool use_m = ml.enabled && std::abs(node.stats.Q()) > ml.threshold;
  // A bonus for a winning edge whose child has fewer moves left than the node,
  // or a losing one whose child has more; scaled by the edge's |Q| above the
  // threshold.
  const auto m_utility = [&](const TriedEdge<PuctAveraging>& tried, float q) {
    // The edge's M is one ply longer than the child's.
    const float m_diff = tried.stats.M() - 1.0f - node.stats.M();
    const float m =
        std::clamp(ml.slope * m_diff, -ml.max_effect, ml.max_effect) *
        FastSign(-q);
    if (ml.threshold > 0.0f && ml.threshold < 1.0f) {
      q = std::max(0.0f, std::abs(q) - ml.threshold) / (1.0f - ml.threshold);
    }
    return m * (ml.constant_factor + ml.scaled_factor * std::abs(q) +
                ml.quadratic_factor * q * q);
  };
  const auto score = [&](size_t i) {
    const bool is_tried = i < tried_edges.size();
    const bool has_visits = is_tried && tried_edges[i].stats.visits;
    float q = has_visits ? tried_edges[i].stats.Q() : fpu;
    // Untried edges get no moves-left term, so they still share one score
    // apart from the prior.
    if (has_visits && use_m) q += m_utility(tried_edges[i], q);
    const uint32_t n = is_tried ? started(tried_edges[i]) : 0u;
    return q + cpuct * node.edges[i].prior * scale / (1.0f + n);
  };
  const auto pick_best = [&](size_t end) {
    size_t best = 0;
    float best_score = score(0);
    for (size_t i = 1; i < end; ++i) {
      const float s = score(i);
      // Ties keep the lower index, which PreferOnTie prefers.
      if (s > best_score) {
        best = i;
        best_score = s;
      }
    }
    return best;
  };

  // Untried edges share the FPU and are sorted by PreferOnTie, so the first
  // one beats or ties all later ones and wins the tie.
  const size_t best_index =
      pick_best(std::min(tried_edges.size() + 1, node.edges.size()));
  assert(pick_best(node.edges.size()) == best_index);
  return best_index;
}

}  // namespace

void PuctAveraging::Params::Populate(OptionsParser* options) {
  const Params defaults;
  options->Add<FloatOption>(kCpuct, 0.0f, 100.0f) = defaults.cpuct;
  options->Add<FloatOption>(kCpuctBase, 1.0f, 1.0e9f) = defaults.cpuct_base;
  options->Add<FloatOption>(kCpuctFactor, 0.0f, 1000.0f) =
      defaults.cpuct_factor;
  options->Add<ChoiceOption>(
      kFpuStrategy, std::vector<std::string>{"absolute", "reduction"}) =
      defaults.fpu_strategy == FpuStrategy::kAbsolute ? "absolute"
                                                      : "reduction";
  options->Add<FloatOption>(kFpuValue, -100.0f, 100.0f) = defaults.fpu_value;
  options->Add<ChoiceOption>(kPuctTotals,
                             std::vector<std::string>{"started", "completed"}) =
      defaults.totals == PuctTotals::kStarted ? "started" : "completed";
  options->Add<ChoiceOption>(
      kWaiterResume,
      std::vector<std::string>{"select", "reselect", "restart"}) =
      defaults.waiter_resume == WaiterResume::kSelect     ? "select"
      : defaults.waiter_resume == WaiterResume::kReselect ? "reselect"
                                                          : "restart";
  const auto& ml = defaults.moves_left;
  options->Add<BoolOption>(kMovesLeft) = ml.enabled;
  options->Add<FloatOption>(kMovesLeftMaxEffect, 0.0f, 1.0f) = ml.max_effect;
  options->Add<FloatOption>(kMovesLeftThreshold, 0.0f, 1.0f) = ml.threshold;
  options->Add<FloatOption>(kMovesLeftSlope, 0.0f, 1.0f) = ml.slope;
  options->Add<FloatOption>(kMovesLeftConstantFactor, -1.0f, 1.0f) =
      ml.constant_factor;
  options->Add<FloatOption>(kMovesLeftScaledFactor, -2.0f, 2.0f) =
      ml.scaled_factor;
  options->Add<FloatOption>(kMovesLeftQuadraticFactor, -1.0f, 1.0f) =
      ml.quadratic_factor;
}

PuctAveraging::Params PuctAveraging::Params::FromOptions(
    const OptionsDict& options, const BackendAttributes& backend) {
  return {
      .cpuct = options.Get<float>(kCpuct),
      .cpuct_base = options.Get<float>(kCpuctBase),
      .cpuct_factor = options.Get<float>(kCpuctFactor),
      .fpu_strategy = options.Get<std::string>(kFpuStrategy) == "absolute"
                          ? FpuStrategy::kAbsolute
                          : FpuStrategy::kReduction,
      .fpu_value = options.Get<float>(kFpuValue),
      .totals = options.Get<std::string>(kPuctTotals) == "completed"
                    ? PuctTotals::kCompleted
                    : PuctTotals::kStarted,
      .waiter_resume =
          WaiterResumeFromString(options.Get<std::string>(kWaiterResume)),
      // Without a moves-left head, M is only the mean depth of the leaves
      // below.
      .moves_left = {
          .enabled = options.Get<bool>(kMovesLeft) && backend.has_mlh,
          .max_effect = options.Get<float>(kMovesLeftMaxEffect),
          .threshold = options.Get<float>(kMovesLeftThreshold),
          .slope = options.Get<float>(kMovesLeftSlope),
          .constant_factor = options.Get<float>(kMovesLeftConstantFactor),
          .scaled_factor = options.Get<float>(kMovesLeftScaledFactor),
          .quadratic_factor = options.Get<float>(kMovesLeftQuadraticFactor)}};
}

size_t PuctAveraging::Select(const NodeState<PuctAveraging>& node,
                             const Params& params) {
  return SelectEdge(node, node.tried, params);
}

size_t PuctAveraging::Reselect(const NodeState<PuctAveraging>& node,
                               const Params& params, size_t index,
                               const Message& from_child) {
  assert(index < node.tried.size());
  // Only runs when a waiter resumes, so copying the tried edges is cheap
  // enough.
  std::vector<TriedEdge<PuctAveraging>> tried(node.tried.begin(),
                                              node.tried.end());
  tried[index].stats.Add(from_child);
  return SelectEdge(node, tried, params);
}

std::optional<size_t> PuctAveraging::ChooseEdge(
    const NodeState<PuctAveraging>& node) {
  // Untried edges have no visits.
  const auto rank = [&](size_t i) {
    return std::pair(node.tried[i].stats.visits, node.tried[i].stats.Q());
  };
  std::optional<size_t> best;
  for (size_t i = 0; i < node.tried.size(); ++i) {
    if (node.tried[i].stats.visits == 0) continue;
    // Ties keep the lower index, which PreferOnTie prefers.
    if (!best || rank(i) > rank(*best)) best = i;
  }
  return best;
}

void PuctAveraging::Describe(const NodeStats& node, ThinkingInfo& info) {
  if (!node.visits) return;
  info.score = CentipawnsFromQ(node.Q());
  info.wdl = WdlFromValue(node.Q(), node.D());
}

}  // namespace lczero::lc5
