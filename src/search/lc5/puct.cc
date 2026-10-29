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
#include <string>
#include <utility>
#include <vector>

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
const OptionId kRestartWaiters{
    "restart-waiters", "RestartWaiters",
    "Visits that waited for a node's evaluation select again from the root "
    "instead of continuing below that node."};

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
  options->Add<BoolOption>(kRestartWaiters) = defaults.restart_waiters;
}

PuctAveraging::Params PuctAveraging::Params::FromOptions(
    const OptionsDict& options) {
  return {.cpuct = options.Get<float>(kCpuct),
          .cpuct_base = options.Get<float>(kCpuctBase),
          .cpuct_factor = options.Get<float>(kCpuctFactor),
          .fpu_strategy = options.Get<std::string>(kFpuStrategy) == "absolute"
                              ? FpuStrategy::kAbsolute
                              : FpuStrategy::kReduction,
          .fpu_value = options.Get<float>(kFpuValue),
          .restart_waiters = options.Get<bool>(kRestartWaiters)};
}

size_t PuctAveraging::Select(const NodeState<PuctAveraging>& node,
                             const Params& params) {
  using Edge = EdgeState<PuctAveraging>;
  const auto started = [](const Edge& edge) {
    return edge.stats.visits + edge.in_flight;
  };
  // Started edges all lie below started_end, so the sums only need that prefix.
  uint64_t children_started = 0;
  float visited_policy = 0.0f;
  for (uint32_t i = 0; i < node.started_end; ++i) {
    const auto& edge = node.edges[i];
    children_started += started(edge);
    if (started(edge) > 0) visited_policy += edge.prior;
  }
  const float cpuct =
      params.cpuct +
      params.cpuct_factor *
          std::log((node.stats.visits + params.cpuct_base) / params.cpuct_base);
  const float scale =
      std::sqrt(static_cast<float>(std::max<uint64_t>(1, children_started)));
  const float fpu =
      params.fpu_strategy == FpuStrategy::kAbsolute
          ? params.fpu_value
          : node.stats.Q() - params.fpu_value * std::sqrt(visited_policy);
  const auto score = [&](const Edge& edge) {
    const float q = edge.stats.visits ? edge.stats.Q() : fpu;
    return q + cpuct * edge.prior * scale / (1.0f + started(edge));
  };
  const auto pick_best = [&](size_t end) {
    size_t best = 0;
    float best_score = score(node.edges[0]);
    for (size_t i = 1; i < end; ++i) {
      const float s = score(node.edges[i]);
      if (s > best_score ||
          (s == best_score && PreferOnTie(node.edges[i], node.edges[best]))) {
        best = i;
        best_score = s;
      }
    }
    return best;
  };

  // Unstarted edges share the FPU and are sorted by PreferOnTie, so the first
  // one beats or ties all later ones and wins the tie.
  const size_t best_index =
      pick_best(std::min<size_t>(node.started_end + 1, node.edges.size()));
#ifndef NDEBUG
  for (size_t i = node.started_end; i < node.edges.size(); ++i) {
    assert(started(node.edges[i]) == 0);
  }
  assert(pick_best(node.edges.size()) == best_index);
#endif
  return best_index;
}

const EdgeState<PuctAveraging>* PuctAveraging::ChooseEdge(
    const NodeState<PuctAveraging>& node) {
  using Edge = EdgeState<PuctAveraging>;
  const Edge* best = nullptr;
  for (const auto& edge : node.edges) {
    if (edge.stats.visits == 0) continue;
    const auto rank = [](const Edge& e) {
      return std::pair(e.stats.visits, e.stats.Q());
    };
    if (!best || rank(edge) > rank(*best) ||
        (rank(edge) == rank(*best) && PreferOnTie(edge, *best))) {
      best = &edge;
    }
  }
  return best;
}

void PuctAveraging::Describe(const NodeStats& node, ThinkingInfo& info) {
  if (!node.visits) return;
  info.score = CentipawnsFromQ(node.Q());
  info.wdl = WdlFromValue(node.Q(), node.D());
}

}  // namespace lczero::lc5
