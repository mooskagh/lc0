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

#include "search/lc5/policies/puct.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "search/lc5/policies/policy_engine.h"
#include "search/lc5/policies/registry.h"
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

const OptionId kStickyEndgames{
    "sticky-endgames", "StickyEndgames",
    "Prove wins, draws and losses from terminal positions up the tree and use "
    "them in selection, the move choice, early stopping and mate scores. "
    "Repetition draws are not proven."};

}  // namespace

void PuctParams::Populate(OptionsParser* options) {
  const PuctParams defaults;
  options->Add<FloatOption>(kCpuct, 0.0f, 100.0f) = defaults.cpuct;
  options->Add<FloatOption>(kCpuctBase, 1.0f, 1.0e9f) = defaults.cpuct_base;
  options->Add<FloatOption>(kCpuctFactor, 0.0f, 1000.0f) =
      defaults.cpuct_factor;
  options->Add<ChoiceOption>(
      kFpuStrategy, std::vector<std::string>{"absolute", "reduction"}) =
      defaults.fpu_strategy == FpuStrategy::kAbsolute ? "absolute"
                                                      : "reduction";
  options->Add<FloatOption>(kFpuValue, -100.0f, 100.0f) = defaults.fpu_value;
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
  options->Add<BoolOption>(kStickyEndgames) = defaults.sticky_endgames;
}

PuctParams PuctParams::FromOptions(const OptionsDict& options,
                                   const BackendAttributes& backend) {
  return {
      .cpuct = options.Get<float>(kCpuct),
      .cpuct_base = options.Get<float>(kCpuctBase),
      .cpuct_factor = options.Get<float>(kCpuctFactor),
      .fpu_strategy = options.Get<std::string>(kFpuStrategy) == "absolute"
                          ? FpuStrategy::kAbsolute
                          : FpuStrategy::kReduction,
      .fpu_value = options.Get<float>(kFpuValue),
      // Without a moves-left head, M is only the mean depth of the leaves
      // below.
      .moves_left =
          {.enabled = options.Get<bool>(kMovesLeft) && backend.has_mlh,
           .max_effect = options.Get<float>(kMovesLeftMaxEffect),
           .threshold = options.Get<float>(kMovesLeftThreshold),
           .slope = options.Get<float>(kMovesLeftSlope),
           .constant_factor = options.Get<float>(kMovesLeftConstantFactor),
           .scaled_factor = options.Get<float>(kMovesLeftScaledFactor),
           .quadratic_factor = options.Get<float>(kMovesLeftQuadraticFactor)},
      .sticky_endgames = options.Get<bool>(kStickyEndgames)};
}

float PuctParams::MovesLeft::Utility(float child_m, float node_m,
                                     float q) const {
  const float m =
      std::clamp(slope * (child_m - node_m), -max_effect, max_effect) *
      FastSign(-q);
  float abs_q = std::abs(q);
  if (threshold > 0.0f && threshold < 1.0f) {
    abs_q = std::max(0.0f, abs_q - threshold) / (1.0f - threshold);
  }
  return m * (constant_factor + scaled_factor * abs_q +
              quadratic_factor * abs_q * abs_q);
}

ThinkingInfo::WDL WdlFromValue(float q, float d) {
  q = std::clamp(q, -1.0f, 1.0f);
  d = std::clamp(d, 0.0f, 1.0f);
  int w = std::max(0, static_cast<int>(std::lround(500.0 * (1.0 + q - d))));
  int l = std::max(0, static_cast<int>(std::lround(500.0 * (1.0 - q - d))));
  int draw = 1000 - w - l;
  if (draw < 0) {
    // Both w and l landed on .5 and rounded up (w + l == 1001); round to even.
    if (w % 2 != 0) {
      --w;
    } else {
      --l;
    }
    draw = 0;
  }
  return {w, draw, l};
}

int CentipawnsFromQ(float q) {
  return static_cast<int>(90 * std::tan(1.5637541897 * q));
}

// The default policy.
REGISTER_LC5_POLICY(PuctAveraging, 100)

}  // namespace lczero::lc5
