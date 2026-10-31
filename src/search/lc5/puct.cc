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
#include <cmath>
#include <string>
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

const OptionId kStickyEndgames{
    "sticky-endgames", "StickyEndgames",
    "Prove wins, draws and losses from terminal positions up the tree and use "
    "them in selection, the move choice, early stopping and mate scores. "
    "Repetition draws are not proven."};
const OptionId kTwoFoldDraws{
    "two-fold-draws", "TwoFoldDraws",
    "How far a two-fold repetition inside the search is backed up as a draw: "
    "to the search root, or only to the repeated position's first occurrence, "
    "whose own value then goes further up."};

WaiterResume WaiterResumeFromString(const std::string& value) {
  if (value == "reselect") return WaiterResume::kReselect;
  if (value == "restart") return WaiterResume::kRestart;
  return WaiterResume::kSelect;
}

}  // namespace

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
  options->Add<BoolOption>(kStickyEndgames) = defaults.sticky_endgames;
  options->Add<ChoiceOption>(kTwoFoldDraws,
                             std::vector<std::string>{"root", "contained"}) =
      defaults.two_fold_draws == TwoFoldDraws::kRoot ? "root" : "contained";
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
      .totals = options.Get<std::string>(kPuctTotals) == "completed"
                    ? PuctTotals::kCompleted
                    : PuctTotals::kStarted,
      .waiter_resume =
          WaiterResumeFromString(options.Get<std::string>(kWaiterResume)),
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
      .sticky_endgames = options.Get<bool>(kStickyEndgames),
      .two_fold_draws = options.Get<std::string>(kTwoFoldDraws) == "contained"
                            ? TwoFoldDraws::kContained
                            : TwoFoldDraws::kRoot};
}

}  // namespace lczero::lc5
