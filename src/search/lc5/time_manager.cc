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

#include "search/lc5/time_manager.h"

#include <algorithm>
#include <cmath>

#include "chess/uciloop.h"

namespace lczero::lc5 {
namespace {

// Log-logistic horizon parameters with sudden-death endgame protection.
constexpr float kDefaultAlphaZeroTimePct = 3.0f;
constexpr float kTimeCurveMidpoint = 51.5f;
constexpr float kTimeCurveSteepness = 7.0f;
constexpr float kFirstMoveBonus = 1.0f;
constexpr long double kMaxSingleMoveFraction = 0.08L;
constexpr long double kMinEstimatedMovesToGo = 1.0L / kMaxSingleMoveFraction;
constexpr long double kLowClockSavedThresholdMs = 1500.0L;

// Warmup window and node cushion before smart pruning trusts the visit rate.
constexpr auto kSmartPruningTolerance = std::chrono::milliseconds(80);
constexpr uint64_t kSmartPruningToleranceNodes = 200;

float ComputeEstimatedMovesToGo(int ply, float midpoint, float steepness) {
  const float move = std::max(0, ply) / 2.0f;
  return midpoint *
             std::pow(1.0f + 2.0f * std::pow(move / midpoint, steepness),
                      1.0f / steepness) -
         move;
}

}  // namespace

TimeManager::TimeManager(Config config, const GoParams& params,
                         bool black_to_move, Clock::time_point start_time,
                         int64_t saved_ms)
    : start_time_(start_time), saved_ms_(std::max<int64_t>(0, saved_ms)) {
  // Compute the budget inline rather than passing `saved_ms_` to a helper:
  // GCC 16 `-O3 -flto` IPA-CP folds `MAX_EXPR(saved_ms, 0)` pass-throughs to 0.
  if (params.ponder) return;
  if (params.movetime) {
    const int64_t budget =
        std::max<int64_t>(0, *params.movetime - config.move_overhead_ms);
    deadline_ = start_time + std::chrono::milliseconds(budget);
    return;
  }
  if (params.infinite) return;
  const auto& remaining = black_to_move ? params.btime : params.wtime;
  if (!remaining) return;

  clock_budget_ = true;
  smart_pruning_factor_ = config.smart_pruning_factor;
  if (*remaining <= config.move_overhead_ms) {
    deadline_ = start_time;
    return;
  }

  const int64_t usable = *remaining - config.move_overhead_ms;
  const int64_t increment = std::max<int64_t>(
      0, (black_to_move ? params.binc : params.winc).value_or(0));

  long double budget = 0.0L;
  if (config.alphazero_time_pct != kDefaultAlphaZeroTimePct) {
    // Preserve explicit AlphaZeroTimePct customization as a flat percentage.
    long double fraction = config.alphazero_time_pct / 100.0L;
    if (params.movestogo && *params.movestogo > 0) {
      fraction = std::max(fraction, 1.0L / *params.movestogo);
    }
    // Divide the clock and future increments over the implied move horizon.
    // The current move's increment is only received after we finish searching.
    budget = usable * fraction + increment * (1.0L - fraction) + saved_ms_;
  } else {
    long double movestogo = ComputeEstimatedMovesToGo(
        config.game_ply, kTimeCurveMidpoint, kTimeCurveSteepness);
    movestogo = std::max(movestogo, kMinEstimatedMovesToGo);
    const bool has_explicit_movestogo =
        params.movestogo && *params.movestogo > 0;
    if (has_explicit_movestogo && *params.movestogo < movestogo) {
      movestogo = *params.movestogo;
    }
    const long double fraction = 1.0L / movestogo;
    // Taper saved-time immediate spending below 1.5s usable so late sudden-
    // death endgames do not dump banked time onto a single move.
    const long double saved_scale =
        usable < kLowClockSavedThresholdMs
            ? static_cast<long double>(usable) / kLowClockSavedThresholdMs
            : 1.0L;
    const long double saved_to_spend = saved_ms_ * saved_scale;
    // Saved time is already in `usable`, so set aside the portion we will
    // spend now before splitting the remaining clock across future moves.
    const long double base_usable =
        std::max<long double>(0.0L, usable - saved_to_spend);
    budget = base_usable * fraction + increment * (1.0L - fraction);
    if (config.first_move_of_game) {
      budget *= 1.0L + kFirstMoveBonus;
      if (!has_explicit_movestogo) {
        budget = std::min(budget, usable * kMaxSingleMoveFraction);
      }
    }
    budget += saved_to_spend;
  }
  const auto clamped = static_cast<int64_t>(
      std::clamp(budget, 1.0L, static_cast<long double>(usable)));
  deadline_ = start_time + std::chrono::milliseconds(clamped);
}

bool TimeManager::CanAffordVisits(uint64_t visits, uint64_t visits_done,
                                  Clock::time_point now) const {
  if (!clock_budget_ || smart_pruning_factor_ <= 0.0f) return true;
  if (visits_done == 0) return true;
  if (!first_eval_time_) {
    first_eval_time_ = now;
    return true;
  }
  const auto elapsed = now - *first_eval_time_;
  if (elapsed < kSmartPruningTolerance) return true;
  if (now >= *deadline_) return visits == 0;
  const double rate_visits =
      static_cast<double>(visits_done + kSmartPruningToleranceNodes);
  const double remaining = rate_visits * (*deadline_ - now).count() /
                           elapsed.count() / smart_pruning_factor_;
  return visits <= remaining;
}

int64_t TimeManager::UnusedMs(Clock::time_point stop_time) const {
  if (!clock_budget_) return saved_ms_;
  return std::max<int64_t>(
      0, std::chrono::duration_cast<std::chrono::milliseconds>(*deadline_ -
                                                               stop_time)
             .count());
}

}  // namespace lczero::lc5
