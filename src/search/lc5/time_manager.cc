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
#include <limits>

#include "chess/uciloop.h"

namespace lczero::lc5 {
namespace {

// Smooth time manager parameters ported from SmoothTimeManager with an
// explicit safety buffer beyond `move_overhead_ms` so clock flags cannot occur.
constexpr float kTimeCurveMidpoint = 45.2f;
constexpr float kTimeCurveSteepness = 5.93f;
constexpr float kMaxTreeReuse = 0.73f;
constexpr float kTreeReuseUpdateRate = 3.39f;
constexpr float kMinTimeUse = 0.34f;
constexpr float kTimeUseUpdateRate = 5.51f;
constexpr float kMaxSingleMoveFraction = 0.42f;
constexpr float kInitPiggybankFraction = 0.09f;
constexpr float kPerMovePiggybankFraction = 0.12f;
constexpr float kMaxPiggybankUse = 0.94f;
constexpr float kMaxPiggybankMoves = 36.5f;
constexpr float kMaxCrisisMultiplier = 3.0f;
constexpr int64_t kSafetyBufferMs = 30;
constexpr float kNpsUpdatePeriodMs = 20000.0f;
constexpr double kReliableNpsBlendTauSec = 0.05;
constexpr double kInitialNpsBlendTauSec = 0.015;

double ComputeEstimatedMovesToGo(int ply, float midpoint, float steepness) {
  const double move = std::max(0, ply) / 2.0;
  return midpoint *
             std::pow(1.0 + 2.0 * std::pow(move / midpoint, steepness),
                      1.0 / steepness) -
         move;
}

double ExponentialDecay(double from, double to, double step, double value) {
  if (step <= 0.0 || value <= 0.0) return from;
  return to - (to - from) * std::pow(0.5, value / step);
}

double LinearDecay(double cur_value, double target_value, double elapsed_ms,
                   double window_size_ms) {
  if (cur_value <= target_value) return target_value;
  if (elapsed_ms <= 0.0 || window_size_ms <= 0.0) return cur_value;
  if (elapsed_ms >= window_size_ms) return target_value;
  return cur_value + (target_value - cur_value) * (elapsed_ms / window_size_ms);
}

}  // namespace

TimeManager::TimeManager(Config config, const GoParams& params,
                         bool black_to_move, Clock::time_point start_time,
                         int64_t saved_ms)
    : TimeManager(config, params, black_to_move, start_time, State{}, 0,
                  saved_ms) {}

TimeManager::TimeManager(Config config, const GoParams& params,
                         bool black_to_move, Clock::time_point start_time,
                         State state, uint64_t reused_root_visits)
    : TimeManager(config, params, black_to_move, start_time, state,
                  reused_root_visits, 0) {}

TimeManager::TimeManager(Config config, const GoParams& params,
                         bool black_to_move, Clock::time_point start_time,
                         State state, uint64_t reused_root_visits,
                         int64_t saved_ms)
    : start_time_(start_time),
      initial_state_(state),
      saved_ms_(std::max<int64_t>(0, saved_ms)),
      working_tree_reuse_(state.tree_reuse),
      updated_piggybank_ms_(std::max(0.0f, state.piggybank_ms)) {
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

  if (config.alphazero_time_pct != kDefaultAlphaZeroTimePct) {
    // Preserve explicit AlphaZeroTimePct customization as a flat percentage.
    if (*remaining <= config.move_overhead_ms) {
      deadline_ = start_time;
      return;
    }
    const int64_t usable = *remaining - config.move_overhead_ms;
    const int64_t increment = std::max<int64_t>(
        0, (black_to_move ? params.binc : params.winc).value_or(0));
    long double fraction = config.alphazero_time_pct / 100.0L;
    if (params.movestogo && *params.movestogo > 0) {
      fraction = std::max(fraction, 1.0L / *params.movestogo);
    }
    const long double budget =
        usable * fraction + increment * (1.0L - fraction) + saved_ms_;
    const auto clamped = static_cast<int64_t>(
        std::clamp(budget, 1.0L, static_cast<long double>(usable)));
    deadline_ = start_time + std::chrono::milliseconds(clamped);
    base_budget_ms_ = clamped;
    return;
  }

  smooth_budget_ = true;
  const int64_t safe_usable = std::max<int64_t>(
      0, *remaining - config.move_overhead_ms - kSafetyBufferMs);
  if (safe_usable <= 0) {
    deadline_ = start_time;
    crisis_deadline_ = start_time;
    return;
  }

  double piggybank = std::max(0.0, static_cast<double>(state.piggybank_ms));
  if (state.first_timed_move) {
    piggybank = static_cast<double>(safe_usable) * kInitPiggybankFraction;
  } else if (state.last_move_total_nodes > 0 && state.avg_ms_per_move > 0.0f) {
    const double this_move_time_fraction =
        std::max(0.0, static_cast<double>(state.last_move_elapsed_ms) /
                          static_cast<double>(state.avg_ms_per_move));
    const double observed_reuse = std::clamp(
        static_cast<double>(reused_root_visits) /
            static_cast<double>(state.last_move_total_nodes),
        0.0, 1.0);
    working_tree_reuse_ = static_cast<float>(std::clamp(
        ExponentialDecay(state.tree_reuse, observed_reuse, kTreeReuseUpdateRate,
                         this_move_time_fraction),
        0.0, static_cast<double>(kMaxTreeReuse)));
  }

  double remaining_moves = ComputeEstimatedMovesToGo(
      config.game_ply, kTimeCurveMidpoint, kTimeCurveSteepness);
  if (params.movestogo && *params.movestogo > 0 &&
      *params.movestogo < remaining_moves) {
    remaining_moves = static_cast<double>(*params.movestogo);
  }
  remaining_moves = std::max(remaining_moves, 1.0);

  const int64_t increment = std::max<int64_t>(
      0, (black_to_move ? params.binc : params.winc).value_or(0));
  const double max_piggybank_time =
      static_cast<double>(kMaxPiggybankMoves) *
      std::max(0.0, static_cast<double>(safe_usable) +
                        static_cast<double>(increment) *
                            (remaining_moves - 1.0)) /
      remaining_moves;
  piggybank = std::clamp(piggybank, 0.0, max_piggybank_time);

  const double total_remaining_ms = std::max(
      0.0, static_cast<double>(safe_usable) - piggybank +
               static_cast<double>(increment) * (remaining_moves - 1.0));
  avg_ms_per_move_ = static_cast<float>(total_remaining_ms / remaining_moves);

  const double nps = std::max(1.0, static_cast<double>(state.nps));
  const double avg_nodes_per_move =
      (static_cast<double>(avg_ms_per_move_) * nps) / 1000.0;
  const double nodes_per_move_including_reuse =
      avg_nodes_per_move /
      std::max(0.05, 1.0 - static_cast<double>(working_tree_reuse_));
  const double move_estimate_nodes = std::max(
      0.0,
      nodes_per_move_including_reuse - static_cast<double>(reused_root_visits));
  const double expected_movetime_ms_brutto =
      (move_estimate_nodes / nps) * 1000.0;
  const double time_to_piggybank_ms =
      std::min(std::max(0.0, max_piggybank_time - piggybank),
               expected_movetime_ms_brutto *
                   static_cast<double>(kPerMovePiggybankFraction));
  const double expected_movetime_ms =
      expected_movetime_ms_brutto - time_to_piggybank_ms;
  const double unscaled_base_ms =
      expected_movetime_ms / std::max(0.05, static_cast<double>(state.timeuse));

  double max_fraction = static_cast<double>(kMaxSingleMoveFraction);
  if (params.movestogo && *params.movestogo > 0) {
    max_fraction = std::max(max_fraction, 1.0 / *params.movestogo);
  }
  const int64_t max_base_ms = std::clamp<int64_t>(
      static_cast<int64_t>(std::floor(static_cast<double>(*remaining) *
                                      max_fraction)),
      1, safe_usable);
  base_budget_ms_ = std::clamp<int64_t>(
      static_cast<int64_t>(std::llround(unscaled_base_ms)), 1, max_base_ms);

  const double allowed_piggybank_ms = std::min({
      piggybank * static_cast<double>(kMaxPiggybankUse),
      static_cast<double>(base_budget_ms_) *
          static_cast<double>(kMaxCrisisMultiplier),
      static_cast<double>(std::max<int64_t>(0, safe_usable - base_budget_ms_)),
  });
  const int64_t crisis_extra_ms =
      std::max<int64_t>(0, static_cast<int64_t>(std::floor(allowed_piggybank_ms)));
  updated_piggybank_ms_ = static_cast<float>(piggybank + time_to_piggybank_ms);

  deadline_ = start_time + std::chrono::milliseconds(base_budget_ms_);
  crisis_deadline_ =
      start_time + std::chrono::milliseconds(base_budget_ms_ + crisis_extra_ms);
}

std::optional<TimeManager::Clock::time_point> TimeManager::ActiveDeadline(
    bool in_crisis) const {
  if ((in_crisis || crisis_active_) && crisis_deadline_) {
    return crisis_deadline_;
  }
  return deadline_;
}

bool TimeManager::PastHalfBaseBudget(Clock::time_point now) const {
  if (!clock_budget_ || !deadline_) return false;
  return now >= start_time_ + (*deadline_ - start_time_) / 2;
}

void TimeManager::SetCrisisActive(bool in_crisis, Clock::time_point now) {
  crisis_active_ = in_crisis && HasCrisisBudget();
  if (crisis_active_ && deadline_ && now >= *deadline_) {
    used_piggybank_ = true;
  }
}

bool TimeManager::ShouldStop(Clock::time_point now, bool in_crisis) const {
  if (!deadline_) return false;
  const bool use_crisis = (in_crisis || crisis_active_) && HasCrisisBudget();
  if (use_crisis && now >= *deadline_) {
    used_piggybank_ = true;
  }
  const auto active = ActiveDeadline(in_crisis);
  return active && now >= *active;
}

uint64_t TimeManager::EstimatedRemainingVisits(uint64_t visits_done,
                                               Clock::time_point now,
                                               bool in_crisis) const {
  constexpr uint64_t kUnlimited = std::numeric_limits<uint64_t>::max();
  if (!clock_budget_ || smart_pruning_factor_ <= 0.0f) return kUnlimited;
  const auto active = ActiveDeadline(in_crisis);
  if (!active || now >= *active) return 0;

  const double elapsed_s = std::max(
      0.0, std::chrono::duration<double>(now - start_time_).count());
  const double tau_s = initial_state_.nps_is_reliable ? kReliableNpsBlendTauSec
                                                      : kInitialNpsBlendTauSec;
  const double prior_nps =
      std::max(1.0, static_cast<double>(initial_state_.nps));
  const double effective_nps =
      (prior_nps * tau_s + static_cast<double>(visits_done)) /
      (elapsed_s + tau_s);
  const double rem_s = std::chrono::duration<double>(*active - now).count();
  const double remaining =
      (effective_nps * rem_s) / static_cast<double>(smart_pruning_factor_);
  if (remaining >= static_cast<double>(kUnlimited)) return kUnlimited;
  return static_cast<uint64_t>(std::max(0.0, remaining));
}

bool TimeManager::CanAffordVisits(uint64_t visits, uint64_t visits_done,
                                  Clock::time_point now, bool in_crisis) const {
  return visits <= EstimatedRemainingVisits(visits_done, now, in_crisis);
}

int64_t TimeManager::UnusedMs(Clock::time_point stop_time) const {
  if (!clock_budget_ || !deadline_) return saved_ms_;
  return std::max<int64_t>(
      0, std::chrono::duration_cast<std::chrono::milliseconds>(*deadline_ -
                                                               stop_time)
             .count());
}

TimeManager::State TimeManager::FinalizeState(
    Clock::time_point stop_time, uint64_t new_visits,
    uint64_t final_root_visits) const {
  State next = initial_state_;
  if (!smooth_budget_) return next;

  const double elapsed_ms = std::max(
      0.0, std::chrono::duration<double, std::milli>(stop_time - start_time_)
               .count());
  if (base_budget_ms_ <= 0) {
    if (elapsed_ms >= 1.0 && new_visits > 0) {
      const double move_nps =
          std::max(1.0, 1000.0 * static_cast<double>(new_visits) / elapsed_ms);
      next.nps = static_cast<float>(
          next.nps_is_reliable
              ? LinearDecay(next.nps, move_nps, elapsed_ms, kNpsUpdatePeriodMs)
              : move_nps);
      next.nps_is_reliable = true;
    }
    return next;
  }

  next.tree_reuse = working_tree_reuse_;
  double piggybank = updated_piggybank_ms_;
  if (used_piggybank_) {
    const double piggybank_used =
        std::max(0.0, elapsed_ms - static_cast<double>(base_budget_ms_));
    piggybank = std::max(0.0, piggybank - piggybank_used);
  }
  next.piggybank_ms = static_cast<float>(piggybank);

  const double this_move_time_fraction =
      avg_ms_per_move_ > 0.0f
          ? elapsed_ms / static_cast<double>(avg_ms_per_move_)
          : 0.0;
  const double target_time_use =
      used_piggybank_
          ? 1.0
          : std::clamp(elapsed_ms / static_cast<double>(base_budget_ms_), 0.0,
                       1.0);
  next.timeuse = static_cast<float>(std::clamp(
      ExponentialDecay(next.timeuse, target_time_use, kTimeUseUpdateRate,
                       this_move_time_fraction),
      static_cast<double>(kMinTimeUse), 1.0));

  if (elapsed_ms >= 1.0 && new_visits > 0) {
    const double move_nps =
        std::max(1.0, 1000.0 * static_cast<double>(new_visits) / elapsed_ms);
    next.nps = static_cast<float>(
        next.nps_is_reliable
            ? LinearDecay(next.nps, move_nps, elapsed_ms, kNpsUpdatePeriodMs)
            : move_nps);
    next.nps_is_reliable = true;
  }

  next.first_timed_move = false;
  next.last_move_total_nodes = final_root_visits;
  next.last_move_elapsed_ms = static_cast<float>(elapsed_ms);
  next.avg_ms_per_move = avg_ms_per_move_;
  return next;
}

}  // namespace lczero::lc5
