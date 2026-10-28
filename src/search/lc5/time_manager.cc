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

#include "chess/uciloop.h"

namespace lczero::lc5 {
namespace {

std::optional<int64_t> GetTimeBudget(TimeManager::Config config,
                                     const GoParams& params, bool black_to_move,
                                     int64_t saved_ms) {
  if (params.ponder) return std::nullopt;
  if (params.movetime) {
    return std::max<int64_t>(0, *params.movetime - config.move_overhead_ms);
  }
  if (params.infinite) return std::nullopt;
  const auto& remaining = black_to_move ? params.btime : params.wtime;
  if (!remaining) return std::nullopt;
  if (*remaining <= config.move_overhead_ms) return 0;

  const int64_t usable = *remaining - config.move_overhead_ms;
  const int64_t increment = std::max<int64_t>(
      0, (black_to_move ? params.binc : params.winc).value_or(0));
  long double fraction = config.alphazero_time_pct / 100.0L;
  if (params.movestogo && *params.movestogo > 0) {
    fraction = std::max(fraction, 1.0L / *params.movestogo);
  }
  // Divide the clock and future increments over the implied move horizon.
  // The current move's increment is only received after we finish searching.
  const long double budget = usable * fraction + increment * (1 - fraction) +
                             std::max<int64_t>(0, saved_ms);
  return static_cast<int64_t>(
      std::clamp(budget, 1.0L, static_cast<long double>(usable)));
}

}  // namespace

TimeManager::TimeManager(Config config, const GoParams& params,
                         bool black_to_move,
                         std::chrono::steady_clock::time_point start_time,
                         int64_t saved_ms) {
  if (const auto budget =
          GetTimeBudget(config, params, black_to_move, saved_ms)) {
    deadline_ = start_time + std::chrono::milliseconds(*budget);
    if (!params.movetime) smart_pruning_factor_ = config.smart_pruning_factor;
  }
}

TimeManager::Decision TimeManager::Evaluate(
    std::chrono::steady_clock::time_point now) const {
  if (deadline_ && now >= *deadline_) {
    return {.should_stop = true, .next_check = std::nullopt};
  }
  return {.should_stop = false, .next_check = deadline_};
}

}  // namespace lczero::lc5
