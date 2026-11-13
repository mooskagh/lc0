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

#include <chrono>
#include <cstdint>
#include <optional>

namespace lczero {

struct GoParams;

namespace lc5 {

class TimeManager {
 public:
  struct Config {
    int64_t move_overhead_ms;
    float alphazero_time_pct;
    // Remaining visits are divided by this when deciding whether the best
    // root move can still be overtaken; 0 disables early stopping.
    float smart_pruning_factor = 1.33f;
    // Current position's game ply (0 at startpos), used for the move horizon.
    int game_ply = 0;
    // Whether this is the first timed move of the game (invests extra time
    // to populate the persistent search graph).
    bool first_move_of_game = false;
  };
  using Clock = std::chrono::steady_clock;

  // `saved_ms`: unused time from earlier moves, added to a clock budget.
  TimeManager(Config config, const GoParams& params, bool black_to_move,
              Clock::time_point start_time, int64_t saved_ms = 0);

  // When the search must stop; nullopt if only an external stop ends it.
  const std::optional<Clock::time_point>& deadline() const { return deadline_; }
  bool ShouldStop(Clock::time_point now) const {
    return deadline_ && now >= *deadline_;
  }
  // Whether `visits` more visits likely fit before the deadline, at the rate of
  // `visits_done` so far over the pruning factor. True if there's no estimate.
  bool CanAffordVisits(uint64_t visits, uint64_t visits_done,
                       Clock::time_point now) const;
  // Time to save for the next move if the search ends at `stop_time`: the
  // rest of a clock budget, or `saved_ms` untouched if there's none.
  int64_t UnusedMs(Clock::time_point stop_time) const;

 private:
  const Clock::time_point start_time_;
  const int64_t saved_ms_;
  std::optional<Clock::time_point> deadline_;
  // The budget was allocated from the clock (not movetime, ponder, infinite).
  bool clock_budget_ = false;
  float smart_pruning_factor_ = 0.0f;
  // First time a completed visit was observed; smart pruning waits for a
  // warmup window after this point before trusting the visit rate.
  mutable std::optional<Clock::time_point> first_eval_time_;
};

}  // namespace lc5
}  // namespace lczero
