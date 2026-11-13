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
  struct State {
    float nps = 20000.0f;
    bool nps_is_reliable = false;
    float tree_reuse = 0.52f;
    float timeuse = 0.70f;
    float piggybank_ms = 0.0f;
    bool first_timed_move = true;
    uint64_t last_move_total_nodes = 0;
    float last_move_elapsed_ms = 0.0f;
    float avg_ms_per_move = 0.0f;
  };
  using Clock = std::chrono::steady_clock;
  static constexpr float kDefaultAlphaZeroTimePct = 3.0f;

  // `saved_ms`: unused time from earlier moves (used when overriding
  // `alphazero_time_pct` away from its default).
  TimeManager(Config config, const GoParams& params, bool black_to_move,
              Clock::time_point start_time, int64_t saved_ms = 0);

  // Smooth cross-move time manager state constructor. `reused_root_visits` is
  // the visit count already present at the root from prior searches.
  TimeManager(Config config, const GoParams& params, bool black_to_move,
              Clock::time_point start_time, State state,
              uint64_t reused_root_visits = 0);

  // Base move deadline; nullopt if only an external stop ends the search.
  const std::optional<Clock::time_point>& deadline() const { return deadline_; }
  const std::optional<Clock::time_point>& base_deadline() const {
    return deadline_;
  }
  // Extended deadline allowed when a root crisis opens the piggybank.
  const std::optional<Clock::time_point>& crisis_deadline() const {
    return crisis_deadline_;
  }
  bool HasCrisisBudget() const {
    return crisis_deadline_ && deadline_ && *crisis_deadline_ > *deadline_;
  }
  bool PastHalfBaseBudget(Clock::time_point now) const;
  void SetCrisisActive(bool in_crisis, Clock::time_point now);
  bool ShouldStop(Clock::time_point now, bool in_crisis = false) const;
  // Estimated remaining visits before the active deadline using a smooth prior
  // blend of `state.nps` and in-move visit rate divided by
  // `smart_pruning_factor_`. Returns `UINT64_MAX` when early stopping is
  // disabled, and `0` at or past the active deadline.
  uint64_t EstimatedRemainingVisits(uint64_t visits_done, Clock::time_point now,
                                    bool in_crisis = false) const;
  // Whether `visits` more visits likely fit before the active deadline, at the
  // blended visit rate over the pruning factor. True if there's no estimate.
  bool CanAffordVisits(uint64_t visits, uint64_t visits_done,
                       Clock::time_point now, bool in_crisis = false) const;
  // Time to save for the next move if the search ends at `stop_time`: the
  // rest of a clock budget, or `saved_ms` untouched if there's none.
  int64_t UnusedMs(Clock::time_point stop_time) const;
  // Computes updated cross-move state at end of search (`new_visits` during
  // this move, `final_root_visits` total visits at the root including reuse).
  State FinalizeState(Clock::time_point stop_time, uint64_t new_visits,
                      uint64_t final_root_visits) const;

 private:
  TimeManager(Config config, const GoParams& params, bool black_to_move,
              Clock::time_point start_time, State state,
              uint64_t reused_root_visits, int64_t saved_ms);
  std::optional<Clock::time_point> ActiveDeadline(bool in_crisis) const;

  const Clock::time_point start_time_;
  const State initial_state_;
  const int64_t saved_ms_;
  std::optional<Clock::time_point> deadline_;
  std::optional<Clock::time_point> crisis_deadline_;
  // The budget was allocated from the clock (not movetime, ponder, infinite).
  bool clock_budget_ = false;
  bool smooth_budget_ = false;
  float smart_pruning_factor_ = 0.0f;
  // Working smooth-state quantities carried from constructor to FinalizeState.
  float working_tree_reuse_ = initial_state_.tree_reuse;
  float updated_piggybank_ms_ = 0.0f;
  float avg_ms_per_move_ = 0.0f;
  int64_t base_budget_ms_ = 0;
  bool crisis_active_ = false;
  mutable bool used_piggybank_ = false;
};

}  // namespace lc5
}  // namespace lczero
