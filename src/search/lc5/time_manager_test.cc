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

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "chess/uciloop.h"

namespace lczero::lc5 {
namespace {

using Clock = std::chrono::steady_clock;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;

constexpr TimeManager::Config kConfig{.move_overhead_ms = 200,
                                      .alphazero_time_pct = 3.0f};
const Clock::time_point kStart{std::chrono::seconds(10)};

int64_t BudgetMs(const TimeManager& manager) {
  return std::chrono::duration_cast<milliseconds>(*manager.deadline() - kStart)
      .count();
}

int64_t CrisisBudgetMs(const TimeManager& manager) {
  return std::chrono::duration_cast<milliseconds>(*manager.crisis_deadline() -
                                                  kStart)
      .count();
}

void ExpectDeadline(const GoParams& params, bool black_to_move,
                    int64_t budget_ms, TimeManager::Config config = kConfig) {
  const TimeManager manager(config, params, black_to_move, kStart);
  const auto deadline = kStart + milliseconds(budget_ms);
  EXPECT_EQ(manager.deadline(), deadline);
  EXPECT_FALSE(manager.ShouldStop(deadline - nanoseconds(1)));
  for (const auto now : {deadline, deadline + nanoseconds(1),
                         deadline + std::chrono::hours(1)}) {
    EXPECT_TRUE(manager.ShouldStop(now));
  }
  EXPECT_EQ(manager.ShouldStop(kStart), budget_ms <= 0);
}

void ExpectNoDeadline(const GoParams& params, bool black_to_move) {
  const TimeManager manager(kConfig, params, black_to_move, kStart);
  for (const auto now :
       {kStart - nanoseconds(1), kStart, kStart + std::chrono::hours(24)}) {
    EXPECT_FALSE(manager.ShouldStop(now));
  }
  EXPECT_EQ(manager.deadline(), std::nullopt);
}

TEST(Lc5TimeManagerTest, ComputesBudgetFromActiveClock) {
  const GoParams params{.wtime = 10200, .btime = 5200};
  const TimeManager white(kConfig, params, false, kStart, TimeManager::State{});
  const TimeManager black(kConfig, params, true, kStart, TimeManager::State{});
  ASSERT_TRUE(white.deadline().has_value());
  ASSERT_TRUE(black.deadline().has_value());
  EXPECT_GT(BudgetMs(white), BudgetMs(black));
  EXPECT_LE(CrisisBudgetMs(white), 10200 - 200 - 30);
  EXPECT_LE(CrisisBudgetMs(black), 5200 - 200 - 30);
  // 2x clock ratio scales budgets nearly linearly on identical state.
  EXPECT_NEAR(static_cast<double>(BudgetMs(white)), 2.0 * BudgetMs(black), 25.0);
}

TEST(Lc5TimeManagerTest, TimingConfigurationIsUsed) {
  constexpr TimeManager::Config config{.move_overhead_ms = 100,
                                       .alphazero_time_pct = 25.0f};
  ExpectDeadline(GoParams{.wtime = 4100}, false, 1000, config);
  ExpectDeadline(GoParams{.wtime = 50}, false, 0, config);
  ExpectDeadline(GoParams{.wtime = 400}, false, 400,
                 {.move_overhead_ms = 0, .alphazero_time_pct = 100.0f});
}

TEST(Lc5TimeManagerTest, IncludesOnlyActiveSideIncrement) {
  const TimeManager base(kConfig, GoParams{.wtime = 10200, .btime = 5200}, false,
                         kStart, TimeManager::State{});
  const TimeManager with_winc(
      kConfig,
      GoParams{.wtime = 10200, .btime = 5200, .winc = 100, .binc = 200}, false,
      kStart, TimeManager::State{});
  const TimeManager binc_only(
      kConfig, GoParams{.wtime = 10200, .btime = 5200, .binc = 200}, false,
      kStart, TimeManager::State{});
  const TimeManager neg_winc(
      kConfig, GoParams{.wtime = 10200, .btime = 5200, .winc = -100}, false,
      kStart, TimeManager::State{});
  EXPECT_GT(BudgetMs(with_winc), BudgetMs(base));
  EXPECT_EQ(BudgetMs(binc_only), BudgetMs(base));
  EXPECT_EQ(BudgetMs(neg_winc), BudgetMs(base));

  const TimeManager black_base(kConfig, GoParams{.wtime = 10200, .btime = 5200},
                               true, kStart, TimeManager::State{});
  const TimeManager black_binc(
      kConfig, GoParams{.wtime = 10200, .btime = 5200, .binc = 200}, true,
      kStart, TimeManager::State{});
  const TimeManager black_neg(
      kConfig, GoParams{.wtime = 10200, .btime = 5200, .binc = -200}, true,
      kStart, TimeManager::State{});
  EXPECT_GT(BudgetMs(black_binc), BudgetMs(black_base));
  EXPECT_EQ(BudgetMs(black_neg), BudgetMs(black_base));
}

TEST(Lc5TimeManagerTest, HonorsShorterMovesToGoHorizon) {
  const TimeManager unconstrained(kConfig, GoParams{.wtime = 4200, .winc = 100},
                                  false, kStart, TimeManager::State{});
  const TimeManager short_horizon(
      kConfig, GoParams{.wtime = 4200, .winc = 100, .movestogo = 4}, false,
      kStart, TimeManager::State{});
  EXPECT_GT(BudgetMs(short_horizon), BudgetMs(unconstrained));
  // 1 move left in time control uses the full safe clock buffer.
  ExpectDeadline(GoParams{.wtime = 4200, .winc = 100, .movestogo = 1}, false,
                 4200 - 200 - 30);
  for (int moves : {0, -1, 100}) {
    SCOPED_TRACE(moves);
    const TimeManager ignored(
        kConfig, GoParams{.wtime = 4200, .winc = 100, .movestogo = moves}, false,
        kStart, TimeManager::State{});
    EXPECT_EQ(BudgetMs(ignored), BudgetMs(unconstrained));
  }
}

TEST(Lc5TimeManagerTest, CapsBudgetAndPreservesOverheadAndSafetyBuffer) {
  const TimeManager high_inc(kConfig, GoParams{.wtime = 300, .winc = 10000},
                             false, kStart, TimeManager::State{});
  // Smooth mode reserves move_overhead (200) + safety buffer (30).
  EXPECT_LE(BudgetMs(high_inc), 70);
  EXPECT_LE(CrisisBudgetMs(high_inc), 70);
  ExpectDeadline(GoParams{.wtime = 231, .winc = 80}, false, 1);
  for (int64_t remaining : {230, 200, 100, 0, -100}) {
    SCOPED_TRACE(remaining);
    ExpectDeadline(GoParams{.wtime = remaining, .winc = 80}, false, 0);
  }
  constexpr TimeManager::Config config{.move_overhead_ms = 200,
                                       .alphazero_time_pct = 0.0f};
  ExpectDeadline(GoParams{.wtime = 1000}, false, 1, config);
  ExpectDeadline(GoParams{.wtime = 1000, .winc = 80}, false, 80, config);
}

TEST(Lc5TimeManagerTest, PreservesPrecisionAndTruncation) {
  // 3.1f must be promoted directly to long double, not rounded to 3.1.
  ExpectDeadline(GoParams{.wtime = 1000200}, false, 30999,
                 {.move_overhead_ms = 200, .alphazero_time_pct = 3.1f});
  ExpectDeadline(GoParams{.wtime = 4201, .winc = 101, .movestogo = 4}, false,
                 1076, {.move_overhead_ms = 200, .alphazero_time_pct = 5.0f});
}

TEST(Lc5TimeManagerTest, IncrementMaintainsZeroTimeLossOverLongGame) {
  // Simulate 120 moves at 8.0+0.08 STC including PV reuse and periodic crises.
  GoParams params{.wtime = 8000, .winc = 80};
  TimeManager::State state;
  uint64_t prev_final_visits = 0;
  for (int move = 0; move < 120; ++move) {
    SCOPED_TRACE(move);
    TimeManager::Config config = kConfig;
    config.game_ply = move * 2;
    const uint64_t reused =
        move == 0 ? 0 : static_cast<uint64_t>(prev_final_visits * 0.50);
    TimeManager manager(config, params, false, kStart, state, reused);
    ASSERT_FALSE(manager.ShouldStop(kStart));
    ASSERT_TRUE(manager.deadline().has_value());
    const bool crisis = (move % 7 == 3) && manager.HasCrisisBudget();
    const int64_t spent =
        crisis ? CrisisBudgetMs(manager)
               : std::max<int64_t>(1, static_cast<int64_t>(BudgetMs(manager) *
                                                           0.72));
    ASSERT_GT(spent, 0);
    // Strict timeloss==0 guarantee: always keep >= overhead + 30ms buffer.
    ASSERT_LE(spent, *params.wtime - 230);
    if (crisis) {
      EXPECT_FALSE(manager.ShouldStop(kStart + milliseconds(BudgetMs(manager)),
                                      true));
      EXPECT_TRUE(manager.ShouldStop(kStart + milliseconds(spent), true));
    }
    const uint64_t new_visits = static_cast<uint64_t>(spent * 40);
    prev_final_visits = reused + new_visits;
    state = manager.FinalizeState(kStart + milliseconds(spent), new_visits,
                                  prev_final_visits);
    *params.wtime += *params.winc - spent;
    EXPECT_GT(*params.wtime, 230);
  }
}

TEST(Lc5TimeManagerTest, RequiresActiveSideClock) {
  ExpectNoDeadline(GoParams{}, false);
  ExpectNoDeadline(GoParams{}, true);
  ExpectNoDeadline(GoParams{.btime = 1000, .winc = 100}, false);
  ExpectNoDeadline(GoParams{.wtime = 1000, .binc = 100}, true);
}

TEST(Lc5TimeManagerTest, ExplicitMovetimeMinusOverheadWinsOverClocks) {
  for (bool black_to_move : {false, true}) {
    for (int64_t movetime : {500, 201, 200, 1, 0, -1, -500}) {
      SCOPED_TRACE(black_to_move);
      SCOPED_TRACE(movetime);
      const int64_t budget = std::max<int64_t>(0, movetime - 200);
      ExpectDeadline(GoParams{.movetime = movetime}, black_to_move, budget);
      ExpectDeadline(GoParams{.wtime = 0,
                              .btime = 0,
                              .winc = 10000,
                              .binc = 10000,
                              .movestogo = 1,
                              .movetime = movetime},
                     black_to_move, budget);
      ExpectDeadline(GoParams{.movetime = movetime}, black_to_move,
                     std::max<int64_t>(0, movetime),
                     {.move_overhead_ms = 0, .alphazero_time_pct = 3.0f});
    }
  }
}

TEST(Lc5TimeManagerTest, InfiniteSuppressesOnlyClockAllocation) {
  for (bool black_to_move : {false, true}) {
    for (int64_t remaining : {0, 10000}) {
      SCOPED_TRACE(black_to_move);
      SCOPED_TRACE(remaining);
      GoParams params{.wtime = remaining, .btime = remaining, .infinite = true};
      ExpectNoDeadline(params, black_to_move);
      for (int64_t movetime : {500, 0, -1}) {
        SCOPED_TRACE(movetime);
        params.movetime = movetime;
        ExpectDeadline(params, black_to_move,
                       std::max<int64_t>(0, movetime - 200));
      }
    }
  }
}

TEST(Lc5TimeManagerTest, PonderSuppressesAllTimeLimits) {
  for (bool black_to_move : {false, true}) {
    for (bool infinite : {false, true}) {
      for (int64_t remaining : {0, 10000}) {
        for (const std::optional<int64_t> movetime :
             {std::optional<int64_t>{}, std::optional<int64_t>{500},
              std::optional<int64_t>{0}, std::optional<int64_t>{-1}}) {
          SCOPED_TRACE(black_to_move);
          SCOPED_TRACE(infinite);
          SCOPED_TRACE(remaining);
          SCOPED_TRACE(movetime.value_or(-2));
          ExpectNoDeadline(GoParams{.wtime = remaining,
                                    .btime = remaining,
                                    .movetime = movetime,
                                    .infinite = infinite,
                                    .ponder = true},
                           black_to_move);
        }
      }
    }
  }
}

TEST(Lc5TimeManagerTest, NodeParametersDoNotAffectTimePolicy) {
  const int64_t expected_clock =
      BudgetMs(TimeManager(kConfig, GoParams{.wtime = 10200}, false, kStart));
  for (int nodes : {-1, 0, 1, 1000}) {
    SCOPED_TRACE(nodes);
    ExpectNoDeadline(GoParams{.nodes = nodes}, false);
    ExpectDeadline(GoParams{.wtime = 10200, .nodes = nodes}, false,
                   expected_clock);
    ExpectDeadline(GoParams{.nodes = nodes, .movetime = 500}, false, 300);
    ExpectNoDeadline(GoParams{.wtime = 0, .nodes = nodes, .infinite = true},
                     false);
    ExpectDeadline(
        GoParams{.wtime = 0, .nodes = nodes, .movetime = 500, .infinite = true},
        false, 300);
    ExpectNoDeadline(
        GoParams{.wtime = 0, .nodes = nodes, .movetime = 0, .ponder = true},
        false);
  }
}

TEST(Lc5TimeManagerTest, SuppliedStartTimeChargesPreparation) {
  const TimeManager manager(kConfig, GoParams{.movetime = 700}, false, kStart);
  EXPECT_FALSE(manager.ShouldStop(kStart + milliseconds(499)));
  EXPECT_EQ(manager.deadline(), kStart + milliseconds(500));
  EXPECT_TRUE(manager.ShouldStop(kStart + milliseconds(600)));
}

TEST(Lc5TimeManagerTest, ExplicitAlphaZeroTimePctUsesSavedTimeAndIgnoresCurve) {
  const TimeManager::Config custom{.move_overhead_ms = 200,
                                   .alphazero_time_pct = 25.0f,
                                   .game_ply = 80,
                                   .first_move_of_game = true};
  // 10000 * 0.25 = 2500 without saved_ms; + 400 saved = 2900.
  ExpectDeadline(GoParams{.wtime = 10200}, false, 2500, custom);
  const TimeManager with_saved(custom, GoParams{.wtime = 10200}, false, kStart,
                               400);
  EXPECT_EQ(with_saved.deadline(), kStart + milliseconds(2900));
  // Explicit 5.0f ignores game_ply/first_move_of_game too.
  ExpectDeadline(GoParams{.wtime = 10200}, false, 500,
                 {.move_overhead_ms = 200,
                  .alphazero_time_pct = 5.0f,
                  .game_ply = 80,
                  .first_move_of_game = true});
  // Movetime never uses saved time.
  const TimeManager movetime(custom, GoParams{.movetime = 700}, false, kStart,
                             400);
  EXPECT_EQ(movetime.deadline(), kStart + milliseconds(500));
}

TEST(Lc5TimeManagerTest, SavedTimeIsKeptWithoutClockBudget) {
  for (const GoParams& params : {GoParams{.wtime = 10200, .ponder = true},
                                 GoParams{.wtime = 10200, .infinite = true},
                                 GoParams{.wtime = 10200, .movetime = 700},
                                 GoParams{.nodes = 100}, GoParams{}}) {
    const TimeManager manager(kConfig, params, false, kStart, 400);
    EXPECT_EQ(manager.UnusedMs(kStart + milliseconds(10)), 400);
    EXPECT_EQ(manager.UnusedMs(kStart + std::chrono::hours(1)), 400);
  }
}

TEST(Lc5TimeManagerTest, UnusedTimeIsRestOfClockBudget) {
  const TimeManager manager(
      {.move_overhead_ms = 200, .alphazero_time_pct = 10.0f},
      GoParams{.wtime = 10200}, false, kStart, 400);
  // Budget = 1000 + 400 = 1400 ms.
  EXPECT_EQ(manager.UnusedMs(kStart), 1400);
  EXPECT_EQ(manager.UnusedMs(kStart + milliseconds(550)), 850);
  EXPECT_EQ(manager.UnusedMs(kStart + milliseconds(1400)), 0);
  EXPECT_EQ(manager.UnusedMs(kStart + milliseconds(1600)), 0);
}

TEST(Lc5TimeManagerTest, AnyVisitsAffordableWithoutEstimate) {
  const auto now = kStart + milliseconds(100);
  for (const GoParams& params :
       {GoParams{.wtime = 10200, .ponder = true},
        GoParams{.wtime = 10200, .infinite = true},
        GoParams{.wtime = 10200, .movetime = 700}, GoParams{}}) {
    const TimeManager manager(kConfig, params, false, kStart);
    EXPECT_TRUE(manager.CanAffordVisits(1'000'000, 1000, now));
  }
  TimeManager::Config disabled = kConfig;
  disabled.smart_pruning_factor = 0.0f;
  const TimeManager manager(disabled, GoParams{.wtime = 10200}, false, kStart);
  EXPECT_TRUE(manager.CanAffordVisits(1'000'000, 1000, now));
}

TEST(Lc5TimeManagerTest, EstimatedRemainingVisitsAccurateFromZeroMs) {
  constexpr uint64_t kUnlimited = std::numeric_limits<uint64_t>::max();
  for (const GoParams& params :
       {GoParams{.wtime = 10200, .ponder = true},
        GoParams{.wtime = 10200, .infinite = true},
        GoParams{.wtime = 10200, .movetime = 700}, GoParams{}}) {
    const TimeManager unlimited(kConfig, params, false, kStart);
    EXPECT_EQ(
        unlimited.EstimatedRemainingVisits(1000, kStart + milliseconds(100)),
        kUnlimited);
  }

  // Budget 1000 ms, smart_pruning_factor = 2.0f, reliable nps = 40,000.
  TimeManager::Config config = kConfig;
  config.alphazero_time_pct = 10.0f;
  config.smart_pruning_factor = 2.0f;
  TimeManager::State reliable_state;
  reliable_state.nps = 40000.0f;
  reliable_state.nps_is_reliable = true;
  const TimeManager manager(config, GoParams{.wtime = 10200}, false, kStart,
                            reliable_state, 0);
  // At t=0ms with visits_done=0: immediately returns blended prior estimate
  // (no 80ms blind warmup or +200 tolerance nodes).
  // (40000 * 0.05 + 0) / 0.05 * 1.0 / 2.0 = 20000.
  EXPECT_EQ(manager.EstimatedRemainingVisits(0, kStart), 20000u);
  EXPECT_TRUE(manager.CanAffordVisits(20000u, 0, kStart));
  EXPECT_FALSE(manager.CanAffordVisits(20001u, 0, kStart));

  // At t=50ms (tau=50ms, rem=950ms), 4000 observed visits (80k nps):
  // effective_nps = (40000*0.05 + 4000) / (0.05 + 0.05) = 60,000 nps.
  // remaining = 60000 * 0.95 / 2.0 = 28,500.
  const auto at_50ms = kStart + milliseconds(50);
  EXPECT_EQ(manager.EstimatedRemainingVisits(4000, at_50ms), 28500u);
  EXPECT_TRUE(manager.CanAffordVisits(28500u, 4000, at_50ms));
  EXPECT_FALSE(manager.CanAffordVisits(28501u, 4000, at_50ms));

  // At or past deadline, 0 visits remain.
  EXPECT_EQ(manager.EstimatedRemainingVisits(600, kStart + milliseconds(1000)),
            0u);
  EXPECT_TRUE(manager.CanAffordVisits(0u, 600, kStart + milliseconds(1000)));
  EXPECT_FALSE(manager.CanAffordVisits(1u, 600, kStart + milliseconds(1000)));
}

TEST(Lc5TimeManagerTest, SmoothCurveScalesWithGamePlyAndCapsAt42Pct) {
  auto budget_at = [](int ply, int64_t wtime = 8000) {
    TimeManager::Config config = kConfig;
    config.game_ply = ply;
    const TimeManager manager(config, GoParams{.wtime = wtime}, false, kStart,
                              TimeManager::State{});
    return BudgetMs(manager);
  };
  const int64_t ply0 = budget_at(0);
  const int64_t ply40 = budget_at(40);
  const int64_t ply80 = budget_at(80);
  const int64_t ply160 = budget_at(160);
  EXPECT_GT(ply40, ply0);
  EXPECT_GT(ply80, ply40);
  EXPECT_GE(ply160, ply80);
  // Sudden-death cap of 42% of remaining time (and <= safe_usable).
  EXPECT_LE(budget_at(240, 8000), static_cast<int64_t>(8000 * 0.42));
  EXPECT_LE(budget_at(240, 8000), 8000 - 200 - 30);
  // Negative ply clamps to 0.
  EXPECT_EQ(budget_at(-5), ply0);
}

TEST(Lc5TimeManagerTest, TreeReuseReducesPVBudgetAndFundsSurpriseMoves) {
  TimeManager::State state;
  state.first_timed_move = false;
  state.nps = 40000.0f;
  state.nps_is_reliable = true;
  state.tree_reuse = 0.52f;
  state.timeuse = 0.70f;
  state.piggybank_ms = 600.0f;
  state.last_move_total_nodes = 20000;
  state.last_move_elapsed_ms = 250.0f;
  state.avg_ms_per_move = 250.0f;

  const GoParams params{.wtime = 8000, .winc = 80};
  const TimeManager surprise(kConfig, params, false, kStart, state, 0);
  const TimeManager half_reuse(kConfig, params, false, kStart, state, 10000);
  const TimeManager deep_pv(kConfig, params, false, kStart, state, 30000);

  // Surprise move (0 reused visits) gets strictly more base time than PV reuse.
  EXPECT_GT(BudgetMs(surprise), BudgetMs(half_reuse));
  // Heavy PV reuse already exceeded target visits, finishing cleanly >= 1ms.
  EXPECT_EQ(BudgetMs(deep_pv), 1);

  // Observed high reuse raises working tree_reuse, capped at kMaxTreeReuse=0.73.
  state.last_move_elapsed_ms = 2000.0f;
  const TimeManager high_reuse(kConfig, params, false, kStart, state, 20000);
  const auto updated =
      high_reuse.FinalizeState(kStart + milliseconds(100), 4000, 24000);
  EXPECT_GT(updated.tree_reuse, 0.52f);
  EXPECT_LE(updated.tree_reuse, 0.7301f);
}

TEST(Lc5TimeManagerTest, PiggybankAccumulatesAndCapsAtThreeTimesBaseOnCrisis) {
  // Move 1 initializes piggybank at 9% of safe clock and adds 12% per move.
  const TimeManager move1(kConfig, GoParams{.wtime = 8000, .winc = 80}, false,
                          kStart, TimeManager::State{}, 0);
  EXPECT_TRUE(move1.HasCrisisBudget());
  // Large piggybank still caps crisis allowance at 3.0 * base_budget.
  TimeManager::State rich;
  rich.first_timed_move = false;
  rich.nps = 40000.0f;
  rich.nps_is_reliable = true;
  rich.piggybank_ms = 3000.0f;
  TimeManager manager(kConfig, GoParams{.wtime = 8000, .winc = 80}, false,
                      kStart, rich, 0);
  const int64_t base = BudgetMs(manager);
  const int64_t crisis = CrisisBudgetMs(manager);
  EXPECT_GT(crisis, base);
  EXPECT_LE(crisis - base, 3 * base);
  EXPECT_LE(crisis, 8000 - 200 - 30);

  // At base deadline + 10ms: non-crisis stops, crisis keeps running and marks
  // piggybank usage.
  const auto past_base = kStart + milliseconds(base + 10);
  EXPECT_TRUE(manager.ShouldStop(past_base, false));
  EXPECT_FALSE(manager.ShouldStop(past_base, true));
  // Crisis also extends EstimatedRemainingVisits to crisis_deadline.
  manager.SetCrisisActive(true, past_base);
  EXPECT_GT(manager.EstimatedRemainingVisits(1000, past_base), 0u);
  // Exiting crisis after base_deadline stops immediately.
  manager.SetCrisisActive(false, past_base);
  EXPECT_TRUE(manager.ShouldStop(past_base, false));
  EXPECT_EQ(manager.EstimatedRemainingVisits(1000, past_base), 0u);
  // Re-activating stops at crisis_deadline and subtracts spent piggybank.
  EXPECT_TRUE(manager.ShouldStop(kStart + milliseconds(crisis), true));
  const auto finalized = manager.FinalizeState(
      kStart + milliseconds(base + 50), 10000, 10000);
  EXPECT_LT(finalized.piggybank_ms, rich.piggybank_ms);
}

TEST(Lc5TimeManagerTest, FinalizeStateTracksNpsTimeuseAndIgnoresNonClock) {
  const TimeManager::State init;
  // Non-clock search leaves state unchanged.
  const TimeManager non_clock(kConfig, GoParams{.movetime = 500}, false, kStart,
                              init, 0);
  const auto unchanged =
      non_clock.FinalizeState(kStart + milliseconds(200), 10000, 10000);
  EXPECT_FLOAT_EQ(unchanged.nps, init.nps);
  EXPECT_EQ(unchanged.nps_is_reliable, init.nps_is_reliable);
  EXPECT_EQ(unchanged.first_timed_move, init.first_timed_move);

  // First timed move immediately calibrates nps and marks reliable.
  const TimeManager move1(kConfig, GoParams{.wtime = 8000, .winc = 80}, false,
                          kStart, init, 0);
  const auto s1 =
      move1.FinalizeState(kStart + milliseconds(500), 25000, 25000);
  EXPECT_TRUE(s1.nps_is_reliable);
  EXPECT_FALSE(s1.first_timed_move);
  EXPECT_NEAR(s1.nps, 50000.0f, 1.0f);
  EXPECT_EQ(s1.last_move_total_nodes, 25000u);

  // Move 2: higher nps snaps up immediately; lower nps decays slowly over 20s.
  const TimeManager move2(kConfig, GoParams{.wtime = 7580, .winc = 80}, false,
                          kStart, s1, 5000);
  const auto faster =
      move2.FinalizeState(kStart + milliseconds(200), 16000, 21000);
  EXPECT_NEAR(faster.nps, 80000.0f, 1.0f);

  const TimeManager move2_slow(kConfig, GoParams{.wtime = 7580, .winc = 80},
                               false, kStart, s1, 5000);
  const auto slower =
      move2_slow.FinalizeState(kStart + milliseconds(200), 2000, 7000);
  EXPECT_GT(slower.nps, 49000.0f);
  EXPECT_LT(slower.nps, 50000.0f);
  EXPECT_GE(slower.timeuse, 0.34f);
}

}  // namespace
}  // namespace lczero::lc5

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
