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

#include "chess/uciloop.h"

namespace lczero::lc5 {
namespace {

using Clock = std::chrono::steady_clock;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;

constexpr TimeManager::Config kConfig{.move_overhead_ms = 200,
                                      .alphazero_time_pct = 3.0f};
const Clock::time_point kStart{std::chrono::seconds(10)};

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
  GoParams params{.wtime = 10200, .btime = 5200};
  ExpectDeadline(params, false, 300);
  ExpectDeadline(params, true, 150);
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
  GoParams params{.wtime = 10200, .btime = 5200, .winc = 100, .binc = 200};
  ExpectDeadline(params, false, 397);
  ExpectDeadline(params, true, 344);
  params.winc = -100;
  ExpectDeadline(params, false, 300);
  params.binc = -200;
  ExpectDeadline(params, true, 150);
}

TEST(Lc5TimeManagerTest, HonorsShorterMovesToGoHorizon) {
  GoParams params{.wtime = 4200, .winc = 100, .movestogo = 4};
  ExpectDeadline(params, false, 1075);
  params.movestogo = 1;
  ExpectDeadline(params, false, 4000);
  for (int moves : {0, -1, 100}) {
    SCOPED_TRACE(moves);
    params.movestogo = moves;
    ExpectDeadline(params, false, 217);
  }
}

TEST(Lc5TimeManagerTest, CapsBudgetAndPreservesOverhead) {
  ExpectDeadline(GoParams{.wtime = 300, .winc = 10000}, false, 100);
  ExpectDeadline(GoParams{.wtime = 201, .winc = 80}, false, 1);
  ExpectDeadline(GoParams{.wtime = 201}, false, 1);
  for (int64_t remaining : {200, 100, 0, -100}) {
    SCOPED_TRACE(remaining);
    ExpectDeadline(GoParams{.wtime = remaining, .winc = 80}, false, 0);
  }
  constexpr TimeManager::Config config{.move_overhead_ms = 200,
                                       .alphazero_time_pct = 0.0f};
  ExpectDeadline(GoParams{.wtime = 1000}, false, 1, config);
  ExpectDeadline(GoParams{.wtime = 1000, .winc = 80}, false, 80, config);
}

TEST(Lc5TimeManagerTest, PreservesPrecisionAndTruncation) {
  ExpectDeadline(GoParams{.wtime = 10201}, false, 300);
  ExpectDeadline(GoParams{.wtime = 4201, .winc = 101, .movestogo = 4}, false,
                 1076);
  // 3.1f must be promoted directly to long double, not rounded to 3.1.
  ExpectDeadline(GoParams{.wtime = 1000200}, false, 30999,
                 {.move_overhead_ms = 200, .alphazero_time_pct = 3.1f});
}

TEST(Lc5TimeManagerTest, IncrementMaintainsBudgetOverLongGame) {
  GoParams params{.wtime = 8000, .winc = 80};
  for (int move = 0; move < 200; ++move) {
    SCOPED_TRACE(move);
    const TimeManager manager(kConfig, params, false, kStart);
    ASSERT_FALSE(manager.ShouldStop(kStart));
    ASSERT_TRUE(manager.deadline().has_value());
    const auto budget =
        std::chrono::duration_cast<milliseconds>(*manager.deadline() - kStart)
            .count();
    EXPECT_GE(budget, 80);
    EXPECT_LE(budget, *params.wtime - 200);
    *params.wtime += *params.winc - budget;
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
  for (int nodes : {-1, 0, 1, 1000}) {
    SCOPED_TRACE(nodes);
    ExpectNoDeadline(GoParams{.nodes = nodes}, false);
    ExpectDeadline(GoParams{.wtime = 10200, .nodes = nodes}, false, 300);
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

TEST(Lc5TimeManagerTest, SavedTimeExtendsOnlyClockBudgets) {
  // 3% of 10000 + 400 saved.
  ExpectDeadline(GoParams{.wtime = 10200}, false, 300);
  const TimeManager manager(kConfig, GoParams{.wtime = 10200}, false, kStart,
                            400);
  EXPECT_EQ(manager.deadline(), kStart + milliseconds(700));
  const TimeManager movetime(kConfig, GoParams{.movetime = 700}, false, kStart,
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
  // Budget 300 + 400 saved.
  const TimeManager manager(kConfig, GoParams{.wtime = 10200}, false, kStart,
                            400);
  EXPECT_EQ(manager.UnusedMs(kStart), 700);
  EXPECT_EQ(manager.UnusedMs(kStart + milliseconds(650)), 50);
  EXPECT_EQ(manager.UnusedMs(kStart + milliseconds(700)), 0);
  EXPECT_EQ(manager.UnusedMs(kStart + milliseconds(800)), 0);
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

TEST(Lc5TimeManagerTest, AffordableVisitsFollowRateAndFactor) {
  TimeManager::Config config = kConfig;
  config.smart_pruning_factor = 2.0f;
  // Budget 300 ms; at 100 ms, 1000 visits leave 2000 more, 1000 after the
  // factor.
  const TimeManager manager(config, GoParams{.wtime = 10200}, false, kStart);
  const auto now = kStart + milliseconds(100);
  EXPECT_TRUE(manager.CanAffordVisits(1000, 1000, now));
  EXPECT_FALSE(manager.CanAffordVisits(1001, 1000, now));
  // The rate isn't trusted before 100 visits, nor with no time elapsed.
  EXPECT_TRUE(manager.CanAffordVisits(1'000'000, 99, now));
  EXPECT_FALSE(
      manager.CanAffordVisits(1'000'000, 100, kStart + milliseconds(1)));
  EXPECT_TRUE(manager.CanAffordVisits(1'000'000, 1000, kStart));
}

}  // namespace
}  // namespace lczero::lc5

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
