#include "search/lc5/time_manager.h"

#include <gtest/gtest.h>

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
  const auto before = manager.Evaluate(deadline - nanoseconds(1));
  EXPECT_FALSE(before.should_stop);
  EXPECT_EQ(before.next_check, deadline);
  for (const auto now : {deadline, deadline + nanoseconds(1),
                         deadline + std::chrono::hours(1)}) {
    const auto decision = manager.Evaluate(now);
    EXPECT_TRUE(decision.should_stop);
    EXPECT_EQ(decision.next_check, std::nullopt);
  }
  EXPECT_EQ(manager.Evaluate(kStart).should_stop, budget_ms <= 0);
}

void ExpectNoDeadline(const GoParams& params, bool black_to_move) {
  const TimeManager manager(kConfig, params, black_to_move, kStart);
  for (const auto now :
       {kStart - nanoseconds(1), kStart, kStart + std::chrono::hours(24)}) {
    const auto decision = manager.Evaluate(now);
    EXPECT_FALSE(decision.should_stop);
    EXPECT_EQ(decision.next_check, std::nullopt);
  }
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
    const auto decision = manager.Evaluate(kStart);
    ASSERT_FALSE(decision.should_stop);
    ASSERT_TRUE(decision.next_check.has_value());
    const auto budget =
        std::chrono::duration_cast<milliseconds>(*decision.next_check - kStart)
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

TEST(Lc5TimeManagerTest, ExplicitMovetimeIsLiteralAndWinsOverClocks) {
  for (bool black_to_move : {false, true}) {
    for (int64_t movetime : {500, 1, 0, -1, -500}) {
      SCOPED_TRACE(black_to_move);
      SCOPED_TRACE(movetime);
      ExpectDeadline(GoParams{.movetime = movetime}, black_to_move, movetime);
      ExpectDeadline(GoParams{.wtime = 0,
                              .btime = 0,
                              .winc = 10000,
                              .binc = 10000,
                              .movestogo = 1,
                              .movetime = movetime},
                     black_to_move, movetime);
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
        ExpectDeadline(params, black_to_move, movetime);
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
    ExpectDeadline(GoParams{.nodes = nodes, .movetime = 500}, false, 500);
    ExpectNoDeadline(GoParams{.wtime = 0, .nodes = nodes, .infinite = true},
                     false);
    ExpectDeadline(
        GoParams{.wtime = 0, .nodes = nodes, .movetime = 500, .infinite = true},
        false, 500);
    ExpectNoDeadline(
        GoParams{.wtime = 0, .nodes = nodes, .movetime = 0, .ponder = true},
        false);
  }
}

TEST(Lc5TimeManagerTest, SuppliedStartTimeChargesPreparation) {
  const TimeManager manager(kConfig, GoParams{.movetime = 500}, false, kStart);
  const auto preparing = manager.Evaluate(kStart + milliseconds(499));
  EXPECT_FALSE(preparing.should_stop);
  EXPECT_EQ(preparing.next_check, kStart + milliseconds(500));
  const auto initialized = manager.Evaluate(kStart + milliseconds(600));
  EXPECT_TRUE(initialized.should_stop);
  EXPECT_EQ(initialized.next_check, std::nullopt);
}

}  // namespace
}  // namespace lczero::lc5

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
