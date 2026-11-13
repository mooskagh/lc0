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

#include "search/lc5/session/visit_limiter.h"

#include <gtest/gtest.h>

#include <vector>

namespace lczero::lc5 {
namespace {

VisitLimiter::Config TestConfig(size_t workers = 2, uint64_t node_limit = 100) {
  return {.workers = workers,
          .node_limit = node_limit,
          .root_visits_at_start = 0,
          .active_visits_ratio = 0.0f};
}

TEST(Lc5VisitLimiterTest, StopSetsStoppingFlag) {
  VisitPool pool(1);
  Metrics metrics;
  VisitLimiter limiter(TestConfig(), pool, metrics);
  EXPECT_FALSE(limiter.IsStopping());
  limiter.Stop();
  EXPECT_TRUE(limiter.IsStopping());
}

TEST(Lc5VisitLimiterTest, AdmissionClosesOnBudgetAndOnStop) {
  VisitPool pool(4);
  Metrics metrics;
  VisitLimiter limiter(TestConfig(1, 2), pool, metrics);
  EXPECT_EQ(limiter.AdmitVisits(0, {}, 1).size(), 1u);
  const auto admitted = limiter.AdmitVisits(0, {}, 3);
  EXPECT_EQ(admitted.size(), 1u);
  EXPECT_EQ(metrics.visits_admitted.load(), 2u);
  // The budget is spent, so the worker does not wait for capacity.
  for (const auto id : admitted) pool.Release(id);
  EXPECT_TRUE(limiter.TakeWorkersToWake().empty());

  VisitLimiter stopped(TestConfig(1), pool, metrics);
  stopped.Stop();
  EXPECT_TRUE(stopped.AdmitVisits(0, {}, 1).empty());
}

TEST(Lc5VisitLimiterTest, FullPoolWakesWaitingWorkersInTurn) {
  VisitPool pool(2);
  Metrics metrics;
  VisitLimiter limiter(TestConfig(3), pool, metrics);
  const auto admitted = limiter.AdmitVisits(0, {}, 3);
  ASSERT_EQ(admitted.size(), 2u);
  EXPECT_TRUE(limiter.AdmitVisits(1, {}, 1).empty());
  EXPECT_TRUE(limiter.AdmitVisits(2, {}, 1).empty());
  EXPECT_TRUE(limiter.TakeWorkersToWake().empty());
  // One free slot wakes one waiter, then the next one in turn.
  pool.Release(admitted[0]);
  EXPECT_EQ(limiter.TakeWorkersToWake(), std::vector<size_t>({0}));
  EXPECT_EQ(limiter.TakeWorkersToWake(), std::vector<size_t>({1}));
  pool.Release(admitted[1]);
  EXPECT_EQ(limiter.TakeWorkersToWake(), std::vector<size_t>({2}));
  EXPECT_TRUE(limiter.TakeWorkersToWake().empty());
}

TEST(Lc5VisitLimiterTest, ActiveLimitHonorsMinibatchFloorAndScalesWithTree) {
  VisitPool pool(256);
  Metrics metrics;
  VisitLimiter limiter({.workers = 1,
                        .node_limit = 500,
                        .root_visits_at_start = 0,
                        .active_visits_ratio = 0.25f,
                        .min_active_visits = 48},
                       pool, metrics);
  EXPECT_EQ(limiter.GetActiveVisitLimit(), 48u);
  const auto admitted = limiter.AdmitVisits(0, {}, 60);
  EXPECT_EQ(admitted.size(), 48u);
  EXPECT_TRUE(limiter.AdmitVisits(0, {}, 1).empty());
  for (const auto id : admitted) pool.Release(id);

  metrics.visits_completed.store(100);
  EXPECT_EQ(limiter.GetActiveVisitLimit(), 48u);

  metrics.visits_completed.store(200);
  EXPECT_EQ(limiter.GetActiveVisitLimit(), 66u);
}

TEST(Lc5VisitLimiterTest, ActiveLimitRespectsPoolCapacityAndDefaultFloor) {
  VisitPool small_pool(32);
  Metrics metrics;
  VisitLimiter small_limiter({.workers = 1,
                              .node_limit = 500,
                              .root_visits_at_start = 0,
                              .active_visits_ratio = 0.25f,
                              .min_active_visits = 48},
                             small_pool, metrics);
  EXPECT_EQ(small_limiter.GetActiveVisitLimit(), 32u);

  VisitPool large_pool(256);
  VisitLimiter default_limiter({.workers = 1,
                                .node_limit = 500,
                                .root_visits_at_start = 0,
                                .active_visits_ratio = 0.25f,
                                .min_active_visits = 4},
                               large_pool, metrics);
  EXPECT_EQ(default_limiter.GetActiveVisitLimit(),
            VisitLimiter::kMinActiveVisits);
}

}  // namespace
}  // namespace lczero::lc5

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
