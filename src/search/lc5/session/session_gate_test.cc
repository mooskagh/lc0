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

#include "search/lc5/session/session_gate.h"

#include <gtest/gtest.h>

#include <vector>

namespace lczero::lc5 {
namespace {

using StopMode = SessionGate::StopMode;

SessionGate::Config TestConfig(size_t workers = 2, uint64_t node_limit = 100) {
  return {.workers = workers,
          .node_limit = node_limit,
          .root_visits_at_start = 0,
          .active_visits_ratio = 0.0f};
}

TEST(Lc5SessionGateTest, BestmoveIsCommittedUnlessAbortedFirst) {
  VisitPool pool(1);
  Metrics metrics;
  SessionGate gate(TestConfig(), pool, metrics);
  EXPECT_FALSE(gate.IsStopping());
  EXPECT_TRUE(gate.AuthorizeEval());
  gate.RequestStop(StopMode::kRespondBestmove);
  EXPECT_TRUE(gate.IsStopping());
  EXPECT_FALSE(gate.AuthorizeEval());
  // An abort overrides an uncommitted bestmove; a later stop changes nothing.
  gate.RequestStop(StopMode::kAbort);
  gate.RequestStop(StopMode::kRespondBestmove);
  EXPECT_EQ(gate.stop_mode(), StopMode::kAbort);
  EXPECT_FALSE(gate.TryCommitOutput());

  SessionGate committed(TestConfig(), pool, metrics);
  committed.RequestStop(StopMode::kRespondBestmove);
  EXPECT_TRUE(committed.TryCommitOutput());
  committed.RequestStop(StopMode::kAbort);
  EXPECT_EQ(committed.stop_mode(), StopMode::kRespondBestmove);
}

TEST(Lc5SessionGateTest, AdmissionClosesOnBudgetAndOnStop) {
  VisitPool pool(4);
  Metrics metrics;
  SessionGate gate(TestConfig(1, 2), pool, metrics);
  EXPECT_TRUE(gate.Admit({}));
  std::vector<VisitId> admitted;
  gate.AdmitFor(0, {}, 3, [&](VisitId id) { admitted.push_back(id); });
  EXPECT_EQ(admitted.size(), 1u);
  EXPECT_EQ(metrics.visits_admitted.load(), 2u);
  // The budget is spent, so the worker does not wait for capacity.
  for (const auto id : admitted) pool.Release(id);
  EXPECT_TRUE(gate.TakeWaitersToWake().empty());

  SessionGate stopped(TestConfig(1), pool, metrics);
  stopped.RequestStop(StopMode::kAbort);
  EXPECT_FALSE(stopped.Admit({}));
}

TEST(Lc5SessionGateTest, FullPoolWakesWaitersInTurn) {
  VisitPool pool(2);
  Metrics metrics;
  SessionGate gate(TestConfig(3), pool, metrics);
  std::vector<VisitId> admitted;
  const auto add = [&](VisitId id) { admitted.push_back(id); };
  gate.AdmitFor(0, {}, 3, add);
  ASSERT_EQ(admitted.size(), 2u);
  gate.AdmitFor(1, {}, 1, add);
  gate.AdmitFor(2, {}, 1, add);
  EXPECT_EQ(admitted.size(), 2u);
  EXPECT_TRUE(gate.TakeWaitersToWake().empty());
  // One free slot wakes one waiter, then the next one in turn.
  pool.Release(admitted[0]);
  EXPECT_EQ(gate.TakeWaitersToWake(), std::vector<size_t>({0}));
  EXPECT_EQ(gate.TakeWaitersToWake(), std::vector<size_t>({1}));
  pool.Release(admitted[1]);
  EXPECT_EQ(gate.TakeWaitersToWake(), std::vector<size_t>({2}));
  EXPECT_TRUE(gate.TakeWaitersToWake().empty());
}

}  // namespace
}  // namespace lczero::lc5

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
