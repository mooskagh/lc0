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

#include "search/lc5/session/visit_pool.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <limits>
#include <thread>
#include <vector>

namespace lczero::lc5 {
namespace {

TEST(Lc5VisitPoolTest, ExhaustionAndOrderedReuse) {
  VisitPool empty(0);
  EXPECT_FALSE(empty.Allocate({}));
  EXPECT_EQ(empty.active(), 0u);

  VisitPool pool(4);
  std::array<VisitId, 4> ids;
  for (uint32_t i = 0; i < ids.size(); ++i) {
    auto id = pool.Allocate({});
    ASSERT_TRUE(id);
    ids[i] = *id;
    EXPECT_EQ(id->slot(), i);
    EXPECT_EQ(id->epoch(), 1u);
    EXPECT_TRUE(pool.IsLive(*id));
  }
  EXPECT_EQ(pool.capacity(), 4u);
  EXPECT_EQ(pool.active(), 4u);
  EXPECT_FALSE(pool.Allocate({}));

  for (uint32_t i : {3u, 1u, 2u}) pool.Release(ids[i]);
  EXPECT_EQ(pool.active(), 1u);
  for (uint32_t i = 1; i < ids.size(); ++i) {
    auto id = pool.Allocate({});
    ASSERT_TRUE(id);
    EXPECT_EQ(id->slot(), i);
    EXPECT_EQ(id->epoch(), 2u);
    ids[i] = *id;
  }
  EXPECT_FALSE(pool.Allocate({}));
  for (VisitId id : ids) pool.Release(id);
  EXPECT_EQ(pool.active(), 0u);
  for (VisitId id : ids) EXPECT_FALSE(pool.IsLive(id));
}

TEST(Lc5VisitPoolTest, StaleAndForeignIdsAreNotLive) {
  VisitPool pool(2);
  auto old = pool.Allocate({});
  auto other = pool.Allocate({});
  ASSERT_TRUE(old);
  ASSERT_TRUE(other);
  EXPECT_FALSE(pool.IsLive({}));
  EXPECT_FALSE(pool.IsLive(VisitId{(uint64_t{1} << 32) | 2}));
  pool.Release(*old);
  EXPECT_FALSE(pool.IsLive(*old));
  EXPECT_TRUE(pool.IsLive(*other));
  EXPECT_EQ(pool.active(), 1u);
  auto current = pool.Allocate({});
  ASSERT_TRUE(current);
  EXPECT_EQ(current->slot(), old->slot());
  EXPECT_NE(current->epoch(), old->epoch());
  EXPECT_FALSE(pool.IsLive(*old));
  EXPECT_TRUE(pool.IsLive(*current));
  EXPECT_EQ(pool.Get(*current).id, *current);
  pool.Release(*current);
  pool.Release(*other);
  EXPECT_EQ(pool.active(), 0u);
}

TEST(Lc5VisitPoolTest, InitializesAndResetsVisit) {
  VisitOrigin origin{
      .key = NodeKey{42},
      .history = {},
      .backup_prefix = {
          {.key = NodeKey{7}, .generation = 9, .selected_move = std::nullopt}}};
  origin.history.Reset(ChessBoard::kStartposBoard, 0, 0);
  VisitPool pool(1);
  for (uint32_t epoch = 1; epoch <= 2; ++epoch) {
    auto id = pool.Allocate(origin);
    ASSERT_TRUE(id);
    EXPECT_EQ(id->epoch(), epoch);
    auto& visit = pool.Get(*id);
    EXPECT_EQ(visit.id, *id);
    EXPECT_EQ(visit.state, VisitState::kReadySelect);
    EXPECT_EQ(visit.origin.key, origin.key);
    EXPECT_EQ(visit.origin.history.Last(), origin.history.Last());
    ASSERT_EQ(visit.origin.backup_prefix.size(), 1u);
    EXPECT_EQ(visit.origin.backup_prefix[0].key, NodeKey{7});
    EXPECT_EQ(visit.history.GetLength(), 1);
    EXPECT_EQ(visit.history.Last(), origin.history.Last());
    EXPECT_EQ(visit.current_key, origin.key);
    ASSERT_EQ(visit.path.size(), 1u);
    EXPECT_EQ(visit.path[0].key, NodeKey{7});
    EXPECT_EQ(visit.path[0].generation, 9u);
    EXPECT_FALSE(visit.path[0].selected_move);
    EXPECT_EQ(visit.waiting_materialization_id, 0u);
    EXPECT_FLOAT_EQ(visit.result.q, 0.0f);
    EXPECT_FLOAT_EQ(visit.result.d, 0.0f);
    EXPECT_FLOAT_EQ(visit.result.m, 0.0f);
    EXPECT_FALSE(visit.two_fold_origin);
    visit.waiting_materialization_id = 17;
    visit.result = {1.0f, 0.5f, 3.0f};
    visit.two_fold_origin = 3;
    pool.Release(*id);
    // The slot itself is reset, not only marked free.
    EXPECT_FALSE(visit.id);
    EXPECT_EQ(visit.state, VisitState::kFree);
    EXPECT_EQ(visit.origin.history.GetLength(), 0);
    EXPECT_TRUE(visit.origin.backup_prefix.empty());
    EXPECT_EQ(visit.history.GetLength(), 0);
    EXPECT_TRUE(visit.path.empty());
    EXPECT_EQ(visit.waiting_materialization_id, 0u);
    EXPECT_EQ(pool.active(), 0u);
  }
}

TEST(Lc5VisitPoolTest, EpochWrapSkipsZero) {
  VisitPool pool(1);
  auto id = pool.Allocate({});
  ASSERT_TRUE(id);
  pool.Release(*id);
  pool.SetEpochForTesting(0, std::numeric_limits<uint32_t>::max());
  auto wrapped = pool.Allocate({});
  ASSERT_TRUE(wrapped);
  EXPECT_EQ(wrapped->epoch(), 1u);
  EXPECT_TRUE(static_cast<bool>(*wrapped));
  pool.Release(*wrapped);
}

TEST(Lc5VisitPoolTest, ConcurrentAllocationAndRelease) {
  VisitPool pool(3);
  std::array<std::atomic<bool>, 3> owned{};
  std::vector<std::thread> workers;
  for (int worker = 0; worker < 4; ++worker) {
    workers.emplace_back([&] {
      for (int iteration = 0; iteration < 2000; ++iteration) {
        std::optional<VisitId> id;
        while (!(id = pool.Allocate({}))) std::this_thread::yield();
        EXPECT_FALSE(owned[id->slot()].exchange(true));
        const Visit& visit = pool.Get(*id);
        EXPECT_EQ(visit.id, *id);
        EXPECT_EQ(visit.state, VisitState::kReadySelect);
        EXPECT_LE(pool.active(), pool.capacity());
        EXPECT_TRUE(owned[id->slot()].exchange(false));
        pool.Release(*id);
      }
    });
  }
  for (auto& worker : workers) worker.join();
  EXPECT_EQ(pool.active(), 0u);
  for (uint32_t i = 0; i < pool.capacity(); ++i) {
    auto id = pool.Allocate({});
    ASSERT_TRUE(id);
    EXPECT_EQ(id->slot(), i);
  }
  EXPECT_FALSE(pool.Allocate({}));
}

}  // namespace
}  // namespace lczero::lc5

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
