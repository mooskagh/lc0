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

#include "search/lc5/session/materialization_table.h"

#include <gtest/gtest.h>

namespace lczero::lc5 {
namespace {

VisitWakeup Wakeup(uint64_t id, size_t worker) {
  return {.id = VisitId{id}, .worker = worker};
}

TEST(Lc5MaterializationTableTest, OwnerOpensAndWaitersJoin) {
  MaterializationTable table;
  EXPECT_TRUE(table.empty());
  const auto attach = [](MaterializationId) { return true; };
  const auto owner = table.Join(NodeKey{1}, Wakeup(10, 0), attach);
  ASSERT_TRUE(owner);
  EXPECT_TRUE(owner->owner);
  const auto waiter = table.Join(NodeKey{1}, Wakeup(11, 1), attach);
  ASSERT_TRUE(waiter);
  EXPECT_FALSE(waiter->owner);
  EXPECT_EQ(waiter->id, owner->id);
  EXPECT_EQ(waiter->waiters, 1u);
  const auto other = table.Join(NodeKey{2}, Wakeup(12, 0), attach);
  ASSERT_TRUE(other);
  EXPECT_TRUE(other->owner);
  EXPECT_NE(other->id, owner->id);

  const auto waiters = table.Close(owner->id, [](NodeKey key) {
    EXPECT_EQ(key, NodeKey{1});
    return true;
  });
  ASSERT_TRUE(waiters);
  EXPECT_EQ(waiters->key, NodeKey{1});
  EXPECT_EQ(waiters->owner.id, VisitId{10});
  ASSERT_EQ(waiters->waiters.size(), 1u);
  EXPECT_EQ(waiters->waiters[0].id, VisitId{11});
  EXPECT_EQ(waiters->waiters[0].worker, 1u);
  EXPECT_FALSE(table.empty());
  // A closed node's next visit opens a new materialization.
  const auto reopened = table.Join(NodeKey{1}, Wakeup(13, 0), attach);
  ASSERT_TRUE(reopened);
  EXPECT_TRUE(reopened->owner);
  EXPECT_NE(reopened->id, owner->id);
  for (const auto id : {other->id, reopened->id}) {
    EXPECT_TRUE(table.Close(id, [](NodeKey) { return true; }));
  }
  EXPECT_TRUE(table.empty());
}

TEST(Lc5MaterializationTableTest, AttachCanDecline) {
  MaterializationTable table;
  MaterializationId offered = 0;
  EXPECT_FALSE(table.Join(NodeKey{1}, Wakeup(10, 0),
                          [&](MaterializationId materialization_id) {
                            offered = materialization_id;
                            return false;
                          }));
  EXPECT_NE(offered, 0u);
  EXPECT_TRUE(table.empty());
  const auto owner = table.Join(NodeKey{1}, Wakeup(11, 0),
                                [](MaterializationId) { return true; });
  ASSERT_TRUE(owner);
  EXPECT_TRUE(owner->owner);
  EXPECT_FALSE(table.Join(NodeKey{1}, Wakeup(12, 0),
                          [&](MaterializationId materialization_id) {
                            EXPECT_EQ(materialization_id, owner->id);
                            return false;
                          }));
  const auto waiters = table.Close(owner->id, [](NodeKey) { return true; });
  ASSERT_TRUE(waiters);
  EXPECT_TRUE(waiters->waiters.empty());
}

TEST(Lc5MaterializationTableTest, DeclinedCloseKeepsMaterializationOpen) {
  MaterializationTable table;
  const auto attach = [](MaterializationId) { return true; };
  const auto owner = table.Join(NodeKey{1}, Wakeup(10, 0), attach);
  ASSERT_TRUE(owner);
  EXPECT_FALSE(table.Close(owner->id, [](NodeKey) { return false; }));
  const auto waiter = table.Join(NodeKey{1}, Wakeup(11, 0), attach);
  ASSERT_TRUE(waiter);
  EXPECT_FALSE(waiter->owner);
  EXPECT_EQ(waiter->id, owner->id);
  const auto waiters = table.Close(owner->id, [](NodeKey) { return true; });
  ASSERT_TRUE(waiters);
  EXPECT_EQ(waiters->waiters.size(), 1u);
  EXPECT_TRUE(table.empty());
}

}  // namespace
}  // namespace lczero::lc5

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
