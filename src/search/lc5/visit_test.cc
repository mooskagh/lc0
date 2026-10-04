#include "search/lc5/visit.h"

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <future>
#include <limits>
#include <thread>

namespace lczero::lc5 {
namespace {

bool Release(VisitPool& pool, VisitId id) {
  auto* slot = pool.Lookup(id);
  if (!slot) return false;
  std::lock_guard lock(slot->mutex);
  return pool.ReleaseLocked(*slot, id);
}

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
  }
  EXPECT_EQ(pool.capacity(), 4u);
  EXPECT_EQ(pool.active(), 4u);
  EXPECT_FALSE(pool.Allocate({}));
  EXPECT_EQ(pool.ActiveIds(), (std::vector<VisitId>(ids.begin(), ids.end())));

  for (uint32_t i : {3u, 1u, 2u}) EXPECT_TRUE(Release(pool, ids[i]));
  EXPECT_EQ(pool.active(), 1u);
  for (uint32_t i = 1; i < ids.size(); ++i) {
    auto id = pool.Allocate({});
    ASSERT_TRUE(id);
    EXPECT_EQ(id->slot(), i);
    EXPECT_EQ(id->epoch(), 2u);
    ids[i] = *id;
  }
  EXPECT_FALSE(pool.Allocate({}));
  for (VisitId id : ids) EXPECT_TRUE(Release(pool, id));
  EXPECT_EQ(pool.active(), 0u);
  EXPECT_TRUE(pool.ActiveIds().empty());
}

TEST(Lc5VisitPoolTest, RejectsStaleDuplicateAndMismatchedIds) {
  VisitPool pool(2);
  auto old = pool.Allocate({});
  auto other = pool.Allocate({});
  ASSERT_TRUE(old);
  ASSERT_TRUE(other);
  auto* slot = pool.Lookup(*old);
  ASSERT_NE(slot, nullptr);
  {
    std::lock_guard lock(slot->mutex);
    EXPECT_FALSE(pool.ReleaseLocked(*slot, {}));
    EXPECT_FALSE(pool.ReleaseLocked(*slot, *other));
    EXPECT_EQ(pool.active(), 2u);
    EXPECT_TRUE(pool.ReleaseLocked(*slot, *old));
    EXPECT_FALSE(pool.ReleaseLocked(*slot, *old));
  }
  EXPECT_EQ(pool.active(), 1u);
  auto current = pool.Allocate({});
  ASSERT_TRUE(current);
  EXPECT_EQ(current->slot(), old->slot());
  EXPECT_NE(current->epoch(), old->epoch());
  EXPECT_EQ(pool.Lookup(*old), nullptr);
  EXPECT_EQ(pool.Lookup({}), nullptr);
  EXPECT_EQ(pool.Lookup(VisitId{(uint64_t{1} << 32) | 2}), nullptr);
  {
    std::lock_guard lock(slot->mutex);
    EXPECT_FALSE(pool.ReleaseLocked(*slot, *old));
    EXPECT_EQ(slot->visit.id, *current);
  }
  EXPECT_EQ(pool.active(), 2u);
  EXPECT_FALSE(pool.Allocate({}));
  EXPECT_TRUE(Release(pool, *current));
  EXPECT_TRUE(Release(pool, *other));
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
    auto* slot = pool.Lookup(*id);
    ASSERT_NE(slot, nullptr);
    std::lock_guard lock(slot->mutex);
    auto& visit = slot->visit;
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
    EXPECT_EQ(visit.waiting_ticket, 0u);
    EXPECT_FLOAT_EQ(visit.result.q, 0.0f);
    EXPECT_FLOAT_EQ(visit.result.d, 0.0f);
    EXPECT_FLOAT_EQ(visit.result.m, 0.0f);
    visit.waiting_ticket = 17;
    visit.result = {1.0f, 0.5f, 3.0f};
    EXPECT_TRUE(pool.ReleaseLocked(*slot, *id));
    EXPECT_FALSE(visit.id);
    EXPECT_EQ(visit.state, VisitState::kFree);
    EXPECT_EQ(visit.origin.history.GetLength(), 0);
    EXPECT_TRUE(visit.origin.backup_prefix.empty());
    EXPECT_EQ(visit.history.GetLength(), 0);
    EXPECT_TRUE(visit.path.empty());
    EXPECT_EQ(visit.waiting_ticket, 0u);
    EXPECT_EQ(pool.active(), 0u);
  }
}

TEST(Lc5VisitPoolTest, EpochWrapSkipsZero) {
  VisitPool pool(1);
  auto id = pool.Allocate({});
  ASSERT_TRUE(id);
  auto* slot = pool.Lookup(*id);
  ASSERT_NE(slot, nullptr);
  {
    std::lock_guard lock(slot->mutex);
    EXPECT_TRUE(pool.ReleaseLocked(*slot, *id));
    slot->epoch.store(std::numeric_limits<uint32_t>::max(),
                      std::memory_order_relaxed);
  }
  auto wrapped = pool.Allocate({});
  ASSERT_TRUE(wrapped);
  EXPECT_EQ(wrapped->epoch(), 1u);
  EXPECT_TRUE(static_cast<bool>(*wrapped));
  EXPECT_TRUE(Release(pool, *wrapped));
}

TEST(Lc5VisitPoolTest, AllocationDoesNotLockOccupiedSlots) {
  VisitPool pool(2);
  auto occupied = pool.Allocate({});
  ASSERT_TRUE(occupied);
  auto* slot = pool.Lookup(*occupied);
  ASSERT_NE(slot, nullptr);
  std::unique_lock lock(slot->mutex);
  auto allocation =
      std::async(std::launch::async, [&] { return pool.Allocate({}); });
  EXPECT_EQ(allocation.wait_for(std::chrono::seconds(2)),
            std::future_status::ready);
  // Unlock before joining even on failure, so the old scan fails without
  // hanging.
  lock.unlock();
  auto id = allocation.get();
  ASSERT_TRUE(id);
  EXPECT_EQ(id->slot(), 1u);
  EXPECT_TRUE(Release(pool, *id));
  EXPECT_TRUE(Release(pool, *occupied));
}

TEST(Lc5VisitPoolTest, WaitingForReservedSlotDoesNotHoldPoolLock) {
  VisitPool pool(2);
  auto old = pool.Allocate({});
  ASSERT_TRUE(old);
  auto* slot = pool.Lookup(*old);
  ASSERT_NE(slot, nullptr);
  std::unique_lock lock(slot->mutex);
  ASSERT_TRUE(pool.ReleaseLocked(*slot, *old));
  auto first =
      std::async(std::launch::async, [&] { return pool.Allocate({}); });
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (pool.active() == 0 && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  EXPECT_EQ(pool.active(), 1u);
  EXPECT_EQ(first.wait_for(std::chrono::seconds(0)),
            std::future_status::timeout);
  auto second =
      std::async(std::launch::async, [&] { return pool.Allocate({}); });
  EXPECT_EQ(second.wait_for(std::chrono::seconds(2)),
            std::future_status::ready);
  lock.unlock();
  auto first_id = first.get();
  auto second_id = second.get();
  ASSERT_TRUE(first_id);
  ASSERT_TRUE(second_id);
  EXPECT_EQ(first_id->slot(), 0u);
  EXPECT_EQ(second_id->slot(), 1u);
  EXPECT_TRUE(Release(pool, *first_id));
  EXPECT_TRUE(Release(pool, *second_id));
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
        auto* slot = pool.Lookup(*id);
        ASSERT_NE(slot, nullptr);
        std::lock_guard lock(slot->mutex);
        EXPECT_EQ(slot->visit.id, *id);
        EXPECT_EQ(slot->visit.state, VisitState::kReadySelect);
        EXPECT_LE(pool.active(), pool.capacity());
        EXPECT_TRUE(owned[id->slot()].exchange(false));
        EXPECT_TRUE(pool.ReleaseLocked(*slot, *id));
        EXPECT_FALSE(pool.ReleaseLocked(*slot, *id));
      }
    });
  }
  for (auto& worker : workers) worker.join();
  EXPECT_EQ(pool.active(), 0u);
  EXPECT_TRUE(pool.ActiveIds().empty());
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
