#pragma once

#include <cstdint>
#include <atomic>
#include <compare>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "chess/position.h"
#include "search/lc5/graph.h"

namespace lczero::lc5 {

struct VisitId {
  uint64_t value = 0;
  uint32_t slot() const { return static_cast<uint32_t>(value); }
  uint32_t epoch() const { return static_cast<uint32_t>(value >> 32); }
  explicit operator bool() const { return value != 0; }
  auto operator<=>(const VisitId&) const = default;
};

struct PathStep {
  NodeKey key;
  uint64_t generation = 0;
  std::optional<Move> selected_move;
};

struct VisitOrigin {
  NodeKey key;
  PositionHistory history;
  std::vector<PathStep> backup_prefix;
};

enum class VisitState : uint8_t {
  kFree,
  kReadySelect,
  kWaitingMaterialization,
  kReadyBackup,
  kBackingUp,
  kCancelling,
};

struct Visit {
  VisitId id;
  VisitState state = VisitState::kFree;
  VisitOrigin origin;
  PositionHistory history;
  NodeKey current_key;
  std::vector<PathStep> path;
  MaterializationTicketId waiting_ticket = 0;
  SearchValue result;
};

class VisitPool {
 public:
  struct Slot {
    std::mutex mutex;
    std::atomic<uint32_t> epoch{0};
    Visit visit;
  };

  explicit VisitPool(size_t capacity) {
    slots_.reserve(capacity);
    for (size_t i = 0; i < capacity; ++i) slots_.push_back(std::make_unique<Slot>());
  }
  size_t capacity() const { return slots_.size(); }

  std::optional<VisitId> Allocate(const VisitOrigin& origin) {
    std::lock_guard pool_lock(pool_mutex_);
    for (uint32_t i = 0; i < slots_.size(); ++i) {
      Slot& slot = *slots_[i];
      std::lock_guard slot_lock(slot.mutex);
      if (slot.visit.state != VisitState::kFree) continue;
      uint32_t epoch = slot.epoch.load(std::memory_order_relaxed) + 1;
      if (epoch == 0) ++epoch;
      slot.epoch.store(epoch, std::memory_order_release);
      VisitId id{(static_cast<uint64_t>(epoch) << 32) | i};
      slot.visit = Visit{.id = id,
                         .state = VisitState::kReadySelect,
                         .origin = origin,
                         .history = origin.history,
                         .current_key = origin.key,
                         .path = origin.backup_prefix,
                         .result = {}};
      ++active_;
      return id;
    }
    return std::nullopt;
  }

  Slot* Lookup(VisitId id) {
    if (!id || id.slot() >= slots_.size()) return nullptr;
    Slot* slot = slots_[id.slot()].get();
    return slot->epoch.load(std::memory_order_acquire) == id.epoch() ? slot
                                                                    : nullptr;
  }

  bool ReleaseLocked(Slot& slot, VisitId id) {
    if (slot.epoch.load(std::memory_order_relaxed) != id.epoch() ||
        slot.visit.state == VisitState::kFree)
      return false;
    slot.visit = Visit{};
    active_.fetch_sub(1, std::memory_order_relaxed);
    return true;
  }

  size_t active() const { return active_.load(std::memory_order_relaxed); }
  std::vector<VisitId> ActiveIds() const {
    std::vector<VisitId> ids;
    for (const auto& ptr : slots_) {
      std::lock_guard lock(ptr->mutex);
      if (ptr->visit.state != VisitState::kFree) ids.push_back(ptr->visit.id);
    }
    return ids;
  }

 private:
  mutable std::mutex pool_mutex_;
  std::vector<std::unique_ptr<Slot>> slots_;
  std::atomic<size_t> active_{0};
};

}  // namespace lczero::lc5
