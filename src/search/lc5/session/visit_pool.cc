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

#include <algorithm>
#include <functional>

namespace lczero::lc5 {

VisitPool::VisitPool(size_t capacity) {
  slots_.reserve(capacity);
  free_indices_.reserve(capacity);
  for (size_t i = 0; i < capacity; ++i) {
    slots_.push_back(std::make_unique<Slot>());
    free_indices_.push_back(i);
  }
  // Ascending indices already form a min-heap.
}

std::optional<VisitId> VisitPool::Allocate(const VisitOrigin& origin) {
  uint32_t i;
  {
    absl::MutexLock pool_lock(pool_mutex_);
    if (free_indices_.empty()) return std::nullopt;
    std::pop_heap(free_indices_.begin(), free_indices_.end(),
                  std::greater<uint32_t>{});
    i = free_indices_.back();
    free_indices_.pop_back();
    // Count reservations too, so admission cannot overfill the pool.
    ++active_;
  }
  // Never hold pool_mutex_ while waiting for a slot: release locks
  // slot -> pool.
  Slot& slot = *slots_[i];
  absl::MutexLock slot_lock(slot.mutex);
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
  return id;
}

bool VisitPool::ReleaseLocked(Slot& slot, VisitId id) {
  if (slot.epoch.load(std::memory_order_relaxed) != id.epoch() ||
      slot.visit.id != id || slot.visit.state == VisitState::kFree)
    return false;
  slot.visit = Visit{};
  // Publish only after resetting the visit.
  absl::MutexLock pool_lock(pool_mutex_);
  free_indices_.push_back(id.slot());
  std::push_heap(free_indices_.begin(), free_indices_.end(),
                 std::greater<uint32_t>{});
  active_.fetch_sub(1, std::memory_order_relaxed);
  return true;
}

std::vector<VisitId> VisitPool::ActiveIds() const {
  std::vector<VisitId> ids;
  for (const auto& ptr : slots_) {
    absl::MutexLock lock(ptr->mutex);
    if (ptr->visit.state != VisitState::kFree) ids.push_back(ptr->visit.id);
  }
  return ids;
}

}  // namespace lczero::lc5
