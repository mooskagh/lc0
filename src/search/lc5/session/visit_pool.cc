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

VisitPool::VisitPool(size_t capacity) : slots_(capacity) {
  free_indices_.reserve(capacity);
  // Ascending indices already form a min-heap.
  for (size_t i = 0; i < capacity; ++i) free_indices_.push_back(i);
}

std::optional<VisitId> VisitPool::Allocate(const VisitOrigin& origin) {
  uint32_t i;
  {
    absl::MutexLock lock(mutex_);
    if (free_indices_.empty()) return std::nullopt;
    std::pop_heap(free_indices_.begin(), free_indices_.end(),
                  std::greater<uint32_t>{});
    i = free_indices_.back();
    free_indices_.pop_back();
    // Count reservations too, so admission cannot overfill the pool.
    ++active_;
  }
  // The slot is ours now; initialize it outside the lock.
  Slot& slot = slots_[i];
  if (++slot.epoch == 0) ++slot.epoch;
  const VisitId id{(static_cast<uint64_t>(slot.epoch) << 32) | i};
  slot.visit = Visit{.id = id,
                     .state = VisitState::kReadyDescend,
                     .origin = origin,
                     .history = origin.history,
                     .current_key = origin.key,
                     .path = origin.backup_prefix,
                     .result = {}};
  return id;
}

void VisitPool::Release(VisitId id) {
  assert(IsLive(id));
  slots_[id.slot()].visit = Visit{};
  // Publish only after resetting the visit.
  absl::MutexLock lock(mutex_);
  free_indices_.push_back(id.slot());
  std::push_heap(free_indices_.begin(), free_indices_.end(),
                 std::greater<uint32_t>{});
  active_.fetch_sub(1, std::memory_order_relaxed);
}

}  // namespace lczero::lc5
