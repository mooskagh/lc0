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

#pragma once

#include <absl/synchronization/mutex.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "search/lc5/visit.h"

namespace lczero::lc5 {

// Fixed-capacity pool of visit slots. A VisitId packs the slot index with a
// per-slot epoch, so a stale id never resolves to a reused slot.
class VisitPool {
 public:
  struct Slot {
    absl::Mutex mutex;
    std::atomic<uint32_t> epoch{0};
    Visit visit ABSL_GUARDED_BY(mutex);
  };

  explicit VisitPool(size_t capacity);
  size_t capacity() const { return slots_.size(); }

  // Reserves the lowest free slot and initializes its visit from `origin`.
  // Returns nullopt when the pool is full.
  std::optional<VisitId> Allocate(const VisitOrigin& origin,
                                  size_t worker = 0);

  Slot* Lookup(VisitId id) {
    if (!id || id.slot() >= slots_.size()) return nullptr;
    Slot* slot = slots_[id.slot()].get();
    return slot->epoch.load(std::memory_order_acquire) == id.epoch() ? slot
                                                                     : nullptr;
  }

  // Whether `slot`, found by Lookup(id), still holds live visit `id`: it may
  // have finished or been reused before its mutex was taken.
  static bool IsLiveLocked(const Slot& slot, VisitId id)
      ABSL_SHARED_LOCKS_REQUIRED(slot.mutex) {
    return slot.epoch.load(std::memory_order_relaxed) == id.epoch() &&
           slot.visit.state != VisitState::kFree;
  }

  bool ReleaseLocked(Slot& slot, VisitId id)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(slot.mutex);

  size_t active() const { return active_.load(std::memory_order_relaxed); }
  std::vector<VisitId> ActiveIds() const;

 private:
  absl::Mutex pool_mutex_;
  std::vector<std::unique_ptr<Slot>> slots_;
  // Every index is either free or reserved/active.
  std::vector<uint32_t> free_indices_ ABSL_GUARDED_BY(pool_mutex_);
  std::atomic<size_t> active_{0};
};

}  // namespace lczero::lc5
