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
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "search/lc5/visit.h"

namespace lczero::lc5 {

// Fixed-capacity pool of visits. A VisitId packs the slot index with a
// per-slot epoch, so a stale id never resolves to a reused slot.
//
// A visit has a single owner from Allocate() to Release(): only the thread
// that owns it reads or changes it, so slots need no lock. The pool's mutex
// publishes a released slot to its next owner.
class VisitPool {
 public:
  explicit VisitPool(size_t capacity);
  size_t capacity() const { return slots_.size(); }

  // Reserves the lowest free slot and initializes its visit from `origin`.
  // Returns nullopt when the pool is full.
  std::optional<VisitId> Allocate(const VisitOrigin& origin);

  // Whether `id` is a live visit, not a released or reused one.
  bool IsLive(VisitId id) const {
    if (!id || id.slot() >= slots_.size()) return false;
    const Slot& slot = slots_[id.slot()];
    return slot.epoch == id.epoch() && slot.visit.state != VisitState::kFree;
  }
  // The live visit `id`.
  Visit& Get(VisitId id) {
    assert(IsLive(id));
    return slots_[id.slot()].visit;
  }
  // Resets the live visit `id` and frees its slot.
  void Release(VisitId id);

  // Live visits, including ones allocated but not yet initialized.
  size_t active() const { return active_.load(std::memory_order_relaxed); }

  void SetEpochForTesting(uint32_t slot, uint32_t epoch) {
    slots_[slot].epoch = epoch;
  }

 private:
  // Its own cache lines: neighbouring slots usually have different owners.
  struct alignas(64) Slot {
    uint32_t epoch = 0;
    Visit visit;
  };

  absl::Mutex mutex_;
  std::vector<Slot> slots_;
  // A min-heap: Allocate reuses the lowest free index.
  std::vector<uint32_t> free_indices_ ABSL_GUARDED_BY(mutex_);
  std::atomic<size_t> active_{0};
};

}  // namespace lczero::lc5
