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
#include <absl/time/time.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <deque>
#include <utility>

namespace lczero::lc5 {

// Closable multi-producer multi-consumer FIFO. PopUntil() can also wait on a
// caller predicate over external state; whoever changes that state must call
// NotifyPredicateChanged() afterwards. TryPopIf() and PopUntil() take the front
// item only if `accept(front)` holds, and otherwise leave it in the queue.
template <typename T>
class WorkQueue {
 public:
  // Must not be called after Close().
  void Push(T value) {
    absl::MutexLock lock(mutex_);
    assert(!closed_);
    queue_.push_back(std::move(value));
    size_.store(queue_.size(), std::memory_order_relaxed);
  }
  // Blocks until an item is available; false once closed and empty.
  bool Pop(T* value) {
    absl::MutexLock lock(mutex_);
    mutex_.Await(absl::Condition(this, &WorkQueue::ReadyLocked));
    return TakeFrontLocked(value);
  }
  template <typename Accept>
  bool TryPopIf(T* value, Accept accept) {
    absl::MutexLock lock(mutex_);
    return TakeFrontLocked(value, accept);
  }
  // Waits for an item like Pop, but also gives up at `deadline` or once
  // `give_up()` holds. `give_up` may run on any thread that unlocks the queue,
  // under its mutex, and may read external state. A front item that `accept`
  // rejects ends the wait too, and stays in the queue.
  template <typename GiveUp, typename Accept>
  bool PopUntil(T* value, std::chrono::steady_clock::time_point deadline,
                GiveUp give_up, Accept accept) {
    absl::MutexLock lock(mutex_);
    const auto ready = [&] {
      mutex_.AssertReaderHeld();
      return ReadyLocked() || give_up();
    };
    mutex_.AwaitWithTimeout(
        absl::Condition(&ready),
        absl::FromChrono(deadline - std::chrono::steady_clock::now()));
    return TakeFrontLocked(value, accept);
  }
  // Makes PopUntil() waiters re-check `give_up` after external state changed:
  // the mutex re-evaluates waiters' conditions when it is unlocked.
  void NotifyPredicateChanged() { absl::MutexLock lock(mutex_); }
  void Close() {
    absl::MutexLock lock(mutex_);
    closed_ = true;
  }
  size_t Size() const { return size_.load(std::memory_order_relaxed); }

 private:
  bool ReadyLocked() const ABSL_SHARED_LOCKS_REQUIRED(mutex_) {
    return closed_ || !queue_.empty();
  }
  bool TakeFrontLocked(T* value) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return TakeFrontLocked(value, [](const T&) { return true; });
  }
  template <typename Accept>
  bool TakeFrontLocked(T* value, Accept accept)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    if (queue_.empty() || !accept(queue_.front())) return false;
    *value = std::move(queue_.front());
    queue_.pop_front();
    size_.store(queue_.size(), std::memory_order_relaxed);
    return true;
  }

  absl::Mutex mutex_;
  std::deque<T> queue_ ABSL_GUARDED_BY(mutex_);
  // Lock-free copy of queue_.size(), for metrics and tests only.
  std::atomic<size_t> size_{0};
  bool closed_ ABSL_GUARDED_BY(mutex_) = false;
};

}  // namespace lczero::lc5
