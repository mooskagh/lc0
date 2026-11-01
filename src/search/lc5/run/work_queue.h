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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <utility>

namespace lczero::lc5 {

// Closable multi-producer multi-consumer FIFO.
template <typename T>
class WorkQueue {
 public:
  bool Push(T value) {
    {
      std::lock_guard lock(mutex_);
      if (closed_) return false;
      queue_.push_back(std::move(value));
      size_.store(queue_.size(), std::memory_order_relaxed);
    }
    cv_.notify_one();
    return true;
  }
  // Blocks until an item is available; false once closed and empty.
  bool Pop(T* value) {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [&] { return closed_ || !queue_.empty(); });
    return TakeFrontLocked(value);
  }
  bool TryPop(T* value) {
    std::lock_guard lock(mutex_);
    return TakeFrontLocked(value);
  }
  // Like Pop, but also gives up at `deadline` or once `give_up()` holds. The
  // predicate may read external state; WakeAll() makes its changes visible.
  template <typename Predicate>
  bool PopUntil(T* value, std::chrono::steady_clock::time_point deadline,
                Predicate give_up) {
    std::unique_lock lock(mutex_);
    cv_.wait_until(lock, deadline,
                   [&] { return closed_ || !queue_.empty() || give_up(); });
    return TakeFrontLocked(value);
  }
  // Locking before notifying avoids lost wakeups for external predicates.
  void WakeAll() {
    {
      std::lock_guard lock(mutex_);
    }
    cv_.notify_all();
  }
  void Close() {
    {
      std::lock_guard lock(mutex_);
      closed_ = true;
    }
    cv_.notify_all();
  }
  size_t Size() const { return size_.load(std::memory_order_relaxed); }

 private:
  bool TakeFrontLocked(T* value) {
    if (queue_.empty()) return false;
    *value = std::move(queue_.front());
    queue_.pop_front();
    size_.store(queue_.size(), std::memory_order_relaxed);
    return true;
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<T> queue_;
  std::atomic<size_t> size_{0};
  bool closed_ = false;
};

}  // namespace lczero::lc5
