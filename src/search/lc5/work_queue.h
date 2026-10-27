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
    { std::lock_guard lock(mutex_); }
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
