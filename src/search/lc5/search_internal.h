#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <utility>

#include "search/lc5/node_store.h"
#include "search/lc5/value.h"

namespace lczero::lc5::detail {

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
  bool Pop(T* value) {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [&] { return closed_ || !queue_.empty(); });
    if (queue_.empty()) return false;
    *value = std::move(queue_.front());
    queue_.pop_front();
    size_.store(queue_.size(), std::memory_order_relaxed);
    return true;
  }
  bool TryPop(T* value) {
    std::lock_guard lock(mutex_);
    if (queue_.empty()) return false;
    *value = std::move(queue_.front());
    queue_.pop_front();
    size_.store(queue_.size(), std::memory_order_relaxed);
    return true;
  }
  // The predicate reads only atomic scheduler state. WakeAll synchronizes
  // external predicate transitions with this queue's wait, avoiding lost wakes.
  template <typename Predicate>
  bool PopUntil(T* value, std::chrono::steady_clock::time_point deadline,
                Predicate flush) {
    std::unique_lock lock(mutex_);
    cv_.wait_until(lock, deadline,
                   [&] { return closed_ || !queue_.empty() || flush(); });
    if (queue_.empty()) return false;
    *value = std::move(queue_.front());
    queue_.pop_front();
    size_.store(queue_.size(), std::memory_order_relaxed);
    return true;
  }
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
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<T> queue_;
  std::atomic<size_t> size_{0};
  bool closed_ = false;
};

template <typename T>
void RaiseHighWater(std::atomic<T>& target, T value) {
  T old = target.load(std::memory_order_relaxed);
  while (old < value &&
         !target.compare_exchange_weak(old, value, std::memory_order_relaxed)) {
  }
}

inline SearchValue TerminalValue(TerminalKind terminal) {
  return terminal == TerminalKind::kCheckmate ? SearchValue{-1.0f, 0.0f, 0.0f}
                                              : SearchValue{0.0f, 1.0f, 0.0f};
}

}  // namespace lczero::lc5::detail
