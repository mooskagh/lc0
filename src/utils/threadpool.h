#pragma once

#include <cstddef>
#include <deque>
#include <functional>
#include <future>

#include "absl/synchronization/mutex.h"

namespace lczero {

struct ThreadPoolOptions {
  // If true, starts new thread when task is enqueued and no threads are idle.
  bool grow_automatically = false;
};

class ThreadPool {
 public:
  ThreadPool(size_t initial_threads,
             const ThreadPoolOptions& options = ThreadPoolOptions());

  // Blocks until all tasks are completed.
  ~ThreadPool();

  // Enqueues a task for execution and returns a std::future.
  template <typename F, typename... Args>
  auto Enqueue(F&& f, Args&&... args)
      -> std::future<std::invoke_result_t<F, Args...>>;

  // Number of tasks that are not yet started.
  size_t num_pending_tasks() const;

  // Number of tasks that are currently running.
  size_t num_running_tasks_approx() const;

  // Number of worker threads (busy or not).
  size_t num_threads() const;

 private:
  void WorkerLoop();
  void StartWorkerThread() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  bool TaskAvailableCond() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return stop_ || !tasks_.empty();
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  ThreadPoolOptions options_;
  mutable absl::Mutex mutex_;

  std::vector<std::thread> threads_ ABSL_GUARDED_BY(mutex_);
  std::deque<std::function<void()>> tasks_ ABSL_GUARDED_BY(mutex_);
  bool stop_ ABSL_GUARDED_BY(mutex_) = false;

  std::atomic<size_t> running_tasks_{0};
};

inline ThreadPool::ThreadPool(size_t initial_threads,
                              const ThreadPoolOptions& options)
    : options_(options) {
  absl::MutexLock lock(&mutex_);
  for (size_t i = 0; i < initial_threads; ++i) {
    threads_.emplace_back(&ThreadPool::WorkerLoop, this);
  }
}

inline ThreadPool::~ThreadPool() {
  {
    absl::MutexLock lock(&mutex_);
    stop_ = true;
  }
  for (std::thread& worker : threads_) worker.join();
}

template <typename F, typename... Args>
auto ThreadPool::Enqueue(F&& f, Args&&... args)
    -> std::future<std::invoke_result_t<F, Args...>> {
  using ReturnType = std::invoke_result_t<F, Args...>;

  std::packaged_task<ReturnType()> task(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<ReturnType> future = task.get_future();

  {
    absl::MutexLock lock(&mutex_);
    // If all threads are busy, create a new one if allowed.
    const size_t idle_threads =
        threads_.size() - running_tasks_.load(std::memory_order_relaxed);
    if (options_.grow_automatically && idle_threads == 0) StartWorkerThread();
    tasks_.emplace_back([task = std::move(task)]() mutable { task(); });
  }

  task_available_cv_.Signal();
  return future;
}

inline void ThreadPool::WorkerLoop() {
  while (true) {
    std::function<void()> task;
    {
      mutex_.LockWhen(absl::Condition(this, &ThreadPool::TaskAvailableCond));
      if (stop_ && tasks_.empty()) return;
      task = std::move(tasks_.front());
      tasks_.pop_front();
    }

    running_tasks_.fetch_add(1, std::memory_order_relaxed);
    std::move(task)();
    running_tasks_.fetch_sub(1, std::memory_order_relaxed);
  }
}

inline void ThreadPool::StartWorkerThread() {
  threads_.emplace_back(&ThreadPool::WorkerLoop, this);
}

inline size_t ThreadPool::num_pending_tasks() const {
  absl::MutexLock lock(&mutex_);
  return tasks_.size();
}

inline size_t ThreadPool::num_running_tasks_approx() const {
  return running_tasks_.load(std::memory_order_relaxed);
}

inline size_t ThreadPool::num_threads() const {
  absl::MutexLock lock(&mutex_);
  return threads_.size();
}

}  // namespace lczero