#pragma once

#include "utils/thread_pool.h"

namespace lczero {

template <typename T>
class WorkerPool {
 public:
  ~WorkerPool() { Wait(); }

  template <typename F>
  void Start(ThreadPool* thread_pool, size_t count, F&& factory) {
    for (size_t i = 0; i < count; ++i) {
      workers_.emplace_back(factory());
      futures_.emplace_back(
          thread_pool->Enqueue(&T::Run, workers_.back().get()));
    }
  }

  void Wait() {
    for (auto& future : futures_) future.wait();
  }

  T* GetWorker(size_t index) { return workers_.at(index).get(); }

 private:
  std::vector<std::unique_ptr<T>> workers_;
  std::vector<std::future<void>> futures_;
};

}  // namespace lczero
