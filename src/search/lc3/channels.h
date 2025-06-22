#pragma once

#include "search/lc3/positions.h"
#include "third_party/moodycamel/blockingconcurrentqueue.h"
#include "third_party/moodycamel/concurrentqueue.h"

namespace lczero {
namespace lc3 {

struct EvalItem {
  EvalItem(Variation variation, size_t num_visits)
      : variation(std::move(variation)), num_visits(num_visits) {}

  EvalItem(Variation variation, size_t num_visits, bool is_terminal, float v,
           float d, float m)
      : variation(std::move(variation)),
        num_visits(num_visits),
        is_terminal(is_terminal),
        v(v),
        d(d),
        m(m) {}

  // Input.
  Variation variation;
  size_t num_visits;

  // Result.
  bool is_terminal;
  float v;
  float d;
  float m;
  std::vector<Move> moves;
  std::vector<float> p;
};

template <typename Vec, typename... Args>
void ResizeVector(Vec& vec, size_t size, Args&&... args) {
  if (vec.size() > size) vec.erase(vec.begin() + size, vec.end());
  while (vec.size() < size) vec.emplace_back(std::forward<Args>(args)...);
}

class GatherWorkerChannels {
 public:
  void SendForEval(EvalItem* item) {
    eval_queue_->enqueue(*eval_queue_token_, item);
  }
  void SendForBackprop(EvalItem* item) {
    backprop_queue_->enqueue(*backprop_queue_token_, item);
  }

 private:
  moodycamel::BlockingConcurrentQueue<EvalItem*>* const eval_queue_;
  moodycamel::ProducerToken* const eval_queue_token_;
  moodycamel::BlockingConcurrentQueue<EvalItem*>* const backprop_queue_;
  moodycamel::ProducerToken* const backprop_queue_token_;

  friend class SearchChannels;
  GatherWorkerChannels(
      moodycamel::BlockingConcurrentQueue<EvalItem*>* eval_queue,
      moodycamel::ProducerToken* eval_queue_token,
      moodycamel::BlockingConcurrentQueue<EvalItem*>* backprop_queue,
      moodycamel::ProducerToken* backprop_queue_token)
      : eval_queue_(eval_queue),
        eval_queue_token_(eval_queue_token),
        backprop_queue_(backprop_queue),
        backprop_queue_token_(backprop_queue_token) {}
};

class EvalWorkerChannels {
 public:
  absl::Mutex* EvalTasksMutex() { return eval_tasks_mutex_; }
  size_t CollectEvalTasks(std::span<EvalItem*> items, bool block)
      REQUIRES(eval_tasks_mutex_) {
    if (block) {
      return eval_queue_->wait_dequeue_bulk(*eval_queue_token_, items.data(),
                                            items.size());
    } else {
      return eval_queue_->try_dequeue_bulk(*eval_queue_token_, items.data(),
                                           items.size());
    }
  }
  void SendForBackprop(std::span<EvalItem*> items) {
    backprop_queue_->enqueue_bulk(*backprop_queue_token_, items.data(),
                                  items.size());
  }

 private:
  absl::Mutex* const eval_tasks_mutex_;
  moodycamel::BlockingConcurrentQueue<EvalItem*>* const eval_queue_;
  moodycamel::ConsumerToken* const eval_queue_token_;

  moodycamel::BlockingConcurrentQueue<EvalItem*>* const backprop_queue_;
  moodycamel::ProducerToken* const backprop_queue_token_;

  friend class SearchChannels;
  EvalWorkerChannels(
      absl::Mutex* eval_tasks_mutex,
      moodycamel::BlockingConcurrentQueue<EvalItem*>* eval_queue,
      moodycamel::ConsumerToken* eval_queue_token,
      moodycamel::BlockingConcurrentQueue<EvalItem*>* backprop_queue,
      moodycamel::ProducerToken* backprop_queue_token)
      : eval_tasks_mutex_(eval_tasks_mutex),
        eval_queue_(eval_queue),
        eval_queue_token_(eval_queue_token),
        backprop_queue_(backprop_queue),
        backprop_queue_token_(backprop_queue_token) {}
};

class BackpropWorkerChannels {
 public:
  size_t CollectBackpropTasks(std::span<EvalItem*> items, bool block) {
    if (block) {
      return backprop_queue_->wait_dequeue_bulk(*backprop_queue_token_,
                                                items.data(), items.size());
    } else {
      return backprop_queue_->try_dequeue_bulk(*backprop_queue_token_,
                                               items.data(), items.size());
    }
  }

  absl::Mutex* BackpropTasksMutex() { return backprop_tasks_mutex_; }

 private:
  absl::Mutex* backprop_tasks_mutex_;
  moodycamel::BlockingConcurrentQueue<EvalItem*>* const backprop_queue_;
  moodycamel::ConsumerToken* const backprop_queue_token_;

  friend class SearchChannels;
  BackpropWorkerChannels(
      absl::Mutex* backprop_tasks_mutex,
      moodycamel::BlockingConcurrentQueue<EvalItem*>* backprop_queue,
      moodycamel::ConsumerToken* backprop_queue_token)
      : backprop_tasks_mutex_(backprop_tasks_mutex),
        backprop_queue_(backprop_queue),
        backprop_queue_token_(backprop_queue_token) {}
};

class SearchChannels {
 public:
  SearchChannels(size_t num_gather_threads, size_t num_eval_threads) {
    Resize(num_gather_threads, num_eval_threads);
  }

  GatherWorkerChannels MakeGatherWorkerChannels(size_t gather_task_idx) {
    assert(gather_task_idx < gather_to_eval_tokens_.size());
    assert(gather_task_idx < gather_to_backprop_tokens_.size());
    return GatherWorkerChannels(
        &eval_queue_, &gather_to_eval_tokens_[gather_task_idx],
        &backprop_queue_, &gather_to_backprop_tokens_[gather_task_idx]);
  }

  EvalWorkerChannels MakeEvalWorkerChannels(size_t eval_task_idx) {
    assert(eval_task_idx < eval_to_backprop_tokens_.size());
    return EvalWorkerChannels(&eval_tasks_mutex_, &eval_queue_, &eval_token_,
                              &backprop_queue_,
                              &eval_to_backprop_tokens_[eval_task_idx]);
  }

  BackpropWorkerChannels MakeBackpropWorkerChannels() {
    return BackpropWorkerChannels(&backprop_tasks_mutex_, &backprop_queue_,
                                  &backprop_token_);
  }

  size_t GetApproximateNumPendingEvalRequests() const {
    return eval_queue_.size_approx();
  }

 private:
  void Resize(size_t num_gather_threads, size_t num_eval_threads) {
    ResizeVector(gather_to_eval_tokens_, num_gather_threads, eval_queue_);
    ResizeVector(gather_to_backprop_tokens_, num_gather_threads,
                 backprop_queue_);
    ResizeVector(eval_to_backprop_tokens_, num_eval_threads, backprop_queue_);
  }

  absl::Mutex eval_tasks_mutex_;
  absl::Mutex backprop_tasks_mutex_;

  using EvalItemQueue = moodycamel::BlockingConcurrentQueue<EvalItem*>;

  // Channel for sending from gather threads to eval threads.
  EvalItemQueue eval_queue_;
  std::vector<moodycamel::ProducerToken> gather_to_eval_tokens_;
  moodycamel::ConsumerToken eval_token_{eval_queue_};

  // Channel for sending from eval threads to backprop threads.
  EvalItemQueue backprop_queue_;
  std::vector<moodycamel::ProducerToken> gather_to_backprop_tokens_;
  std::vector<moodycamel::ProducerToken> eval_to_backprop_tokens_;
  moodycamel::ConsumerToken backprop_token_{backprop_queue_};
};

}  // namespace lc3
}  // namespace lczero