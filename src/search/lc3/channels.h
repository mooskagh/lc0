#pragma once

#include "search/lc3/positions.h"
#include "third_party/moodycamel/blockingconcurrentqueue.h"
#include "third_party/moodycamel/concurrentqueue.h"

namespace lczero {
namespace lc3 {

struct EvalItem {
  EvalItem(Variation variation, size_t num_visits)
      : variation(std::move(variation)), num_visits(num_visits) {}

  // Input.
  Variation variation;
  size_t num_visits;

  // Result.
  enum class TerminalType { kNonTerminal, kCheckmate, kDraw };
  TerminalType terminal_type{TerminalType::kNonTerminal};
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
  void SendEvalRequest(EvalItem* item) {
    request_queue_->enqueue(*request_producer_token_, item);
  }

 private:
  moodycamel::BlockingConcurrentQueue<EvalItem*>* const request_queue_;
  moodycamel::ProducerToken* const request_producer_token_;

  friend class SearchChannels;
  GatherWorkerChannels(
      moodycamel::BlockingConcurrentQueue<EvalItem*>* request_queue,
      moodycamel::ProducerToken* request_producer_token)
      : request_queue_(request_queue),
        request_producer_token_(request_producer_token) {}
};

class EvalWorkerChannels {
 public:
  size_t FetchEvalRequests(std::span<EvalItem*> items, bool block)
      REQUIRES(request_consumer_mutex_) {
    if (block) {
      return request_queue_->wait_dequeue_bulk(*request_consumer_token_,
                                               items.data(), items.size());
    } else {
      return request_queue_->try_dequeue_bulk(*request_consumer_token_,
                                              items.data(), items.size());
    }
  }
  void SendEvalResults(std::span<EvalItem*> items) {
    result_queue_->enqueue_bulk(*result_producer_token_, items.data(),
                                items.size());
  }

  absl::Mutex* GetMutex() { return request_consumer_mutex_; }

 private:
  absl::Mutex* const request_consumer_mutex_;
  moodycamel::BlockingConcurrentQueue<EvalItem*>* const request_queue_;
  moodycamel::ConsumerToken* const request_consumer_token_;

  moodycamel::BlockingConcurrentQueue<EvalItem*>* const result_queue_;
  moodycamel::ProducerToken* const result_producer_token_;

  friend class SearchChannels;
  EvalWorkerChannels(
      absl::Mutex* request_consumer_mutex,
      moodycamel::BlockingConcurrentQueue<EvalItem*>* request_queue,
      moodycamel::ConsumerToken* request_consumer_token,
      moodycamel::BlockingConcurrentQueue<EvalItem*>* result_queue,
      moodycamel::ProducerToken* result_producer_token)
      : request_consumer_mutex_(request_consumer_mutex),
        request_queue_(request_queue),
        request_consumer_token_(request_consumer_token),
        result_queue_(result_queue),
        result_producer_token_(result_producer_token) {}
};

class BackpropWorkerChannels {
 public:
  size_t FetchEvalResults(std::span<EvalItem*> items, bool block) {
    if (block) {
      return result_queue_->wait_dequeue_bulk(*result_consumer_token_,
                                              items.data(), items.size());
    } else {
      return result_queue_->try_dequeue_bulk(*result_consumer_token_,
                                             items.data(), items.size());
    }
  }

  absl::Mutex* GetMutex() { return result_consumer_mutex_; }

 private:
  absl::Mutex* result_consumer_mutex_;
  moodycamel::BlockingConcurrentQueue<EvalItem*>* const result_queue_;
  moodycamel::ConsumerToken* const result_consumer_token_;

  friend class SearchChannels;
  BackpropWorkerChannels(
      absl::Mutex* result_consumer_mutex,
      moodycamel::BlockingConcurrentQueue<EvalItem*>* result_queue,
      moodycamel::ConsumerToken* result_consumer_token)
      : result_consumer_mutex_(result_consumer_mutex),
        result_queue_(result_queue),
        result_consumer_token_(result_consumer_token) {}
};

class SearchChannels {
 public:
  SearchChannels(size_t num_gather_threads, size_t num_eval_threads) {
    Resize(num_gather_threads, num_eval_threads);
  }

  GatherWorkerChannels MakeGatherWorkerChannels(size_t gather_task_idx) {
    assert(gather_task_idx < request_producer_tokens_.size());
    return GatherWorkerChannels(&request_queue_,
                                &request_producer_tokens_[gather_task_idx]);
  }

  EvalWorkerChannels MakeEvalWorkerChannels(size_t eval_task_idx) {
    assert(eval_task_idx < result_producer_tokens_.size());
    return EvalWorkerChannels(&request_consumer_mutex_, &request_queue_,
                              &request_consumer_token_, &result_queue_,
                              &result_producer_tokens_[eval_task_idx]);
  }

  BackpropWorkerChannels MakeBackpropWorkerChannels(size_t backprop_task_idx) {
    assert(backprop_task_idx < result_producer_tokens_.size());
    return BackpropWorkerChannels(&result_consumer_mutex_, &result_queue_,
                                  &result_consumer_token_);
  }

  size_t GetApproximateNumPendingEvalRequests() const {
    return request_queue_.size_approx();
  }

 private:
  void Resize(size_t num_gather_threads, size_t num_eval_threads) {
    ResizeVector(request_producer_tokens_, num_gather_threads, request_queue_);
    ResizeVector(result_producer_tokens_, num_eval_threads, result_queue_);
  }
  
  absl::Mutex request_consumer_mutex_;
  absl::Mutex result_consumer_mutex_;

  using EvalItemQueue = moodycamel::BlockingConcurrentQueue<EvalItem*>;

  // Channel for sending from gather threads to eval threads.
  EvalItemQueue request_queue_;
  std::vector<moodycamel::ProducerToken> request_producer_tokens_;
  moodycamel::ConsumerToken request_consumer_token_{request_queue_};

  // Channel for sending from eval threads to backprop threads.
  EvalItemQueue result_queue_;
  std::vector<moodycamel::ProducerToken> result_producer_tokens_;
  moodycamel::ConsumerToken result_consumer_token_{result_queue_};
};

}  // namespace lc3
}  // namespace lczero