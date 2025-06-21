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

class SearchChannels {
 public:
  SearchChannels(size_t num_gather_threads, size_t num_eval_threads) {
    Resize(num_gather_threads, num_eval_threads);
  }

  void SendEvalRequest(size_t gather_task_idx, EvalItem* item) {
    assert(gather_task_idx < request_producer_tokens_.size());
    request_queue_.enqueue(request_producer_tokens_[gather_task_idx], item);
  }

  void SendEvalRequests(size_t gather_task_idx, std::span<EvalItem*> items) {
    assert(gather_task_idx < request_producer_tokens_.size());
    request_queue_.enqueue_bulk(request_producer_tokens_[gather_task_idx],
                                items.data(), items.size());
  }
  size_t FetchEvalRequests(std::span<EvalItem*> items, bool block)
      REQUIRES(request_consumer_mutex_) {
    if (block) {
      return request_queue_.wait_dequeue_bulk(request_consumer_token_,
                                              items.data(), items.size());
    } else {
      return request_queue_.try_dequeue_bulk(request_consumer_token_,
                                             items.data(), items.size());
    }
  }
  void SendEvalResults(size_t eval_task_idx, std::span<EvalItem*> items) {
    assert(eval_task_idx < result_producer_tokens_.size());
    result_queue_.enqueue_bulk(result_producer_tokens_[eval_task_idx],
                               items.data(), items.size());
  }
  size_t FetchEvalResults(std::span<EvalItem*> items, bool block) {
    if (block) {
      return result_queue_.wait_dequeue_bulk(result_consumer_token_,
                                             items.data(), items.size());
    } else {
      return result_queue_.try_dequeue_bulk(result_consumer_token_,
                                            items.data(), items.size());
    }
  }

  void Resize(size_t num_gather_threads, size_t num_eval_threads) {
    ResizeVector(request_producer_tokens_, num_gather_threads, request_queue_);
    ResizeVector(result_producer_tokens_, num_eval_threads, result_queue_);
  }

  // TODO public mutex is ugly.
  absl::Mutex request_consumer_mutex_;
  absl::Mutex result_consumer_mutex_;

 private:
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