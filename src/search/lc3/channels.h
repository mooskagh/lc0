#pragma once

#include "search/lc3/treedata.h"
#include "third_party/moodycamel/blockingconcurrentqueue.h"
#include "third_party/moodycamel/concurrentqueue.h"

namespace lczero {
namespace lc3 {

struct EvalTask {
  size_t from_task_id;
  WorkTreeNode* pending_node;
  size_t num_visits;

  // Result.
  enum class TerminalType { kNonTerminal, kCheckmate, kDraw };
  TerminalType terminal_type{TerminalType::kNonTerminal};
  float q;
  float d;
  float m;
  std::vector<float> p;
};

class SearchChannels {
 public:
  SearchChannels(size_t size) { Resize(size); }

  void SendRequests(size_t task_idx, std::span<EvalTask*> tasks) {
    assert(task_idx < request_producer_tokens_.size());
    request_queue_.enqueue_bulk(request_producer_tokens_[task_idx],
                                tasks.data(), tasks.size());
  }
  size_t FetchRequests(std::span<EvalTask*> tasks, bool block)
      REQUIRES(request_consumer_mutex_) {
    if (block) {
      return request_queue_.wait_dequeue_bulk(request_consumer_token_,
                                              tasks.data(), tasks.size());
    } else {
      return request_queue_.try_dequeue_bulk(request_consumer_token_,
                                             tasks.data(), tasks.size());
    }
  }
  void SendResult(size_t to_task_idx, EvalTask* task) { NotImplemented(); }
  void SendResults(size_t to_task_idx, std::span<EvalTask*> tasks) {
    NotImplemented();
  }
  size_t FetchResults(std::span<EvalTask*>, bool block) { NotImplemented(); }

  void Resize(size_t count) {
    if (request_producer_tokens_.size() > count) {
      request_producer_tokens_.erase(request_producer_tokens_.begin() + count,
                                     request_producer_tokens_.end());
    }
    while (request_producer_tokens_.size() < count) {
      request_producer_tokens_.emplace_back(request_queue_);
    }
  }

  // TODO public mutex is ugly.
  absl::Mutex request_consumer_mutex_;

 private:
  using EvalRequestQueue = moodycamel::BlockingConcurrentQueue<EvalTask*>;
  using EvalResultQueue = moodycamel::ConcurrentQueue<EvalTask*>;

  EvalRequestQueue request_queue_;
  std::vector<moodycamel::ProducerToken> request_producer_tokens_;
  moodycamel::ConsumerToken request_consumer_token_{request_queue_};
};

}  // namespace lc3
}  // namespace lczero