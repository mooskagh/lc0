#pragma once

#include "search/lc3/positions.h"
#include "third_party/moodycamel/blockingconcurrentqueue.h"
#include "third_party/moodycamel/concurrentqueue.h"

namespace lczero {
namespace lc3 {

struct EvalTask {
  EvalTask(Variation* variation, size_t num_visits)
      : variation(variation), num_visits(num_visits) {}

  // Input.
  Variation* variation;
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
  SearchChannels(size_t num_gather_threads, size_t num_eval_threads);
  void SendEvalRequests(size_t gather_task_idx, std::span<EvalTask*> tasks);
  size_t FetchEvalRequests(std::span<EvalTask*> tasks, bool block)
      REQUIRES(request_consumer_mutex_);

  // TODO public mutex is ugly.
  absl::Mutex request_consumer_mutex_;
};

// class SearchChannels {
//  public:
//   SearchChannels(size_t num_mcts_threads, size_t num_eval_threads) {
//     Resize(num_mcts_threads, num_eval_threads);
//   }

//   void SendRequests(size_t gather_task_idx, std::span<EvalTask*> tasks) {
//     assert(gather_task_idx < request_producer_tokens_.size());
//     request_queue_.enqueue_bulk(request_producer_tokens_[gather_task_idx],
//                                 tasks.data(), tasks.size());
//   }
//   size_t FetchRequests(std::span<EvalTask*> tasks, bool block)
//       REQUIRES(request_consumer_mutex_) {
//     if (block) {
//       return request_queue_.wait_dequeue_bulk(request_consumer_token_,
//                                               tasks.data(), tasks.size());
//     } else {
//       return request_queue_.try_dequeue_bulk(request_consumer_token_,
//                                              tasks.data(), tasks.size());
//     }
//   }
//   void SendResult(size_t /*eval_task_idx*/, size_t /*gather_task_idx*/,
//                   EvalTask*) {
//     NotImplemented();
//   }
//   void SendResults(size_t eval_task_idx, size_t gather_task_idx,
//                    std::span<EvalTask*> tasks) {
//     result_queues_[gather_task_idx].enqueue_bulk(
//         result_producer_tokens_[eval_task_idx][gather_task_idx],
//         tasks.data(), tasks.size());
//   }
//   size_t FetchResults(std::span<EvalTask*>, bool block) { NotImplemented(); }

//   void Resize(size_t num_mcts_threads, size_t num_eval_threads) {
//     auto resize = [](auto& vec, size_t size, auto&& initializer) {
//       if (vec.size() > size) vec.erase(vec.begin() + size, vec.end());
//       while (vec.size() < size) vec.emplace_back(initializer(vec.size()));
//     };
//     resize(request_producer_tokens_, num_mcts_threads, [this](size_t) {
//       return moodycamel::ProducerToken(request_queue_);
//     });
//     result_queues_.resize(num_mcts_threads);
//     result_producer_tokens_.resize(num_eval_threads);
//     for (auto& token_vec : result_producer_tokens_) {
//       resize(token_vec, num_mcts_threads, [&](size_t i) {
//         return moodycamel::ProducerToken(result_queues_[i]);
//       });
//     }
//   }

//   size_t GetNumMctsThreads() const { return request_producer_tokens_.size();
//   }

//   // TODO public mutex is ugly.
//   absl::Mutex request_consumer_mutex_;

//  private:
//   using EvalRequestQueue = moodycamel::BlockingConcurrentQueue<EvalTask*>;
//   using EvalResultQueue = moodycamel::ConcurrentQueue<EvalTask*>;

//   EvalRequestQueue request_queue_;
//   std::vector<moodycamel::ProducerToken> request_producer_tokens_;
//   moodycamel::ConsumerToken request_consumer_token_{request_queue_};

//   std::vector<EvalResultQueue> result_queues_;
//   std::vector<moodycamel::ConsumerToken> result_consumer_tokens_;
//   // Outer vector is for each eval thread, inner vector is for each mcts
//   thread. std::vector<std::vector<moodycamel::ProducerToken>>
//   result_producer_tokens_;
// };

}  // namespace lc3
}  // namespace lczero