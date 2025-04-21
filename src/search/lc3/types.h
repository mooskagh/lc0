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
  void SendRequests(size_t task_idx, std::span<EvalTask*> tasks) {
    NotImplemented();
  }
  size_t FetchRequests(std::span<EvalTask*> tasks, bool block)
      REQUIRES(request_consumer_mutex_) {
    NotImplemented();
  }
  void SendResult(size_t to_task_idx, EvalTask* task) { NotImplemented(); }
  void SendResults(size_t to_task_idx, std::span<EvalTask*> tasks) {
    NotImplemented();
  }
  size_t FetchResults(std::span<EvalTask*>, bool block) { NotImplemented(); }

  // TODO public mutex is ugly.
  absl::Mutex request_consumer_mutex_;

 private:
  using EvalRequestQueue = moodycamel::BlockingConcurrentQueue<EvalTask*>;
  using EvalResultQueue = moodycamel::ConcurrentQueue<EvalTask*>;
};

}  // namespace lc3
}  // namespace lczero