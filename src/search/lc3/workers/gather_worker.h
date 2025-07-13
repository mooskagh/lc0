#pragma once

#include "chess/position.h"
#include "search/lc3/node_repository/node_repository.h"
#include "search/lc3/search_policy/search_policy.h"
#include "search/lc3/workers/node_event_queue.h"
#include "utils/exception.h"

namespace lczero {
namespace lc3 {

// TODO make a class.
struct GatherRateLimiter {
  absl::Condition condition;
  absl::Mutex mutex = {};

  void Wait() {
    mutex.LockWhen(condition);
    mutex.Unlock();
  };
};

struct GatherWorkerEnvironment {
  NodeEventSender eval_sender;
  NodeEventSender backprop_sender;
  GatherRateLimiter* const rate_limiter;
  NodeRepository* node_repository;
  Variation* head;
  NodeEventPool* node_event_pool;
};

class GatherWorker {
 public:
  GatherWorker(GatherWorkerEnvironment env);
  ~GatherWorker();

  void Run();
  void Stop();

 private:
  using Policy = SearchPolicy;
  struct NodeAndBatch;

  void GatherDescent(size_t target_batch_size);
  void ProcessNode(size_t depth, NodeAndBatch& item);
  void ForwardToChildren(NodeHandle& handle, size_t depth, size_t batch_size,
                         size_t parent_n, Variation& node);

  void EnqueueNodeForEval(Variation&& node, size_t batch_size);
  void EnqueueNodeForBackprop(Variation&& node,
                              const NodeHandle::NodeAggregates& aggregates,
                              size_t batch_size);
  void EnqueueNodeForCollisionRollback(Variation&& node, size_t batch_size);
  template <typename... Args>
  NodeEvent* MakeNodeEvent(Args&&... args);

  GatherWorkerEnvironment env_;
  std::atomic<bool> stop_requested_{false};

  std::vector<NodeAndBatch> work_queue_;
  std::vector<NodeAndBatch> next_depth_work_queue_;
};

}  // namespace lc3
}  // namespace lczero