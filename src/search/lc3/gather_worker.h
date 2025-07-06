#pragma once

#include "chess/position.h"
#include "search/lc3/channels.h"
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
  EvalItemSender eval_sender;
  EvalItemSender backprop_sender;
  GatherRateLimiter* const rate_limiter;
  NodeRepository* node_repository;
  Variation* head;
  EvalItemPool* eval_item_pool;
};

class GatherWorker {
 public:
  GatherWorker(GatherWorkerEnvironment env);
  ~GatherWorker();

  void Run();
  void Stop();

 private:
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
  EvalItem* MakeEvalItem(Args&&... args);

  GatherWorkerEnvironment env_;
  std::atomic<bool> stop_requested_{false};

  std::vector<NodeAndBatch> work_queue_;
  std::vector<NodeAndBatch> next_depth_work_queue_;
};

}  // namespace lc3
}  // namespace lczero