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
  std::atomic<bool>* gather_can_exit;
};

class GatherWorker {
 public:
  GatherWorker(GatherWorkerEnvironment env) : env_(std::move(env)) {}

  void Abort() { TODO(); }
  void Wait() { TODO(); }

  void Run();

 private:
  void GatherDescent(size_t target_batch_size);

  void EnqueueNodeForEval(Variation&& node, size_t batch_size);
  void EnqueueNodeForBackprop(Variation&& node,
                              const NodeHandle::NodeAggregates& aggregates,
                              size_t batch_size);
  void EnqueueNodeForCollisionRollback(Variation&& node, size_t batch_size);
  template <typename... Args>
  EvalItem* MakeEvalItem(Args&&... args);

  GatherWorkerEnvironment env_;
};

}  // namespace lc3
}  // namespace lczero