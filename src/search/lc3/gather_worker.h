#pragma once

#include "chess/position.h"
#include "search/lc3/channels.h"
#include "search/lc3/context.h"
#include "utils/exception.h"

namespace lczero {
namespace lc3 {

struct GatherWorkerQueues {
  EvalItemSender eval_sender;
  EvalItemSender backprop_sender;
};

class MctsGatherWorker {
 public:
  MctsGatherWorker(const Context& context, GatherWorkerQueues);

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

  Context const ctx_;
  GatherWorkerQueues queues_;
};

}  // namespace lc3
}  // namespace lczero