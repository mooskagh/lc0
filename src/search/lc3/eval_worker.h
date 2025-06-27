#pragma once

#include <span>

#include "neural/backend.h"
#include "search/lc3/channels.h"
#include "search/lc3/context.h"

namespace lczero {
namespace lc3 {

struct WorkTreeNode;

struct EvalWorkerQueues {
  EvalItemReceiver* const eval_receiver;
  EvalItemSender backprop_sender;
  absl::Mutex* const eval_queue_unblocker;
};

class EvalWorker {
 public:
  EvalWorker(const Context& context, EvalWorkerQueues queues, Backend* backend)
      : ctx_(context), queues_(std::move(queues)), backend_(backend) {}

  void Run();

 private:
  void OneStep();
  void SendCompletedEvalItem(EvalItem*);
  void SendCompletedBatchItems();

  void Collect();
  void EnqueueIncomingTasks(std::span<EvalItem*> tasks);

  Context ctx_;
  EvalWorkerQueues queues_;

  Backend* const backend_;
  std::unique_ptr<BackendComputation> computation_;
  std::vector<EvalItem*> batched_eval_items_;
};

}  // namespace lc3
}  // namespace lczero