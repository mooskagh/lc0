#pragma once

#include <span>

#include "neural/backend.h"
#include "search/lc3/channels.h"

namespace lczero {
namespace lc3 {

struct WorkTreeNode;

struct EvalWorkerEnvironment {
  EvalItemReceiver* const eval_receiver;
  EvalItemSender backprop_sender;
  absl::Mutex* const eval_queue_unblocker;
  Backend* const backend;
};

class EvalWorker {
 public:
  EvalWorker(EvalWorkerEnvironment env) : env_(std::move(env)) {}

  void Run();

 private:
  void OneStep();
  void SendCompletedEvalItem(EvalItem*);
  void SendCompletedBatchItems();

  void Collect();
  void EnqueueIncomingTasks(std::span<EvalItem*> tasks);

  EvalWorkerEnvironment env_;

  std::unique_ptr<BackendComputation> computation_;
  std::vector<EvalItem*> batched_eval_items_;
};

}  // namespace lc3
}  // namespace lczero