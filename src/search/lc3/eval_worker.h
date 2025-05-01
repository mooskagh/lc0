#pragma once

#include <span>

#include "neural/backend.h"
#include "search/lc3/channels.h"
#include "search/lc3/context.h"

namespace lczero {
namespace lc3 {

struct WorkTreeNode;

class EvalWorker {
 public:
  EvalWorker(const Context& context, size_t eval_task_idx, Backend* backend)
      : ctx_(context), eval_task_idx_(eval_task_idx), backend_(backend) {}
  void OneStep();

 private:
  void NotifyEvalTaskDone(EvalItem*) { NotImplemented(); }
  void NotifyAllPendingTasksDone() { NotImplemented(); }

  void Gather();
  void EnqueueIncomingTasks(std::span<EvalItem*> tasks);

  Context ctx_;
  const size_t eval_task_idx_;

  Backend* const backend_;
  std::unique_ptr<BackendComputation> computation_;
  std::vector<EvalItem*> tasks_to_notify_after_computation_done_;
};

}  // namespace lc3
}  // namespace lczero