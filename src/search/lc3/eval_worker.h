#pragma once

#include <span>

#include "neural/backend.h"
#include "search/lc3/channels.h"

namespace lczero {
namespace lc3 {

struct WorkTreeNode;

class EvalWorker {
 public:
  EvalWorker(SearchChannels* search_channels, size_t eval_task_idx,
             Backend* backend)
      : search_channels_(search_channels),
        eval_task_idx_(eval_task_idx),
        backend_(backend) {}

  void OneStep();

 private:
  void NotifyEvalTaskDone(EvalTask* task) { NotImplemented(); }
  void NotifyAllPendingTasksDone();

  void Gather();
  void EnqueueIncomingTasks(std::span<EvalTask*> tasks);

  SearchChannels* const search_channels_;
  const size_t eval_task_idx_;

  Backend* const backend_;
  std::unique_ptr<BackendComputation> computation_;
  std::vector<std::vector<EvalTask*>> tasks_to_notify_;
};

}  // namespace lc3
}  // namespace lczero