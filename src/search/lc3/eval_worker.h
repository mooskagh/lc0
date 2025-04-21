#pragma once

#include <span>

#include "neural/backend.h"
#include "search/lc3/types.h"

namespace lczero {
namespace lc3 {

struct WorkTreeNode;

class EvalWorker {
 public:
  EvalWorker(SearchChannels* search_channels, Backend* backend)
      : search_channels_(search_channels), backend_(backend) {}

  void OneStep();

 private:
  void NotifyEvalTaskDone(EvalTask* task) { NotImplemented(); }
  void NotifyAllPendingTasksDone() { NotImplemented(); }

  void Gather();
  void EnqueueIncomingTasks(std::span<EvalTask*> tasks);

  SearchChannels* const search_channels_;

  Backend* const backend_;
  std::unique_ptr<BackendComputation> computation_;
  std::vector<EvalTask*> tasks_to_notify_;
};

}  // namespace lc3
}  // namespace lczero