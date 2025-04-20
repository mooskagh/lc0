#pragma once

#include <span>

#include "neural/backend.h"
#include "search/lc3/types.h"

namespace lczero {
namespace lc3 {

struct WorkTreeNode;

class EvalWorker {
 public:
  EvalWorker(EvalQueue* eval_queue, moodycamel::ConsumerToken* ctok,
             absl::Mutex* queue_mutex, Backend* backend)
      : eval_queue_(eval_queue),
        ctok_(ctok),
        queue_mutex_(queue_mutex),
        backend_(backend) {}

  void OneStep();

 private:
  void NotifyEvalTaskDone(EvalTask* task) { NotImplemented(); }
  void NotifyAllPendingTasksDone() { NotImplemented(); }

  void Gather();
  void EnqueueIncomingTasks(std::span<EvalTask*> tasks);

  EvalQueue* const eval_queue_ GUARDED_BY(queue_mutex_);
  moodycamel::ConsumerToken* const ctok_;
  absl::Mutex* const queue_mutex_;

  Backend* const backend_;
  std::unique_ptr<BackendComputation> computation_;
  std::vector<EvalTask*> tasks_to_notify_;
};

}  // namespace lc3
}  // namespace lczero