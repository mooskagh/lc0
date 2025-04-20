#include "search/lc3/eval_worker.h"

namespace lczero {
namespace lc3 {

void EvalWorker::OneStep() {
  auto computation = backend_->CreateComputation();
  const size_t recommended_batch_size =
      backend_->GetAttributes().recommended_batch_size;
  {
    absl::MutexLock lock(queue_mutex_);
    // TODO replace with unique_ptr[]
    std::vector<EvalTask*> eval_tasks(recommended_batch_size);

    size_t num_nodes = eval_queue_->wait_dequeue_bulk(
        *ctok_, eval_tasks.begin(), recommended_batch_size);

    for (size_t i = 0; i < num_nodes; ++i) {
      std::vector<EvalTask*> eval_tasks(recommended_batch_size);
      EvalTask* task = eval_tasks[i];
    }
  }
}

}  // namespace lc3
}  // namespace lczero