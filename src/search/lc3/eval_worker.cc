#include "search/lc3/eval_worker.h"

#include <array>

namespace lczero {
namespace lc3 {

void EvalWorker::EnqueueIncomingTasks(std::span<EvalTask*> tasks) {
  // TODO REQUIRES_MUTEX(queue_mutex_)
  for (EvalTask* task : tasks) {
    const auto& board = task->pending_node->position.position.GetBoard();
    std::vector<Move> legal_moves = board.GenerateLegalMoves();

    // Handle terminals.
    if (legal_moves.empty()) {
      task->terminal_type = board.IsUnderCheck()
                                ? EvalTask::TerminalType::kCheckmate
                                : EvalTask::TerminalType::kDraw;
      NotifyEvalTaskDone(task);
      continue;
    }
    if (!board.HasMatingMaterial() ||
        task->pending_node->position.position.GetRule50Ply() >= 100 ||
        task->pending_node->position.GetRepetitionCount() >= 2) {
      // TODO have more proper handling of repetitions.
      task->terminal_type = EvalTask::TerminalType::kDraw;
      NotifyEvalTaskDone(task);
      continue;
    }

    // Attempt to call the backend.
    task->p.resize(legal_moves.size());
    std::array<Position, 8> positions;
    size_t num_positions =
        UnpackPositionsBackwards(task->pending_node->position, positions);
    const auto addinput_result = computation_->AddInput(
        EvalPosition{.pos = std::span<const Position>(
                         positions.begin() + (positions.size() - num_positions),
                         positions.end()),
                     .legal_moves = legal_moves},
        EvalResultPtr{
            .q = &task->q, .d = &task->d, .m = &task->m, .p = task->p});
    if (addinput_result == BackendComputation::FETCHED_IMMEDIATELY) {
      NotifyEvalTaskDone(task);
      continue;
    }
    tasks_to_notify_.push_back(task);
  }
}

void EvalWorker::OneStep() {
  auto computation_ = backend_->CreateComputation();
  Gather();
  computation_->ComputeBlocking();
  NotifyAllPendingTasksDone();
}

void EvalWorker::Gather() {
  const size_t recommended_batch_size =
      backend_->GetAttributes().recommended_batch_size;
  absl::MutexLock lock(queue_mutex_);
  // TODO replace with unique_ptr[]
  std::vector<EvalTask*> eval_tasks(recommended_batch_size);

  // While we have nothing to compute, wait blockingly.
  while (computation_->UsedBatchSize() == 0) {
    size_t num_nodes = eval_queue_->wait_dequeue_bulk(
        *ctok_, eval_tasks.begin(), recommended_batch_size);
    EnqueueIncomingTasks(std::span(eval_tasks).subspan(0, num_nodes));
  }

  // Now we have something to compute, but if we still have capacity, check if
  // there's more.
  while (computation_->UsedBatchSize() < recommended_batch_size) {
    size_t num_nodes = eval_queue_->try_dequeue_bulk(
        *ctok_, eval_tasks.begin(),
        recommended_batch_size - computation_->UsedBatchSize());
    if (num_nodes == 0) break;
    EnqueueIncomingTasks(std::span(eval_tasks).subspan(0, num_nodes));
  }
}

}  // namespace lc3
}  // namespace lczero