#include "search/lc3/eval_worker.h"

#include <array>

#include "absl/synchronization/mutex.h"

namespace lczero {
namespace lc3 {

void EvalWorker::EnqueueIncomingTasks(std::span<EvalItem*> tasks) {
  // TODO absl REQUIRES_MUTEX(queue_mutex_)
  for (EvalItem* task : tasks) {
    const auto& board = task->variation->position.GetBoard();
    task->moves = board.GenerateLegalMoves();

    // Handle terminals.
    if (task->moves.empty()) {
      task->terminal_type = board.IsUnderCheck()
                                ? EvalItem::TerminalType::kCheckmate
                                : EvalItem::TerminalType::kDraw;
      SendCompletedEvalItem(task);
      continue;
    }
    if (!board.HasMatingMaterial() ||
        task->variation->position.GetRule50Ply() >= 100 ||
        GetPositionRepetitionCount(task->variation) >= 2) {
      // TODO have more proper handling of repetitions.
      task->terminal_type = EvalItem::TerminalType::kDraw;
      SendCompletedEvalItem(task);
      continue;
    }

    // Attempt to call the backend.
    task->p.resize(task->moves.size());
    std::array<Position, 8> positions;
    size_t num_positions = UnpackPositionsBackwards(task->variation, positions);
    const auto addinput_result = computation_->AddInput(
        EvalPosition{.pos = std::span<const Position>(
                         positions.begin() + (positions.size() - num_positions),
                         positions.end()),
                     .legal_moves = task->moves},
        EvalResultPtr{
            .q = &task->v, .d = &task->d, .m = &task->m, .p = task->p});
    if (addinput_result == BackendComputation::FETCHED_IMMEDIATELY) {
      SendCompletedEvalItem(task);
      continue;
    }
    batched_eval_items_.push_back(task);
  }
}

void EvalWorker::OneStep() {
  computation_ = backend_->CreateComputation();
  Gather();
  computation_->ComputeBlocking();
  SendCompletedBatchItems();
}

void EvalWorker::Gather() {
  const size_t recommended_batch_size =
      backend_->GetAttributes().recommended_batch_size;
  absl::MutexLock lock(&ctx_.search_channels->request_consumer_mutex_);
  // TODO replace with unique_ptr[]
  std::vector<EvalItem*> eval_tasks(recommended_batch_size);

  // While we have nothing to compute, wait blockingly.
  while (computation_->UsedBatchSize() == 0) {
    size_t num_nodes = ctx_.search_channels->FetchEvalRequests(
        std::span<EvalItem*>(eval_tasks.data(), recommended_batch_size),
        /*blocking=*/true);
    EnqueueIncomingTasks(std::span(eval_tasks).subspan(0, num_nodes));
  }

  // Now we have something to compute, but if we still have capacity, check if
  // there's more.
  while (computation_->UsedBatchSize() < recommended_batch_size) {
    size_t num_nodes = ctx_.search_channels->FetchEvalRequests(
        std::span<EvalItem*>(
            eval_tasks.data(),
            recommended_batch_size - computation_->UsedBatchSize()),
        /*blocking=*/false);
    if (num_nodes == 0) break;
    EnqueueIncomingTasks(std::span(eval_tasks).subspan(0, num_nodes));
  }
}

void EvalWorker::SendCompletedBatchItems() {
  ctx_.search_channels->SendEvalResults(eval_task_idx_,
                                        std::span(batched_eval_items_));
  batched_eval_items_.clear();
}

}  // namespace lc3
}  // namespace lczero