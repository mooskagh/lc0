// #define LCZERO_DEBUG_LOGGING

#include "search/lc3/eval_worker.h"

#include <array>

#include "absl/synchronization/mutex.h"

namespace lczero {
namespace lc3 {

void EvalWorker::EnqueueIncomingTasks(std::span<EvalItem*> tasks) {
  // TODO absl REQUIRES_MUTEX(queue_mutex_)
  DPRINT_SCOPE("EnqueueIncomingTasks, size=" + std::to_string(tasks.size()));
  for (EvalItem* task : tasks) {
    const auto& board = task->variation->position.GetBoard();
    DPRINT_SCOPE("Board: " + board.DebugString());
    task->moves = board.GenerateLegalMoves();

    // Handle terminals.
    if (task->moves.empty()) {
      DPRINT << "Terminal position (no legal moves)";
      task->is_terminal = true;
      const bool is_under_check = board.IsUnderCheck();
      task->v = is_under_check ? -1.0f : 0.0f;
      task->d = is_under_check ? 0.0f : 1.0f;
      task->m = 0.0f;
      SendCompletedEvalItem(task);
      continue;
    }
    if (!board.HasMatingMaterial() ||
        task->variation->position.GetRule50Ply() >= 100 ||
        GetPositionRepetitionCount(task->variation) >= 2) {
      DPRINT << "Terminal position (draw by various rules)";
      task->is_terminal = true;
      task->v = 0.0f;
      task->d = 1.0f;
      task->m = 0.0f;
      task->moves.clear();
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
      DPRINT << "Fetched from cache";
      SendCompletedEvalItem(task);
      continue;
    }
    DPRINT << "Adding to batch for eval.";
    batched_eval_items_.push_back(task);
  }
}

void EvalWorker::OneStep() {
  computation_ = backend_->CreateComputation();
  Collect();
  DPRINT << computation_->UsedBatchSize() << " items to compute";
  if (computation_->UsedBatchSize() > 0) computation_->ComputeBlocking();
  SendCompletedBatchItems();
}

void EvalWorker::Collect() {
  const size_t recommended_batch_size =
      backend_->GetAttributes().recommended_batch_size;
  DPRINT_SCOPE("EvalWorker::Collect, recommended_batch_size=" +
               std::to_string(recommended_batch_size));
  absl::MutexLock lock(channels_.EvalTasksMutex());
  // TODO replace with unique_ptr[]
  std::vector<EvalItem*> eval_tasks(recommended_batch_size);

  // Do one blocking fetch to get initial work.
  {
    DPRINT << "Waiting blockingly";
    size_t num_nodes = channels_.CollectEvalTasks(
        std::span<EvalItem*>(eval_tasks.data(), recommended_batch_size),
        /*blocking=*/true);
    EnqueueIncomingTasks(std::span(eval_tasks).subspan(0, num_nodes));
  }

  // Now we have something to compute, but if we still have capacity, check if
  // there's more.
  while (computation_->UsedBatchSize() < recommended_batch_size) {
    size_t num_nodes = channels_.CollectEvalTasks(
        std::span<EvalItem*>(
            eval_tasks.data(),
            recommended_batch_size - computation_->UsedBatchSize()),
        /*blocking=*/false);
    if (num_nodes == 0) break;
    EnqueueIncomingTasks(std::span(eval_tasks).subspan(0, num_nodes));
  }
  DPRINT << "Collected " << batched_eval_items_.size() << " items";
}

void EvalWorker::SendCompletedEvalItem(EvalItem* item) {
  channels_.SendForBackprop(std::span(&item, 1));
}

void EvalWorker::SendCompletedBatchItems() {
  channels_.SendForBackprop(std::span(batched_eval_items_));
  batched_eval_items_.clear();
}

}  // namespace lc3
}  // namespace lczero