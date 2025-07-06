#include "search/lc3/eval_worker.h"

#include <array>

#include "absl/synchronization/mutex.h"

namespace lczero {
namespace lc3 {

void EvalWorker::EnqueueIncomingEvents(std::span<NodeEvent*> events) {
  // TODO absl REQUIRES_MUTEX(queue_mutex_)
  for (NodeEvent* event : events) {
    if (!event) continue;  // Skip the sentinel used for draining.
    const auto& board = event->variation->position.GetBoard();
    event->moves = board.GenerateLegalMoves();

    // Handle terminals.
    if (event->moves.empty()) {
      event->result_type = NodeEvent::ResultType::kTerminal;
      const bool is_under_check = board.IsUnderCheck();
      event->v = is_under_check ? -1.0f : 0.0f;
      event->d = is_under_check ? 0.0f : 1.0f;
      event->m = 0.0f;
      SendCompletedNodeEvent(event);
      continue;
    }
    if (!board.HasMatingMaterial() ||
        event->variation->position.GetRule50Ply() >= 100 ||
        GetPositionRepetitionCount(event->variation) >= 2) {
      event->result_type = NodeEvent::ResultType::kTerminal;
      event->v = 0.0f;
      event->d = 1.0f;
      event->m = 0.0f;
      event->moves.clear();
      SendCompletedNodeEvent(event);
      continue;
    }

    event->result_type = NodeEvent::ResultType::kNormal;
    // Attempt to call the backend.
    event->p.resize(event->moves.size());
    std::array<Position, 8> positions;
    size_t num_positions =
        UnpackPositionsBackwards(event->variation, positions);
    const auto addinput_result = computation_->AddInput(
        EvalPosition{.pos = std::span<const Position>(
                         positions.begin() + (positions.size() - num_positions),
                         positions.end()),
                     .legal_moves = event->moves},
        EvalResultPtr{
            .q = &event->v, .d = &event->d, .m = &event->m, .p = event->p});
    if (addinput_result == BackendComputation::FETCHED_IMMEDIATELY) {
      SendCompletedNodeEvent(event);
      continue;
    }
    batched_node_events_.push_back(event);
  }
}

void EvalWorker::Run() { while (OneStep()); }

bool EvalWorker::OneStep() {
  computation_ = env_.backend->CreateComputation();
  if (!Collect()) return false;
  if (computation_->UsedBatchSize() > 0) computation_->ComputeBlocking();
  SendCompletedBatchItems();
  return true;
}

bool EvalWorker::Collect() {
  const size_t recommended_batch_size =
      env_.backend->GetAttributes().recommended_batch_size;

  absl::MutexLock lock(env_.eval_receiver->GetConsumerMutex());
  // TODO replace with unique_ptr[]
  std::vector<NodeEvent*> events(recommended_batch_size);

  // Do one blocking fetch to get initial work.
  {
    size_t num_nodes = env_.eval_receiver->Collect(
        std::span<NodeEvent*>(events.data(), recommended_batch_size),
        /*blocking=*/true);
    if (num_nodes == 0) {
      // Fetched 0 despite blocking, we are in the draining mode and queue is
      // empty. Return.
      return false;
    }
    EnqueueIncomingEvents(std::span(events).subspan(0, num_nodes));
  }

  // Now we have something to compute, but if we still have capacity, check if
  // there's more.
  while (computation_->UsedBatchSize() < recommended_batch_size) {
    size_t num_nodes = env_.eval_receiver->Collect(
        std::span<NodeEvent*>(events.data(), recommended_batch_size -
                                                 computation_->UsedBatchSize()),
        /*blocking=*/false);
    if (num_nodes == 0) break;
    EnqueueIncomingEvents(std::span(events).subspan(0, num_nodes));
  }

  // Now that we took some tasks from the queue, the gather worker may want to
  // continue working, so unblocking it.
  env_.gather_worker_unblocker->Lock();
  env_.gather_worker_unblocker->Unlock();
  return true;
}

void EvalWorker::SendCompletedNodeEvent(NodeEvent* event) {
  env_.backprop_sender.Enqueue(event);
}

void EvalWorker::SendCompletedBatchItems() {
  env_.backprop_sender.EnqueueBulk(std::span(batched_node_events_));
  batched_node_events_.clear();
}

}  // namespace lc3
}  // namespace lczero