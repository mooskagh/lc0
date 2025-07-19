#include "search/lc3/session.h"

namespace lczero {
namespace lc3 {

SearchSession::SearchSession(ThreadPool* thread_pool,
                             NodeRepository* node_repository,
                             const GameState& game_state, Backend* backend,
                             UciResponder* uci_responder,
                             const OptionsDict* options)
    : position_tree_(
          /*key=*/NodeKey{game_state.startpos.Hash()},
          /*position=*/game_state.startpos,
          /*depth=*/0,
          /*idx_in_parent=*/-1),
      head_(position_tree_.root()),
      gather_rate_limiter_{
          .condition = {this, &SearchSession::OkToGather},
      },
      settings_(*options) {
  for (const auto& move : game_state.moves) {
    Position move_position = Position(head_->position, move);
    head_ = head_.make_child(
        /*key=*/Policy::MakeNodeKey(head_->key, move, move_position),
        /*position=*/move_position,
        /*depth=*/head_->depth + 1,
        /*idx_in_parent=*/kNoIdxInParent);
  }

  gather_workers_.Start(thread_pool, settings_.GetNumGatherThreads(), [&]() {
    return std::make_unique<GatherWorker>(
        GatherWorkerEnvironment{.eval_sender = eval_queue_.MakeSender(),
                                .backprop_sender = backprop_queue_.MakeSender(),
                                .rate_limiter = &gather_rate_limiter_,
                                .node_repository = node_repository,
                                .head = &head_,
                                .node_event_pool = &node_event_pool_});
  });
  eval_workers_.Start(thread_pool, settings_.GetNumEvalThreads(), [&]() {
    return std::make_unique<EvalWorker>(EvalWorkerEnvironment{
        .eval_receiver = &eval_queue_,
        .backprop_sender = backprop_queue_.MakeSender(),
        .gather_worker_unblocker = &gather_rate_limiter_.mutex,
        .backend = backend});
  });
  backprop_workers_.Start(
      thread_pool, settings_.GetNumBackpropThreads(), [&]() {
        return std::make_unique<BackpropWorker>(
            BackpropWorkerEnvironment{.backprop_receiver = &backprop_queue_,
                                      .node_repository = node_repository,
                                      .eval_item_pool = &node_event_pool_});
      });
  watchdog_worker_.Start(thread_pool, 1, [&]() {
    return std::make_unique<WatchdogWorker>(WatchdogWorkerEnvironment{
        .node_repository = node_repository,
        .head = &head_,
        .uci_responder = uci_responder,
    });
  });
}

void SearchSession::Abort() {
  watchdog_worker_.NotifyAll([](WatchdogWorker* worker) {
    worker->Stop(/* must_respond_bestmove= */ false);
  });
  // Stop all gather workers.
  gather_workers_.NotifyAll([](GatherWorker* worker) { worker->Stop(); });
}

void SearchSession::Stop() {
  watchdog_worker_.NotifyAll([](WatchdogWorker* worker) {
    worker->Stop(/* must_respond_bestmove= */ true);
  });
  // Wait for the watchdog worker to respond best move.
  watchdog_worker_.Wait();
  // Then, stop all gather workers.
  gather_workers_.NotifyAll([](GatherWorker* worker) { worker->Stop(); });
}

void SearchSession::DrainPipeline() {
  // Wait for gather workers to stop.
  gather_workers_.Wait();
  // Then, set eval to drain mode, send sentinel, and wait for them to finish.
  eval_queue_.Drain();
  eval_workers_.Wait();
  // then, set backprop to drain mode, send sentinel, and wait for them to
  // finish.
  backprop_queue_.Drain();
  backprop_workers_.Wait();

  // If watchdog worker still happens to be alive, wait for it to finish.
  watchdog_worker_.Wait();
}

void SearchSession::Wait() { DrainPipeline(); }

}  // namespace lc3
}  // namespace lczero
