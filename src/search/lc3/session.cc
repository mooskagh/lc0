#include "search/lc3/session.h"

namespace lczero {
namespace lc3 {

SearchSession::SearchSession(ThreadPool* thread_pool,
                             NodeRepository* node_repository,
                             const GameState& game_state, Backend* backend,
                             UciResponder* uci_responder,
                             const OptionsDict* options)
    : position_tree_(
          /*hash=*/NodeKey{game_state.startpos.Hash()},
          /*position=*/game_state.startpos,
          /*depth=*/0,
          /*idx_in_parent=*/-1),
      head_(position_tree_.root()),
      gather_rate_limiter_{
          .condition = {this, &SearchSession::OkToGather},
      },
      settings_(*options) {
  for (const auto& move : game_state.moves) {
    head_ = head_.make_child(
        /*hash=*/NodeKey{HashCat(head_->key.hash, move.raw_data())},
        /*position=*/Position(head_->position, move),
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
                                .eval_item_pool = &eval_item_pool_,
                                .gather_can_exit = &gather_can_exit_});
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
                                      .eval_item_pool = &eval_item_pool_});
      });
  watchdog_worker_.Start(thread_pool, 1, [&]() {
    return std::make_unique<WatchdogWorker>(WatchdogWorkerEnvironment{
        .node_repository = node_repository,
        .head = &head_,
        .uci_responder = uci_responder,
        .ok_to_respond_bestmove = &ok_to_respond_bestmove_,
        .must_exit = &watchdog_must_exit_,
    });
  });
}

void SearchSession::Abort() {
  ok_to_respond_bestmove_.store(false, std::memory_order_relaxed);
  DrainPipeline();
}
void SearchSession::Stop() { DrainPipeline(); }
void SearchSession::DrainPipeline() {
  // First, ensure bestmove is sent.
  watchdog_must_exit_.Notify();
  watchdog_worker_.Wait();
  // Then, stop all gather workers.
  gather_can_exit_.store(true, std::memory_order_relaxed);
  gather_workers_.Wait();
  // Then, set eval to drain mode, send sentinel, and wait for them to finish.
  eval_queue_.Drain();
  eval_workers_.Wait();
  // then, set backprop to drain mode, send sentinel, and wait for them to
  // finish.
  backprop_queue_.Drain();
  backprop_workers_.Wait();
}

void SearchSession::Wait() { NotImplemented(); }

}  // namespace lc3
}  // namespace lczero
