#include "search/lc3/session.h"

namespace lczero {
namespace lc3 {

SearchSession::SearchSession(NodeRepository* node_repository,
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

  // TODO Thread pool.
  for (int i = 0; i < settings_.GetNumGatherThreads(); ++i) {
    gather_workers_.emplace_back(std::make_unique<GatherWorker>(
        GatherWorkerEnvironment{.eval_sender = eval_queue_.MakeSender(),
                                .backprop_sender = backprop_queue_.MakeSender(),
                                .rate_limiter = &gather_rate_limiter_,
                                .node_repository = node_repository,
                                .head = &head_,
                                .eval_item_pool = &eval_item_pool_}));
    gather_threads_.emplace_back(&GatherWorker::Run,
                                 gather_workers_.back().get());
  }
  for (int i = 0; i < settings_.GetNumEvalThreads(); ++i) {
    eval_workers_.emplace_back(
        std::make_unique<EvalWorker>(EvalWorkerEnvironment{
            .eval_receiver = &eval_queue_,
            .backprop_sender = backprop_queue_.MakeSender(),
            .eval_queue_unblocker = &gather_rate_limiter_.mutex,
            .backend = backend}));
    eval_threads_.emplace_back(&EvalWorker::Run, eval_workers_.back().get());
  }
  for (int i = 0; i < settings_.GetNumBackpropThreads(); ++i) {
    backprop_workers_.emplace_back(std::make_unique<BackpropWorker>(
        BackpropWorkerEnvironment{.backprop_receiver = &backprop_queue_,
                                  .node_repository = node_repository,
                                  .eval_item_pool = &eval_item_pool_}));
    backprop_threads_.emplace_back(&BackpropWorker::Run,
                                   backprop_workers_.back().get());
  }
  watchdog_worker_ = std::make_unique<WatchdogWorker>(
      WatchdogWorkerEnvironment{.node_repository = node_repository,
                                .head = &head_,
                                .uci_responder = uci_responder});
  watchdog_thread_ = std::thread(&WatchdogWorker::Run, watchdog_worker_.get());
}

void SearchSession::Abort() { NotImplemented(); }

void SearchSession::Wait() { NotImplemented(); }

}  // namespace lc3
}  // namespace lczero
