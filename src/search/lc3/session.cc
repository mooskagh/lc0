#include "search/lc3/session.h"

namespace lczero {
namespace lc3 {

SearchSession::SearchSession(NodeRepository* node_repository,
                             const GameState& game_state, Backend* backend,
                             UciResponder* uci_responder,
                             SearchChannels* search_channels,
                             const OptionsDict* options)
    : position_tree_(
          /*hash=*/NodeKey{game_state.startpos.Hash()},
          /*position=*/game_state.startpos,
          /*depth=*/0,
          /*idx_in_parent=*/-1),
      head_(position_tree_.root()),
      search_channels_(search_channels),
      settings_(*options) {
  for (const auto& move : game_state.moves) {
    head_ = head_.make_child(
        /*hash=*/NodeKey{HashCat(head_->key.hash, move.raw_data())},
        /*position=*/Position(head_->position, move),
        /*depth=*/head_->depth + 1,
        /*idx_in_parent=*/kNoIdxInParent);
  }
  Context context{
      .node_repository = node_repository,
      .head = &head_,
      .eval_item_pool = &eval_item_pool_,
      .uci_responder = uci_responder,
  };
  // TODO Thread pool.
  for (int i = 0; i < settings_.GetNumGatherThreads(); ++i) {
    gather_workers_.emplace_back(std::make_unique<MctsGatherWorker>(
        context, search_channels_->MakeGatherWorkerChannels()));
    threads_.emplace_back(&MctsGatherWorker::Run, gather_workers_.back().get());
  }
  for (int i = 0; i < settings_.GetNumEvalThreads(); ++i) {
    eval_workers_.emplace_back(std::make_unique<EvalWorker>(
        context, search_channels_->MakeEvalWorkerChannels(), backend));
    threads_.emplace_back(&EvalWorker::Run, eval_workers_.back().get());
  }
  for (int i = 0; i < settings_.GetNumBackpropThreads(); ++i) {
    backprop_workers_.emplace_back(std::make_unique<BackpropWorker>(
        context, search_channels_->MakeBackpropWorkerChannels()));
    threads_.emplace_back(&BackpropWorker::Run, backprop_workers_.back().get());
  }
  watchdog_worker_ = std::make_unique<WatchdogWorker>(context);
  threads_.emplace_back(&WatchdogWorker::Run, watchdog_worker_.get());
}

void SearchSession::Abort() { NotImplemented(); }

void SearchSession::Wait() { NotImplemented(); }

// void SearchSession::StartSyncronized() {
//   tmp_thread_ = std::thread([this]() {
//     for (int i = 0; i < 500000; ++i) {
//       gather_worker_->GatherDescent(2560);
//       while (search_channels_.GetApproximateNumPendingEvalRequests()) {
//         eval_worker_->OneStep();
//       }
//       backprop_worker_->OneStep();
//       watchdog_worker_->CheckOnce();
//     }
//   });
// }

}  // namespace lc3
}  // namespace lczero
