#pragma once

#include <memory>
#include <vector>
#include <thread>

#include "search/lc3/backprop_worker.h"
#include "search/lc3/eval_worker.h"
#include "search/lc3/gather_worker.h"
#include "search/lc3/positions.h"
#include "search/lc3/watchdog_worker.h"
#include "utils/exception.h"
#include "utils/freelist.h"

namespace lczero {
namespace lc3 {

class SearchSession {
 public:
  SearchSession(NodeRepository* node_repository, const GameState& game_state,
                Backend* backend, UciResponder* uci_responder)
      : position_tree_(
            /*hash=*/NodeKey{game_state.startpos.Hash()},
            /*position=*/game_state.startpos,
            /*depth=*/0,
            /*idx_in_parent=*/-1),
        head_(position_tree_.root()),
        search_channels_(1, 1)  // TODO: make this configurable
  {
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
    gather_worker_ = std::make_unique<MctsGatherWorker>(
        context, search_channels_.MakeGatherWorkerChannels(
                     /*gather_task_idx=*/0));
    eval_worker_ =
        std::make_unique<EvalWorker>(context,
                                     search_channels_.MakeEvalWorkerChannels(
                                         /*eval_task_idx=*/0),
                                     backend);
    backprop_worker_ = std::make_unique<BackpropWorker>(
        context, search_channels_.MakeBackpropWorkerChannels());
    watchdog_worker_ = std::make_unique<WatchdogWorker>(context);
  }

  void Abort() { NotImplemented(); }
  void Wait() { NotImplemented(); }
  void StartSyncronized() {
    tmp_thread_ = std::thread([this]() {
      for (int i = 0; i < 500000; ++i) {
        gather_worker_->GatherDescent(2560);
        while (search_channels_.GetApproximateNumPendingEvalRequests()) {
          eval_worker_->OneStep();
        }
        backprop_worker_->OneStep();
        watchdog_worker_->CheckOnce();
      }
    });
  }

 private:
  PositionTree position_tree_;
  Variation head_;
  SearchChannels search_channels_;
  std::unique_ptr<MctsGatherWorker> gather_worker_;
  std::unique_ptr<EvalWorker> eval_worker_;
  std::unique_ptr<BackpropWorker> backprop_worker_;
  std::unique_ptr<WatchdogWorker> watchdog_worker_;
  EvalItemPool eval_item_pool_;

  // Temporary for debugging.
  std::thread tmp_thread_;
};

}  // namespace lc3
}  // namespace lczero