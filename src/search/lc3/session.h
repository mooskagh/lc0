#pragma once

#include <memory>
#include <vector>

#include "search/lc3/eval_worker.h"
#include "search/lc3/gather_worker.h"
#include "search/lc3/positions.h"
#include "utils/exception.h"

namespace lczero {
namespace lc3 {

class SearchSession {
 public:
  SearchSession(NodeStorage* storage, const GameState& game_state,
                Backend* backend)
      : position_tree_(game_state.startpos),
        search_channels_(1, 1)  // TODO: make this configurable
  {
    Variation* head = position_tree_.GetRoot();
    for (const auto& move : game_state.moves) {
      head = position_tree_.MakeVariation(head, move, kNoIdxInParent);
    }
    Context context{
        .storage = storage,
        .position_tree = &position_tree_,
        .search_channels = &search_channels_,
        .head = head,
        .eval_item_pool = &eval_item_pool_,
    };
    mcts_worker_ =
        std::make_unique<MctsGatherWorker>(context, /*gather_task_idx=*/0);
    eval_worker_ =
        std::make_unique<EvalWorker>(context, /*eval_task_idx=*/0, backend);
  }

  void Abort() { NotImplemented(); }
  void Wait() { NotImplemented(); }
  void OneStep() {
    mcts_worker_->GatherDescent(256);
    eval_worker_->OneStep();
  }

 private:
  PositionTree position_tree_;
  SearchChannels search_channels_;
  std::unique_ptr<MctsGatherWorker> mcts_worker_;
  std::unique_ptr<EvalWorker> eval_worker_;
  EvalItemPool eval_item_pool_;
};

}  // namespace lc3
}  // namespace lczero