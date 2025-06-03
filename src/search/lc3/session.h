#pragma once

#include <memory>
#include <vector>

#include "search/lc3/backprop_worker.h"
#include "search/lc3/eval_worker.h"
#include "search/lc3/gather_worker.h"
#include "search/lc3/positions.h"
#include "utils/exception.h"
#include "utils/freelist.h"

namespace lczero {
namespace lc3 {

class SearchSession {
 public:
  SearchSession(NodeRepository* node_repository, const GameState& game_state,
                Backend* backend)
      : position_tree_(
            /*hash=*/NodeHash{game_state.startpos.Hash()},
            /*position=*/game_state.startpos,
            /*depth=*/0,
            /*idx_in_parent=*/-1),
        head_(position_tree_.root()),
        search_channels_(1, 1)  // TODO: make this configurable
  {
    for (const auto& move : game_state.moves) {
      head_ = head_.make_child(
          /*hash=*/NodeHash{HashCat(head_->hash.hash, move.raw_data())},
          /*position=*/Position(head_->position, move),
          /*depth=*/head_->depth + 1,
          /*idx_in_parent=*/kNoIdxInParent);
    }
    Context context{
        .node_repository = node_repository,
        .search_channels = &search_channels_,
        .head = &head_,
        .eval_item_pool = &eval_item_pool_,
    };
    gather_worker_ =
        std::make_unique<MctsGatherWorker>(context, /*gather_task_idx=*/0);
    eval_worker_ =
        std::make_unique<EvalWorker>(context, /*eval_task_idx=*/0, backend);
    backprop_worker_ =
        std::make_unique<BackpropWorker>(context,
                                         /*backprop_task_idx=*/0);
  }

  void Abort() { NotImplemented(); }
  void Wait() { NotImplemented(); }
  void OneStep() {
    for (int i = 0; i < 2; ++i) {
      gather_worker_->GatherDescent(256);
      eval_worker_->OneStep();
      backprop_worker_->OneStep();
      CERR << "Done " << i << " step(s) of the search session.";
    }
  }

 private:
  PositionTree position_tree_;
  Variation head_;
  SearchChannels search_channels_;
  std::unique_ptr<MctsGatherWorker> gather_worker_;
  std::unique_ptr<EvalWorker> eval_worker_;
  std::unique_ptr<BackpropWorker> backprop_worker_;
  EvalItemPool eval_item_pool_;
};

}  // namespace lc3
}  // namespace lczero