#pragma once

#include "chess/position.h"
#include "search/lc3/channels.h"
#include "search/lc3/positions.h"
#include "search/lc3/storage.h"
#include "search/lc3/treedata.h"
#include "utils/freelist.h"

namespace lczero {
namespace lc3 {

class MctsWorker {
 public:
  MctsWorker(SearchChannels* search_channels, size_t worker_idx,
             NodeStorage* storage, PositionChain head);

  void Abort() { TODO(); }
  void Wait() { TODO(); }

  void GatherDescent(size_t target_batch_size);

 private:
  NodeStorage* const storage_;
  std::unique_ptr<WorkTreeNode> root_;

  SearchChannels* const search_channels_;
  const size_t worker_idx_;

  FreeList<EvalTask, 1024> eval_task_pool_;
};

}  // namespace lc3
}  // namespace lczero