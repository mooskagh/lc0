#pragma once

#include "chess/position.h"
#include "search/lc3/positions.h"
#include "search/lc3/storage.h"
#include "search/lc3/treedata.h"
#include "search/lc3/worktree.h"

namespace lczero {
namespace lc3 {

class MctsWorker {
 public:
  MctsWorker(NodeStorage* storage, PositionChain head)
      : storage_(storage), head_(head) {}

  void Abort() { TODO(); }
  void Wait() { TODO(); }

  void GatherDescent(size_t target_batch_size);

  void OneStep() { GatherDescent(256); }

 private:
  std::deque<WorkTreeNode> work_tree_nodes_;
  NodeStorage* storage_;
  PositionChain head_;
};

}  // namespace lc3
}  // namespace lczero