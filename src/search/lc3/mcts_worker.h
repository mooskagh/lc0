#pragma once

#include "chess/position.h"
#include "search/lc3/positions.h"
#include "search/lc3/storage.h"
#include "search/lc3/treedata.h"

namespace lczero {
namespace lc3 {

class MctsWorker {
 public:
  MctsWorker(NodeStorage* storage, PositionChain head);

  void Abort() { TODO(); }
  void Wait() { TODO(); }

  void GatherDescent(size_t target_batch_size);

  void OneStep() { GatherDescent(256); }

 private:
  NodeStorage* storage_;
  std::unique_ptr<WorkTreeNode> root_;
};

}  // namespace lc3
}  // namespace lczero