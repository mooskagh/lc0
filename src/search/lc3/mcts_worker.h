#pragma once

#include "chess/position.h"
#include "search/lc3/positions.h"
#include "search/lc3/storage.h"
#include "search/lc3/treedata.h"
#include "search/lc3/types.h"

namespace lczero {
namespace lc3 {

class MctsWorker {
 public:
  MctsWorker(EvalQueue* eval_queue, NodeStorage* storage, PositionChain head);

  void Abort() { TODO(); }
  void Wait() { TODO(); }

  void GatherDescent(size_t target_batch_size);

 private:
  NodeStorage* const storage_;
  std::unique_ptr<WorkTreeNode> root_;
  EvalQueue* const eval_queue_;
  moodycamel::ProducerToken ptok_{*eval_queue_};
};

}  // namespace lc3
}  // namespace lczero