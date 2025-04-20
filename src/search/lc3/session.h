#pragma once

#include <memory>
#include <vector>

#include "search/lc3/mcts_worker.h"
#include "utils/exception.h"

namespace lczero {
namespace lc3 {

class SearchSession {
 public:
  SearchSession(NodeStorage* storage, PositionChain head)
      : search_(std::make_unique<MctsWorker>(storage, head, &eval_queue_)) {}

  void Abort() { NotImplemented(); }
  void Wait() { NotImplemented(); }
  void OneStep() { search_->GatherDescent(256); }

 private:
  std::unique_ptr<MctsWorker> search_;
  // std::unique_ptr<EvalWorker> eval_worker_;
  EvalQueue eval_queue_;
};

}  // namespace lc3
}  // namespace lczero