#pragma once

#include <memory>
#include <vector>

#include "search/lc3/eval_worker.h"
#include "search/lc3/mcts_worker.h"
#include "utils/exception.h"

namespace lczero {
namespace lc3 {

class SearchSession {
 public:
  SearchSession(NodeStorage* storage, PositionChain head, Backend* backend)
      : mcts_worker_(std::make_unique<MctsWorker>(&eval_queue_, storage, head)),
        eval_worker_(std::make_unique<EvalWorker>(&eval_queue_, backend)) {}

  void Abort() { NotImplemented(); }
  void Wait() { NotImplemented(); }
  void OneStep() { mcts_worker_->GatherDescent(256); }

 private:
  EvalQueue eval_queue_;
  std::unique_ptr<MctsWorker> mcts_worker_;
  std::unique_ptr<EvalWorker> eval_worker_;
};

}  // namespace lc3
}  // namespace lczero