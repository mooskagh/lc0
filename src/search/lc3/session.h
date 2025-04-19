#pragma once

#include <memory>
#include <vector>

#include "search/lc3/mcts_worker.h"
#include "utils/exception.h"

namespace lczero {
namespace lc3 {

class SearchSession {
 public:
  void Abort() { NotImplemented(); }
  void Wait() { NotImplemented(); }

 private:
  std::vector<std::unique_ptr<MctsWorker>> mcts_threads_;
};

}  // namespace lc3
}  // namespace lczero