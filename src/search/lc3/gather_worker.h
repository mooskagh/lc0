#pragma once

#include "chess/position.h"
#include "search/lc3/context.h"
#include "utils/exception.h"

namespace lczero {
namespace lc3 {

class MctsGatherWorker {
 public:
  MctsGatherWorker(const Context& context, size_t gather_task_idx);

  void Abort() { TODO(); }
  void Wait() { TODO(); }

  void GatherDescent(size_t target_batch_size);

 private:
  const size_t gather_task_idx_;
  Context const ctx_;
};

}  // namespace lc3
}  // namespace lczero