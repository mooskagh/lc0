#pragma once

#include "neural/backend.h"
#include "search/lc3/types.h"

namespace lczero {
namespace lc3 {

struct WorkTreeNode;

class EvalWorker {
 public:
  EvalWorker(EvalQueue* eval_queue, Backend* backend)
      : eval_queue_(eval_queue), backend_(backend) {}

 private:
  EvalQueue* const eval_queue_;
  Backend* const backend_;
};

}  // namespace lc3
}  // namespace lczero