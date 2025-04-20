#pragma once

#include "third_party/moodycamel/blockingconcurrentqueue.h"

namespace lczero {
namespace lc3 {

struct WorkTreeNode;

class EvalWorker {
 public:
 private:
  moodycamel::BlockingConcurrentQueue<WorkTreeNode*>* task_queue_;
};

}  // namespace lc3
}  // namespace lczero