#pragma once

#include "third_party/moodycamel/blockingconcurrentqueue.h"
#include "search/lc3/treedata.h"

namespace lczero {
namespace lc3 {

using EvalQueue = moodycamel::BlockingConcurrentQueue<WorkTreeNode*>;

}
}  // namespace lczero