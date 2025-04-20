#pragma once

#include "search/lc3/treedata.h"
#include "third_party/moodycamel/blockingconcurrentqueue.h"

namespace lczero {
namespace lc3 {

struct EvalTask {
    WorkTreeNode* pending_node;
    size_t num_visits;
};

using EvalQueue = moodycamel::BlockingConcurrentQueue<EvalTask*>;

}  // namespace lc3
}  // namespace lczero