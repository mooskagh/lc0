#pragma once

#include <cstdint>

#include "search/lc3/storage.h"

namespace lczero {

using TreeNodeId = uint16_t;

class WorkTree {
 public:
 TreeNodeId MakeRootNode();
};

}  // namespace lczero