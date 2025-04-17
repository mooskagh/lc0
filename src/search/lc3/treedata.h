#pragma once

#include <cstdint>

#include "search/lc3/storage.h"
#include "search/lc3/worktree.h"

namespace lczero {
namespace lc3 {

struct WorkTreeNode {
  TreeNodeId parent;
  NodeHash node_hash;
};

}  // namespace lc3
}  // namespace lczero