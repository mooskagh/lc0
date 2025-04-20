#pragma once

#include <cstdint>

#include "search/lc3/positions.h"
#include "search/lc3/storage.h"

namespace lczero {
namespace lc3 {

struct WorkTreeNode {
  WorkTreeNode(WorkTreeNode* parent, PositionChain position,
               uint8_t index_in_parent)
      : parent(parent), position(position), index_in_parent(index_in_parent) {}

  WorkTreeNode* parent;
  PositionChain position;
  uint8_t index_in_parent;
  std::vector<std::unique_ptr<WorkTreeNode>> children;
};

}  // namespace lc3
}  // namespace lczero