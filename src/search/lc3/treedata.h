#pragma once

#include <cstdint>

#include "search/lc3/positions.h"
#include "search/lc3/storage.h"
#include "search/lc3/worktree.h"

namespace lczero {
namespace lc3 {

struct WorkTreeNode {
  WorkTreeNode(size_t parent_id, PositionChain position, size_t batch_size)
      : parent_id(parent_id), position(position), batch_size(batch_size) {}

  static constexpr size_t kNoParent = -1;
  size_t parent_id;
  PositionChain position;
  size_t batch_size;

  WorkTreeNode(const WorkTreeNode&) = delete;
  WorkTreeNode& operator=(const WorkTreeNode&) = delete;
  WorkTreeNode(WorkTreeNode&&) = delete;
  WorkTreeNode& operator=(WorkTreeNode&&) = delete;
};

}  // namespace lc3
}  // namespace lczero