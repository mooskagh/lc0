#pragma once

#include "chess/position.h"
#include "search/lc3/storage.h"
#include "search/lc3/treedata.h"
#include "search/lc3/worktree.h"

namespace lczero {
namespace lc3 {

class Search {
 public:
  void GatherDescent(const Position& head, size_t target_batch_size);

 private:
  WorkTree work_tree_;

  std::vector<WorkTreeNode> work_tree_nodes_;
  Storage storage_;
};

}  // namespace lc3
}  // namespace lczero