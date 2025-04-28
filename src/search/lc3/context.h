#pragma once

#include <cstddef>
#include <cstdint>

namespace lczero {

template <typename T, size_t N>
class FreeList;

namespace lc3 {

class NodeStorage;
class PositionTree;
struct Variation;
class SearchChannels;
struct EvalTask;

using EvalTaskPool = FreeList<EvalTask, 1024>;

struct Context {
  NodeStorage* storage;
  PositionTree* position_tree;
  SearchChannels* search_channels;
  Variation* head;
  EvalTaskPool* eval_task_pool;
};

}  // namespace lc3
}  // namespace lczero