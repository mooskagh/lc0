#pragma once

#include <cstddef>
#include <cstdint>

#include "search/lc3/positions.h"

namespace lczero {

template <typename T, size_t N>
class FreeList;

namespace lc3 {

class NodeStorage;
class SearchChannels;
struct EvalItem;

using EvalItemPool = FreeList<EvalItem, 1024>;

struct Context {
  NodeStorage* storage;
  SearchChannels* search_channels;
  Variation* head;
  EvalItemPool* eval_item_pool;
};

}  // namespace lc3
}  // namespace lczero