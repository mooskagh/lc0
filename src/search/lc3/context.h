#pragma once

#include <cstddef>
#include <cstdint>

#include "search/lc3/positions.h"

namespace lczero {

template <typename T, size_t N>
class FreeListAllocator;

namespace lc3 {

class NodeStorage;
class SearchChannels;
struct EvalItem;

using EvalItemPool = FreeListAllocator<EvalItem, 1024>;

struct Context {
  NodeStorage* storage;
  SearchChannels* search_channels;
  Variation* head;
  EvalItemPool* eval_item_pool;
};

}  // namespace lc3
}  // namespace lczero