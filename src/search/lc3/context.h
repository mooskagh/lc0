#pragma once

#include <cstddef>
#include <cstdint>

#include "search/lc3/positions.h"

namespace lczero {

template <typename T, size_t N>
class FreeListAllocator;

class UciResponder;

namespace lc3 {

class NodeRepository;
class SearchChannels;
struct EvalItem;

using EvalItemPool = FreeListAllocator<EvalItem, 1024>;

struct Context {
  NodeRepository* node_repository;
  SearchChannels* search_channels;
  Variation* head;
  EvalItemPool* eval_item_pool;
  UciResponder* uci_responder;
};

}  // namespace lc3
}  // namespace lczero