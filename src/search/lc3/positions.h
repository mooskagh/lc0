#pragma once

#include <vector>

#include "chess/gamestate.h"
#include "chess/position.h"
#include "search/lc3/node_repository.h"
#include "utils/tree.h"

namespace lczero {
namespace lc3 {

constexpr size_t kNoIdxInParent = static_cast<size_t>(-1);

struct VariationNode {
  NodeKey key;
  Position position;
  size_t depth;
  size_t idx_in_parent;

  VariationNode(NodeKey key, Position position, size_t depth,
                size_t idx_in_parent)
      : key(key),
        position(std::move(position)),
        depth(depth),
        idx_in_parent(idx_in_parent) {}
};

using PositionTree = Tree<VariationNode>;
using Variation = Tree<VariationNode>::node_handle;

inline NodeKey MakeNodeKey(const NodeKey& parent_key, Move move,
                           const Position& /*new_position*/) {
  return NodeKey{HashCat(parent_key.hash, move.raw_data())};
}

}  // namespace lc3
}  // namespace lczero