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

// TODO Move this function somewhere else.
[[nodiscard]] inline size_t UnpackPositionsBackwards(
    Variation variation, std::span<Position> positions) {
  auto iter = positions.rbegin();
  const auto end = positions.rend();

  // TODO iterating to the parent touches ref counters back and forth.
  // If this shows up in profiles, optimize by going by raw pointers.
  while (iter != end && variation) {
    *iter = variation->position;
    variation = variation.parent();
    ++iter;
  }

  return std::distance(positions.rbegin(), iter);
}

// TODO Move this too, maybe
inline int GetPositionRepetitionCount(Variation variation) {
  if (variation->position.GetRule50Ply() < 4) return 0;
  // TODO iterating to the parent touches ref counters back and forth.
  // If this shows up in profiles, optimize by going by raw pointers.
  auto skip = [](Variation node, size_t count) {
    for (; count > 0 && node; --count) node = node.parent();
    return node;
  };
  int num_reps = 0;
  for (Variation node = skip(variation, 4); node; node = skip(node, 2)) {
    if (node->position.GetBoard() == variation->position.GetBoard()) ++num_reps;
    if (node->position.GetRule50Ply() < 2) break;
  }
  return num_reps;
};

}  // namespace lc3
}  // namespace lczero