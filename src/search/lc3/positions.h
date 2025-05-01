#pragma once

#include <vector>

#include "chess/gamestate.h"
#include "chess/position.h"
#include "search/lc3/storage.h"
#include "utils/freelist.h"

namespace lczero {
namespace lc3 {

constexpr size_t kNoIdxInParent = static_cast<size_t>(-1);

struct Variation {
  NodeHash hash;
  Position position;
  size_t depth;
  Variation* parent;
  size_t idx_in_parent;
  std::atomic<size_t> ref_count_;

  Variation(NodeHash hash, Position position, size_t depth, Variation* parent,
            size_t idx_in_parent, size_t ref_count)
      : hash(hash),
        position(std::move(position)),
        depth(depth),
        parent(parent),
        idx_in_parent(idx_in_parent),
        ref_count_(ref_count) {}
};

class PositionTree {
 public:
  PositionTree(const Position& startpos);
  Variation* GetRoot() { return &root_; }
  Variation* MakeVariation(Variation* parent, Move move,
                           size_t idx_in_parent /* = kNoIdxInParent */);
  Variation* Clone(Variation* var);
  void ReleaseVariation(Variation* var);

 private:
  Variation root_;
  FreeList<Variation, 65536> variation_pool_;
};

// TODO Move this function somewhere else.
[[nodiscard]] inline size_t UnpackPositionsBackwards(
    const Variation* variation, std::span<Position> positions) {
  auto iter = positions.rbegin();
  const auto end = positions.rend();

  while (iter != end && variation != nullptr) {
    *iter = variation->position;
    variation = variation->parent;
    ++iter;
  }

  return std::distance(positions.rbegin(), iter);
}

// TODO Move this too, maybe
inline int GetPositionRepetitionCount(const Variation* variation) {
  if (variation->position.GetRule50Ply() < 4) return 0;
  auto skip = [](const Variation* node, size_t count) {
    for (; count > 0 && node; --count) node = node->parent;
    return node;
  };
  int num_reps = 0;
  for (const Variation* node = skip(variation, 4); node; node = skip(node, 2)) {
    if (node->position.GetBoard() == variation->position.GetBoard()) ++num_reps;
    if (node->position.GetRule50Ply() < 2) break;
  }
  return num_reps;
};

}  // namespace lc3
}  // namespace lczero