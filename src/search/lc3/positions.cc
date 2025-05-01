#include "search/lc3/positions.h"

#include "utils/hashcat.h"

namespace lczero {
namespace lc3 {

PositionTree::PositionTree(const Position& startpos)
    : root_{NodeHash{startpos.Hash()},
            startpos,
            /*depth=*/0,
            /*parent=*/nullptr,
            kNoIdxInParent,
            /*ref_count=*/1} {}

Variation* PositionTree::MakeVariation(Variation* parent, Move move,
                                       size_t idx_in_parent) {
  assert(parent->ref_count_.load(std::memory_order_relaxed) > 0);
  parent->ref_count_.fetch_add(1, std::memory_order_relaxed);
  Variation* new_var = variation_pool_.Allocate(
      /*hash=*/NodeHash{HashCat(parent->hash.hash, move.raw_data())},
      /*position=*/Position(parent->position, move),
      /*depth=*/parent->depth + 1,
      /*parent=*/parent,
      /*idx_in_parent=*/idx_in_parent,
      /*ref_count_=*/1);
  return new_var;
}

void PositionTree::ReleaseVariation(Variation* var) {
  assert(var->ref_count_.load(std::memory_order_relaxed) > 0);
  if (var->ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    variation_pool_.Release(var);
  }
}

Variation* PositionTree::Clone(Variation* var) {
  if (var->parent)
    var->parent->ref_count_.fetch_add(1, std::memory_order_relaxed);
  Variation* new_var = variation_pool_.Allocate(
      var->hash, var->position, var->depth, var->parent, var->idx_in_parent,
      /*ref_count_=*/1);
  return new_var;
}

}  // namespace lc3
}  // namespace lczero