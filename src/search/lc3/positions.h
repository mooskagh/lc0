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
  NodeHash hash;
  Position position;
  size_t depth;
  size_t idx_in_parent;

  VariationNode(NodeHash hash, Position position, size_t depth,
                size_t idx_in_parent)
      : hash(hash),
        position(std::move(position)),
        depth(depth),
        idx_in_parent(idx_in_parent) {}
};

using PositionTree = Tree<VariationNode>;
using Variation = Tree<VariationNode>::node_handle;

// struct Variation {
//   NodeHash hash;
//   Position position;
//   size_t depth;
//   Variation* parent;
//   std::atomic<size_t> ref_count_;

//   Variation(NodeHash hash, Position position, size_t depth, Variation*
//   parent,
//             size_t idx_in_parent, size_t ref_count)
//       : hash(hash),
//         position(std::move(position)),
//         depth(depth),
//         parent(parent),
//         idx_in_parent(idx_in_parent),
//         ref_count_(ref_count) {}
// };

// class VariationPtr;
// class PositionTree {
//  public:
//   PositionTree(const Position& startpos);
//   VariationPtr GetRoot();
//   Variation* GetRootRaw() { return &root_; }
//   Variation* MakeVariationRaw(Variation* parent, Move move,
//                               size_t idx_in_parent /* = kNoIdxInParent */);
//   Variation* CloneRaw(Variation* var);
//   void ReleaseVariationRaw(Variation* var);

//  private:
//   Variation root_;
//   FreeList<Variation, 65536> variation_pool_;
// };

// class VariationPtr {
//  public:
//   VariationPtr() noexcept = default;
//   ~VariationPtr() noexcept { reset(); }
//   VariationPtr(const VariationPtr& other)
//       : tree_(other.tree_),
//         variation_(other.variation_ ? tree_->CloneRaw(other.variation_)
//                                     : nullptr) {}

//   VariationPtr& operator=(const VariationPtr& other) {
//     if (this != &other) {
//       Variation* new_var =
//           other.variation_ ? other.tree_->CloneRaw(other.variation_) :
//           nullptr;
//       reset();
//       tree_ = other.tree_;
//       variation_ = new_var;
//     }
//     return *this;
//   }

//   VariationPtr(VariationPtr&& other) noexcept
//       : tree_(other.tree_), variation_(other.variation_) {
//     other.tree_ = nullptr;
//     other.variation_ = nullptr;
//   }

//   VariationPtr& operator=(VariationPtr&& other) noexcept {
//     if (this != &other) {
//       reset();  // Release current resource
//       tree_ = other.tree_;
//       variation_ = other.variation_;
//       other.tree_ = nullptr;
//       other.variation_ = nullptr;
//     }
//     return *this;
//   }

//   Variation& operator*() const noexcept { return *variation_; }
//   Variation* operator->() const noexcept { return variation_; }
//   Variation* get() const noexcept { return variation_; }
//   explicit operator bool() const noexcept { return variation_ != nullptr; }

//   void reset() noexcept {
//     if (variation_) tree_->ReleaseVariationRaw(variation_);
//     tree_ = nullptr;
//     variation_ = nullptr;
//   }

//   void swap(VariationPtr& other) noexcept {
//     using std::swap;
//     swap(tree_, other.tree_);
//     swap(variation_, other.variation_);
//   }

//   VariationPtr AddMove(Move move, size_t idx_in_parent /* = kNoIdxInParent
//   */) {
//     Variation* new_var =
//         tree_->MakeVariationRaw(variation_, std::move(move), idx_in_parent);
//     return VariationPtr(tree_, new_var);
//   }

//  private:
//   VariationPtr(PositionTree* tree, Variation* var) noexcept
//       : tree_(tree), variation_(var) {}
//   PositionTree* tree_ = nullptr;
//   Variation* variation_ = nullptr;
//   friend class PositionTree;
// };

// inline void swap(VariationPtr& lhs, VariationPtr& rhs) noexcept {
//   lhs.swap(rhs);
// }

// inline VariationPtr PositionTree::GetRoot() {
//   return VariationPtr(this, &root_);
// }

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