#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "chess/position.h"

namespace lczero::lc5 {

struct NodeKey {
  uint64_t hash = 0;
  auto operator<=>(const NodeKey&) const = default;

  template <typename H>
  friend H AbslHashValue(H state, const NodeKey& key) {
    return H::combine(std::move(state), key.hash);
  }
};

struct NodeKeyHash {
  constexpr size_t operator()(NodeKey key) const noexcept {
    return static_cast<size_t>(key.hash ^ (key.hash >> 32));
  }
};

inline NodeKey MakeNodeKey(const PositionHistory& history,
                           int history_key_length) {
  return NodeKey{history.HashLast(history_key_length + 1)};
}

}  // namespace lczero::lc5
