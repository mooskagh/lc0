#pragma once

#include <compare>
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

// Keys the last position plus `history_key_length` preceding ones.
inline NodeKey MakeNodeKey(const PositionHistory& history,
                           int history_key_length) {
  return NodeKey{history.HashLast(history_key_length + 1)};
}

}  // namespace lczero::lc5
