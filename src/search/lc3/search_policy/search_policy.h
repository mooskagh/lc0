#pragma once

#include "chess/position.h"
#include "chess/types.h"
#include "search/lc3/node_repository/node_key.h"

namespace lczero {
namespace lc3 {

struct SearchPolicy {
  static NodeKey MakeNodeKey(const NodeKey& parent_key, Move move,
                             const Position& /*new_position*/) {
    return NodeKey{HashCat(parent_key.raw_hash(), move.raw_data())};
  }
};

}  // namespace lc3
}  // namespace lczero
