#pragma once

#include <cstddef>
#include <cstdint>

#include "chess/position.h"
#include "chess/types.h"
#include "search/lc3/node_repository.h"

namespace lczero {
namespace lc3 {

struct DebugEdgeData {
  Move move;
  float p;
  float q;
  uint64_t n;

  std::string ToString() const;
};

struct DebugNodeData {
  uint64_t n = 0;
  double agg_v;
  float agg_d;
  float agg_m;
  std::vector<DebugEdgeData> edges;

  std::vector<std::string> ToStrings() const;
};

void PrintNodeTree(NodeRepository& node_repository, std::ostream& os,
                   const Position& pos, const NodeKey& root, int indent = 0);

}  // namespace lc3
}  // namespace lczero