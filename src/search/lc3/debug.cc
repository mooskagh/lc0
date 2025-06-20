#include "search/lc3/debug.h"

#include <absl/strings/str_cat.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "chess/position.h"

namespace lczero {
namespace lc3 {

std::string DebugEdgeData::ToString() const {
  return absl::StrCat("Move: ", move.ToString(false), ", p: ", p, ", q: ", q,
                      ", n: ", n);
}

std::vector<std::string> DebugNodeData::ToStrings() const {
  std::vector<std::string> result;
  result.reserve(edges.size() + 1);
  result.push_back(absl::StrCat("n: ", n, ", agg_v: ", agg_v,
                                ", agg_d: ", agg_d, ", agg_m: ", agg_m));
  for (const auto& edge : edges) result.push_back(edge.ToString());
  return result;
}

DebugNodeData DebugNodeDataFromStorage(const NodeView& node_view) {
  DebugNodeData result;
  node_view.FetchNodeValue({.n = &result.n,
                            .agg_v = &result.agg_v,
                            .agg_d = &result.agg_d,
                            .agg_m = &result.agg_m});

  const size_t num_moves = node_view.FetchNumMoves();
  if (num_moves == 0) return result;

  std::vector<Move> moves(num_moves);
  std::vector<float> p(num_moves), q(num_moves);
  std::vector<uint64_t> n(num_moves);

  node_view.FetchEdgeData({moves, p, q, n});

  result.edges.reserve(num_moves);
  for (size_t i = 0; i < num_moves; ++i) {
    result.edges.emplace_back(moves[i], p[i], q[i], n[i]);
  }

  return result;
}

void PrintNodeTree(AccessLock* lock, std::ostream& os, const Position& pos,
                   const NodeHash& root, int indent) {
  std::optional<NodeView> node_view = lock->FetchReadOnly(root);
  if (!node_view) {
    os << "(nil)\n";
    return;
  }
  DebugNodeData node = DebugNodeDataFromStorage(*node_view);
  node_view->FetchNodeValue({.n = &node.n,
                             .agg_v = &node.agg_v,
                             .agg_d = &node.agg_d,
                             .agg_m = &node.agg_m});
  os << "AV:" << node.agg_v << " AD:" << node.agg_d << " AM:" << node.agg_m
     << " N:" << node.n << " (" << pos.DebugString() << ")\n";

  for (const auto& edge : node.edges) {
    if (edge.n == 0) continue;  // Skip edges with no visits.
    for (int i = 0; i < indent; ++i) {
      os << "│ ";  // Indentation for child nodes.
    }
    os << edge.move.ToString(false)
       << " P:" << edge.p << " Q:" << edge.q << " N:" << edge.n << " --> ";
    Position child_pos(pos, edge.move);
    PrintNodeTree(lock, os, child_pos,
                  NodeHash{HashCat(root.hash, edge.move.raw_data())},
                  indent + 1);
  }
}

}  // namespace lc3
}  // namespace lczero