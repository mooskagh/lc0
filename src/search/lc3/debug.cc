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

namespace {
DebugNodeData DebugNodeDataFromStorage(const NodeHandle& node_view) {
  NodeHandle::NodeAggregates node_aggregates = node_view.GetNodeAggregates();
  DebugNodeData result{
      .n = node_aggregates.n,
      .agg_v = node_aggregates.agg_v,
      .agg_d = node_aggregates.agg_d,
      .agg_m = node_aggregates.agg_m,
      .edges = {},
  };
  const size_t num_moves = node_view.FetchMoveCounts().with_visits;
  if (num_moves == 0) return result;

  std::vector<Move> moves(num_moves);
  std::vector<float> p(num_moves), q(num_moves);
  std::vector<uint64_t> n(num_moves);

  node_view.FetchEdges({moves, p, q, n});

  result.edges.reserve(num_moves);
  for (size_t i = 0; i < num_moves; ++i) {
    result.edges.emplace_back(moves[i], p[i], q[i], n[i]);
  }

  return result;
}
}  // namespace

void PrintNodeTree(NodeRepository& node_repository, std::ostream& os,
                   const Position& pos, const NodeKey& root, int indent) {
  NodeHandle node_handle =
      node_repository.GetNodeForUpdate(root, /*create_if_missing=*/false);
  if (!node_handle) {
    os << "(nil)\n";
    return;
  }
  DebugNodeData node = DebugNodeDataFromStorage(node_handle);
  os << "AV:" << node.agg_v << " AD:" << node.agg_d << " AM:" << node.agg_m
     << " N:" << node.n << " (" << pos.DebugString() << ")\n";

  for (const auto& edge : node.edges) {
    if (edge.n == 0) continue;  // Skip edges with no visits.
    for (int i = 0; i < indent; ++i) {
      os << "│ ";  // Indentation for child nodes.
    }
    os << edge.move.ToString(false) << " P:" << edge.p << " Q:" << edge.q
       << " N:" << edge.n << " --> ";
    Position child_pos(pos, edge.move);
    PrintNodeTree(node_repository, os, child_pos,
                  NodeKey{HashCat(root.hash, edge.move.raw_data())},
                  indent + 1);
  }
}

}  // namespace lc3
}  // namespace lczero