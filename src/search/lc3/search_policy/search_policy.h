#pragma once

#include <absl/container/fixed_array.h>

#include <cmath>

#include "chess/position.h"
#include "chess/types.h"
#include "search/lc3/node_repository/node_key.h"

namespace lczero {
namespace lc3 {

struct SearchPolicy {
  // Create a hash key given a position.
  // Currently, it uses parent key and move, so the table is a tree.
  // For DAG, we'd hash `Position` instead.
  static NodeKey MakeNodeKey(const NodeKey& parent_key, Move move,
                             const Position& /*new_position*/) {
    return NodeKey{HashCat(parent_key.raw_hash(), move.raw_data())};
  }

  // Number of edges we'll consider distributing visits to for the node.
  // It may be slower to fetch all edges, so we limit the number of edges to
  // fetch.
  static size_t GetNumEdgesToFetch(size_t total_moves, size_t moves_with_visits,
                                   size_t /* depth */,
                                   size_t /* num_visits_to_distribute */) {
    return std::min(total_moves, moves_with_visits + 2);
  }

  // The function distributes a given number of visits to edges.
  // Currently, the following approximation is used:
  // * Compute Q + U for just one visit.
  // * Route `kBatchIterationFraction` of available visits to that edge.
  // * Repeat until all visits are distributed.
  static absl::FixedArray<size_t> DistributeVisits(
      size_t /* depth */, size_t visits_to_distribute, size_t node_n,
      std::span<const float> edge_P, std::span<const float> edge_Q,
      std::span<const uint64_t> edge_N) {
    constexpr float kCpuctConst = 1.745f;  // TODO: Make this configurable
    constexpr float kBatchIterationFraction =
        0.4f;  // TODO: Make this configurable

    assert(edge_P.size() > 0);
    assert(edge_P.size() == edge_Q.size());
    assert(edge_P.size() == edge_N.size());
    // If there is only one edge, we just return all visits to it.
    if (edge_P.size() == 1) return {visits_to_distribute};

    absl::FixedArray<size_t> result(edge_P.size(), 0);

    // parent_n_sqrt × kCpuctConst
    const float factor = std::sqrt(static_cast<float>(node_n)) * kCpuctConst;
    auto q_plus_u = [&](size_t idx) {
      return edge_Q[idx] +
             factor * edge_P[idx] / (1.0f + edge_N[idx] + result[idx]);
    };

    while (visits_to_distribute > 0) {
      const size_t visits_this_step = static_cast<size_t>(
          std::ceil(visits_to_distribute * kBatchIterationFraction));

      float best_score = q_plus_u(0);
      size_t best_idx = 0;
      for (size_t i = 1; i < edge_P.size(); ++i) {
        const float cur_score = q_plus_u(i);
        if (cur_score > best_score) {
          best_score = cur_score;
          best_idx = i;
        }
      }

      result[best_idx] += visits_this_step;
      visits_to_distribute -= visits_this_step;
    }
    return result;
  }
};

}  // namespace lc3
}  // namespace lczero
