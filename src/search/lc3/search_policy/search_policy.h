#pragma once

#include <absl/container/fixed_array.h>

#include <cmath>
#include <span>

#include "chess/position.h"
#include "chess/types.h"
#include "search/lc3/node_repository/node_key.h"
#include "search/lc3/workers/node_event_queue.h"

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

  /////////////////////////////////////////////////////////////////////////////
  // Forward pass (gather).
  /////////////////////////////////////////////////////////////////////////////

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

  /////////////////////////////////////////////////////////////////////////////
  // Backward pass (backprop/backup).
  /////////////////////////////////////////////////////////////////////////////

  // The struct that holds the value that we backpropagate.
  struct ValueDelta {
    size_t num_visits;
    size_t num_visits_to_undo;
    double v;
    float d;
    float m;
    NodeHandle::CertaintyState certainty_state;
  };
  // The struct that holds the edge update that we backpropagate.
  using EdgeDelta = NodeHandle::EdgeMutation;

  // Converts a NodeEvent that we receive from the NN eval into a ValueDelta
  // that we propagate.
  static ValueDelta NodeEventToValueDelta(NodeEvent* event) {
    const size_t num_visits_to_apply = [&]() -> size_t {
      // Determine how many of total visits we will apply to this node. The rest
      // are rolled back. If it's a terminal node, we apply all visits, if it's
      // a normal node, we apply one, if it's a collision, we apply none.
      switch (event->result_type) {
        case NodeEvent::ResultType::kNormal:
          return 1;  // Apply one visit for a normal node.
        case NodeEvent::ResultType::kTerminal:
          return event->num_visits;  // Apply all visits for a terminal node.
        case NodeEvent::ResultType::kCollisionRollback:
          return 0;  // Do not apply visits for a collision rollback.
      }
      assert(false);  // Unreachable, but avoids compiler warning.
      return 0;
    }();

    return {
        .num_visits = num_visits_to_apply,
        .num_visits_to_undo = event->num_visits - num_visits_to_apply,
        .v = event->v,
        .d = event->d,
        .m = event->m,
        .certainty_state =
            event->result_type == NodeEvent::ResultType::kTerminal
                ? NodeHandle::CertaintyState::kTerminal
                : NodeHandle::CertaintyState::kNonTerminal,
    };
  }

  static void MergeNodeUpdates(ValueDelta* dst, const ValueDelta& src) {
    dst->num_visits_to_undo += src.num_visits_to_undo;
    if (src.num_visits == 0) return;

    // dst v, d and q are weighted averages of v, d, q, weighted by
    // num_visits_to_apply.
    const double total_visits = dst->num_visits + src.num_visits;
    const double weight = static_cast<double>(src.num_visits) / total_visits;
    dst->v += (src.v - dst->v) * weight;
    dst->d += (src.d - dst->d) * weight;
    dst->m += (src.m - dst->m) * weight;
    dst->num_visits += src.num_visits;
  }

  // Transforms the ValueDelta for the parent node.
  static void MoveNodeUpdateToParent(ValueDelta* value_delta) {
    value_delta->v = -value_delta->v;  // Negate v for backprop as it's a
                                       // opponent's perspective.
    value_delta->m += 1;  // Increment "moves left" for a parent node.
  }

  // value_delta is the value that we backpropagate.
  // node_value, if provided, is updated node value after value_delta was just
  // applied.
  static EdgeDelta MakeEdgeDelta(size_t idx_in_parent,
                                 const ValueDelta& value_delta,
                                 const NodeHandle::NodeAggregates* node_value) {
    // Computes Q (to use in Q+U) from the node value.
    auto compute_q = [](const NodeHandle::NodeAggregates& node_value) -> float {
      // The value is for the parent node, so we negate it.
      return -node_value.agg_v;
    };
    return {
        .edge_idx = idx_in_parent,
        .visits_delta = static_cast<int64_t>(-value_delta.num_visits_to_undo),
        .agg_q = node_value ? std::optional<float>{compute_q(*node_value)}
                            : std::nullopt,
    };
  }

  static bool UpdateNodeAggregate(NodeHandle::NodeAggregates* dst,
                                  const ValueDelta& src) {
    if (src.num_visits <= 0) return false;

    // Calculate the weight for the new data
    float weight =
        static_cast<float>(src.num_visits) / (dst->n + src.num_visits);

    dst->n += src.num_visits;
    dst->agg_v += weight * (src.v - dst->agg_v);
    dst->agg_d += weight * (src.d - dst->agg_d);
    dst->agg_m += weight * (src.m - dst->agg_m);
    // TODO probably with certainty propagation we'll need something smarter
    // here.
    dst->state = src.certainty_state;
    return true;
  }
};

}  // namespace lc3
}  // namespace lczero
