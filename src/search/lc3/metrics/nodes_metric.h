#pragma once

#include <cstddef>
#include <cstring>

namespace lczero {
namespace lc3 {

struct GatherNodesMetrics {
  // Filled by the GatherWorker.
  // Number of gather instances (i.e. starting from the root node).
  size_t num_gather_iterations = 0;
  // Number of gather iterations, one per depth.
  size_t num_gather_depth_iterations = 0;
  // Number of visits spawned from the root node.
  size_t num_visits_spawned = 0;
  // Number of nodes that were routed as collision.
  size_t num_collision_events = 0;
  // Number of visits (may be many per node) that were routed as collision.
  size_t num_collision_visits = 0;
  // Number of nodes that were routed as terminal.
  size_t num_known_terminal_nodes = 0;
  // Number of visits (may be many per node) that were routed as terminal.
  size_t num_known_terminal_visits = 0;
  // Number of nodes sent for evaluation.
  size_t num_nodes_sent_for_eval = 0;
  // Number of visits sent for evaluation.
  size_t num_visits_sent_for_eval = 0;

  void Reset() { std::memset(this, 0, sizeof(*this)); }
  void MergeFrom(const GatherNodesMetrics& other) {
    num_gather_iterations += other.num_gather_iterations;
    num_gather_depth_iterations += other.num_gather_depth_iterations;
    num_visits_spawned += other.num_visits_spawned;
    num_collision_events += other.num_collision_events;
    num_collision_visits += other.num_collision_visits;
    num_known_terminal_nodes += other.num_known_terminal_nodes;
    num_known_terminal_visits += other.num_known_terminal_visits;
    num_nodes_sent_for_eval += other.num_nodes_sent_for_eval;
    num_visits_sent_for_eval += other.num_visits_sent_for_eval;
  }
};

// struct EvalNodesMetrics {
//   // Filled by the EvalWorker.
//   size_t num_cache_hit_nodes = 0;
//   size_t num_discovered_terminal_nodes = 0;
//   size_t num_nn_evaluation_nodes = 0;
// };

}  // namespace lc3
}  // namespace lczero