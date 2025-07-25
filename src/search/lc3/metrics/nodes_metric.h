#pragma once

#include <absl/strings/str_format.h>

#include <cstddef>
#include <cstring>

#include "utils/metrics/printer.h"

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

  size_t num_nodes_touched = 0;
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
    num_nodes_touched += other.num_nodes_touched;
    num_collision_events += other.num_collision_events;
    num_collision_visits += other.num_collision_visits;
    num_known_terminal_nodes += other.num_known_terminal_nodes;
    num_known_terminal_visits += other.num_known_terminal_visits;
    num_nodes_sent_for_eval += other.num_nodes_sent_for_eval;
    num_visits_sent_for_eval += other.num_visits_sent_for_eval;
  }

  void Print(MetricPrinter& printer) const {
    printer.StartGroup("GatherNodesMetrics");
    printer.Print("num_gather_iterations", num_gather_iterations);
    printer.Print("num_gather_depth_iterations", num_gather_depth_iterations);
    printer.Print("num_visits_spawned", num_visits_spawned);
    printer.Print("num_nodes_touched", num_nodes_touched);
    printer.Print("num_collision_events", num_collision_events);
    printer.Print("num_collision_visits", num_collision_visits);
    printer.Print("num_known_terminal_nodes", num_known_terminal_nodes);
    printer.Print("num_known_terminal_visits", num_known_terminal_visits);
    printer.Print("num_nodes_sent_for_eval", num_nodes_sent_for_eval);
    printer.Print("num_visits_sent_for_eval", num_visits_sent_for_eval);
    printer.EndGroup();
  }
};

struct EvalNodesMetrics {
  // Filled by the EvalWorker.
  size_t num_cache_hit_nodes = 0;
  size_t num_cache_hit_visits = 0;
  size_t num_discovered_checkmate_nodes = 0;
  size_t num_discovered_checkmate_visits = 0;
  size_t num_discovered_stalemate_nodes = 0;
  size_t num_discovered_stalemate_visits = 0;
  size_t num_discovered_other_draw_nodes = 0;
  size_t num_discovered_other_draw_visits = 0;
  size_t num_nn_evaluation_nodes = 0;
  size_t num_nn_evaluation_visits = 0;

  void Reset() { std::memset(this, 0, sizeof(*this)); }
  void MergeFrom(const EvalNodesMetrics& other) {
    num_cache_hit_nodes += other.num_cache_hit_nodes;
    num_cache_hit_visits += other.num_cache_hit_visits;
    num_discovered_checkmate_nodes += other.num_discovered_checkmate_nodes;
    num_discovered_checkmate_visits += other.num_discovered_checkmate_visits;
    num_discovered_stalemate_nodes += other.num_discovered_stalemate_nodes;
    num_discovered_stalemate_visits += other.num_discovered_stalemate_visits;
    num_discovered_other_draw_nodes += other.num_discovered_other_draw_nodes;
    num_discovered_other_draw_visits += other.num_discovered_other_draw_visits;
    num_nn_evaluation_nodes += other.num_nn_evaluation_nodes;
    num_nn_evaluation_visits += other.num_nn_evaluation_visits;
  }
  void Print(MetricPrinter& printer) const {
    printer.StartGroup("EvalNodesMetrics");
    printer.Print("num_cache_hit_nodes", num_cache_hit_nodes);
    printer.Print("num_cache_hit_visits", num_cache_hit_visits);
    printer.Print("num_discovered_checkmate_nodes",
                  num_discovered_checkmate_nodes);
    printer.Print("num_discovered_checkmate_visits",
                  num_discovered_checkmate_visits);
    printer.Print("num_discovered_stalemate_nodes",
                  num_discovered_stalemate_nodes);
    printer.Print("num_discovered_stalemate_visits",
                  num_discovered_stalemate_visits);
    printer.Print("num_discovered_other_draw_nodes",
                  num_discovered_other_draw_nodes);
    printer.Print("num_discovered_other_draw_visits",
                  num_discovered_other_draw_visits);
    printer.Print("num_nn_evaluation_nodes", num_nn_evaluation_nodes);
    printer.Print("num_nn_evaluation_visits", num_nn_evaluation_visits);
    printer.EndGroup();
  }
};

}  // namespace lc3
}  // namespace lczero