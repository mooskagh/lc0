#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace lczero::lc5 {

struct Metrics {
  std::atomic<uint64_t> visits_admitted{0};
  std::atomic<uint64_t> visits_completed{0};
  std::atomic<uint64_t> visits_cancelled{0};
  std::atomic<uint64_t> visits_ready_high_water{0};
  std::atomic<uint64_t> visits_suspended{0};
  std::atomic<uint64_t> visits_resumed{0};
  std::atomic<uint64_t> terminal_visits{0};
  std::atomic<uint64_t> selection_node_steps{0};
  std::atomic<uint64_t> backup_node_steps{0};
  std::atomic<uint64_t> stale_generation_node_updates{0};
  std::atomic<uint64_t> stale_generation_edge_updates{0};
  std::atomic<uint64_t> invariant_underflow_prevented{0};
  std::atomic<uint64_t> tickets_created{0};
  std::atomic<uint64_t> ticket_waiters{0};
  std::atomic<uint64_t> maximum_waiters_per_ticket{0};
  std::atomic<uint64_t> node_store_load_batches{0};
  std::atomic<uint64_t> node_store_load_keys{0};
  std::atomic<uint64_t> node_store_hits{0};
  std::atomic<uint64_t> node_store_misses{0};
  std::atomic<uint64_t> node_store_store_batches{0};
  std::atomic<uint64_t> graph_nodes_created{0};
  std::atomic<uint64_t> graph_nodes_rehydrated{0};
  std::atomic<uint64_t> graph_nodes_evicted{0};
  std::atomic<uint64_t> graph_size_high_water{0};
  std::atomic<uint64_t> eval_requests{0};
  std::atomic<uint64_t> cache_hits{0};
  std::atomic<uint64_t> nn_evaluations{0};
  std::atomic<uint64_t> evaluation_batches{0};
  std::atomic<uint64_t> partial_starvation_flushes{0};
  std::atomic<uint64_t> partial_timeout_flushes{0};
  std::atomic<uint64_t> partial_drain_flushes{0};
  std::atomic<uint64_t> max_depth{0};
  std::atomic<uint64_t> max_selected_depth{0};
  std::atomic<uint64_t> ready_eval_high_water{0};
  std::atomic<uint64_t> active_visits_high_water{0};

  std::string Format(size_t graph_size, size_t active, size_t ready_eval) const;
};

}  // namespace lczero::lc5
