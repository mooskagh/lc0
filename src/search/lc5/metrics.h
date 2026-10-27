/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

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
  std::atomic<uint64_t> visits_restarted{0};
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
  std::atomic<uint64_t> graph_size_high_water{0};
  std::atomic<uint64_t> eval_requests{0};
  std::atomic<uint64_t> cache_hits{0};
  std::atomic<uint64_t> nn_evaluations{0};
  std::atomic<uint64_t> evaluation_batches{0};
  std::atomic<uint64_t> partial_starvation_flushes{0};
  std::atomic<uint64_t> partial_timeout_flushes{0};
  std::atomic<uint64_t> max_depth{0};
  std::atomic<uint64_t> max_selected_depth{0};
  std::atomic<uint64_t> ready_eval_high_water{0};
  std::atomic<uint64_t> active_visits_high_water{0};
  std::atomic<uint64_t> mailbox_notifications{0};
  std::atomic<uint64_t> io_jobs_high_water{0};
  std::atomic<uint64_t> publications_rejected{0};

  std::string Format(size_t graph_size, size_t active, size_t ready_eval) const;
};

template <typename T, typename U>
void RaiseHighWater(std::atomic<T>& target, U value) {
  T old = target.load(std::memory_order_relaxed);
  while (old < static_cast<T>(value) &&
         !target.compare_exchange_weak(old, static_cast<T>(value),
                                       std::memory_order_relaxed)) {
  }
}

}  // namespace lczero::lc5
