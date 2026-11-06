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
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "chess/callbacks.h"
#include "chess/uciloop.h"
#include "neural/backend.h"
#include "search/lc5/metrics.h"
#include "search/lc5/session/evaluator.h"
#include "search/lc5/session/materialization_table.h"
#include "search/lc5/session/messages.h"
#include "search/lc5/session/visit_limiter.h"
#include "search/lc5/session/visit_pool.h"
#include "search/lc5/session/visit_worker.h"
#include "search/lc5/session/work_queue.h"
#include "search/lc5/settings.h"
#include "search/lc5/store/node_store.h"
#include "search/lc5/time_manager.h"
#include "search/lc5/visit_engine.h"

namespace lczero::lc5 {

// One `go` command's search: runs visits on worker threads, batches their
// leaves for the store and the network, and reports through the responder.
class SearchSession {
 public:
  SearchSession(VisitEngine* engine, Backend* backend, UciResponder* responder,
                Settings::Resolved settings,
                TimeManager::Config time_management, VisitOrigin root,
                GoParams go_params,
                std::chrono::steady_clock::time_point start_time,
                int64_t saved_ms = 0);
  ~SearchSession();

  void Start();  // Spawns the threads and returns; later calls do nothing.
  void Stop();   // Non-blocking: output bestmove, then drain.
  void Abort();  // Non-blocking: drain with no bestmove, unless already sent.
  void Wait();   // Blocks until drained and all threads are joined.
  bool IsFinished() const { return finished_.load(std::memory_order_acquire); }
  const Metrics& metrics() const { return metrics_; }
  // Time to save for the next move (TimeManager::UnusedMs at stop).
  int64_t unused_time_ms() const { return unused_time_ms_.load(); }

 private:
  friend class SearchSessionTestPeer;

  // Thread bodies; each runs until its thread exits. Start() spawns them.
  void RunControllerWorker();
  void RunStoreWorker();

  // Search: stop, output and shutdown.
  bool IsStopping() const { return limiter_.IsStopping(); }
  void RespondBestMove();
  bool IsFullyDrained();
  void JoinThreads();
  void WakeAllVisitWorkers();
  ThinkingInfo BuildThinkingInfo(bool final) const;
  Move GetFallbackMove() const;
  // Whether the graph has a node limit to keep, by evicting to the store.
  bool HasNodeLimit() const {
    return engine_->HasStore() && settings_.graph_node_limit > 0;
  }

  // Load jobs. A visit worker sends them; a store thread runs them and
  // returns them to the worker.
  void RunLoadJob(AsyncJob& job);
  void ReturnJobToWorker(std::unique_ptr<AsyncJob> job);
  // The controller's job: evict when the graph is over its node limit.
  void SendEvictionJob();
  void RunEvictionJob(const EvictionJob& job);

  // Configuration, fixed for the session (constructor only).
  VisitEngine* const engine_;
  Backend* const backend_;
  UciResponder* const responder_;
  const Settings::Resolved settings_;
  const VisitOrigin root_;
  const std::chrono::steady_clock::time_point start_time_;
  const TimeManager time_manager_;
  // Not in infinite or ponder searches: a decided root or smart pruning may
  // stop the search before its limits.
  const bool may_stop_early_;

  // Lifecycle and the controller's wakeups.
  std::atomic<bool> started_{false};
  std::atomic<bool> aborted_{false};
  std::atomic<bool> finished_{false};
  std::atomic<int64_t> unused_time_ms_{0};
  ControllerSignal controller_;

  // Visits, materializations and counters.
  VisitPool visits_;
  Metrics metrics_;
  VisitLimiter limiter_;
  MaterializationTable materializations_;
  // AsyncJobs in flight, from all workers: queued, running, or returned but
  // not finished. Evictions are not counted; no visit waits on them.
  std::atomic<size_t> jobs_in_flight_{0};
  // Eviction jobs queued or running; the controller queues more only at zero.
  std::atomic<size_t> evictions_in_flight_{0};

  // Queues, workers and threads.
  std::vector<WorkerInbox> inboxes_;  // By visit worker index.
  WorkQueue<StoreJob> store_jobs_;
  Evaluator evaluator_;
  std::vector<std::unique_ptr<VisitWorker>> workers_;
  std::vector<std::thread> visit_threads_;
  std::vector<std::thread> store_threads_;
  std::vector<std::thread> evaluator_threads_;
  std::thread controller_thread_;
};

}  // namespace lczero::lc5
