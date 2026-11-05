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

#include <absl/synchronization/mutex.h>

#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <vector>

#include "chess/callbacks.h"
#include "chess/uciloop.h"
#include "neural/backend.h"
#include "search/lc5/metrics.h"
#include "search/lc5/session/messages.h"
#include "search/lc5/session/session_gate.h"
#include "search/lc5/session/ticket_table.h"
#include "search/lc5/session/visit_pool.h"
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
  using StopMode = SessionGate::StopMode;

  // A visit thread's state; defined below the class.
  struct VisitWorker;

  // Thread bodies; each runs until its thread exits. Start() spawns them.
  void RunControllerWorker();
  void RunVisitWorker(VisitWorker& worker);
  void RunEvaluatorWorker();
  void RunStoreWorker();

  // Search: stop, output and shutdown.
  bool IsStopping() const { return gate_.IsStopping(); }
  void RequestStop(StopMode mode);
  void RespondBestMove();
  bool IsFullyDrained();
  void JoinThreads();
  void WakeController();
  void WakeAllWorkers();
  ThinkingInfo BuildThinkingInfo(bool final) const;
  Move GetFallbackMove() const;
  // Whether the graph has a node limit to keep, by evicting to the store.
  bool HasNodeLimit() const {
    return has_store_ && settings_.graph_node_limit > 0;
  }

  // Admission.
  // Admits new visits from the root for `worker`, up to kVisitsPerPass and
  // its share of the pool.
  void AdmitNewRootVisits(VisitWorker& worker);
  void WakeAdmissionWaiters();

  // Visit worker loop.
  // In one pass of its loop, a worker advances at most this many visits and
  // admits at most this many new ones, then checks its inbox again.
  static constexpr size_t kVisitsPerPass = 32;

  void ProcessInbox(VisitWorker& worker);
  void ResumeLocalVisits(VisitWorker& worker);
  void AdvanceRunnableVisits(VisitWorker& worker, size_t limit);
  void CancelWorkerVisits(VisitWorker& worker);
  void PublishWorkerState(VisitWorker& worker);
  bool HasLocalWork(const VisitWorker& worker) const;

  // Visit.
  void AdvanceVisit(VisitWorker& worker, VisitId id);
  void ResumeVisit(VisitWorker& worker, const VisitWakeup& wakeup);
  bool SuspendVisit(VisitWorker& worker, Visit& visit);
  void BackupVisit(VisitWorker& worker, Visit& visit);
  void CancelVisit(VisitWorker& worker, Visit& visit);
  // Releases the visit's reservations and returns it to its origin.
  void RestartVisit(Visit& visit);
  void FinishVisit(VisitWorker& worker, Visit& visit, bool completed);

  // VisitWakeup.
  void DeliverWakeup(VisitWorker& worker, const VisitWakeup& wakeup);
  void SendWakeups(size_t worker, std::vector<VisitWakeup>& batch);

  // Materialization. SuspendVisit opens a Ticket and its Materialization.
  void PrepareMaterialization(VisitWorker& worker,
                              Materialization materialization);
  void CompleteTicket(VisitWorker& worker, MaterializationTicketId ticket,
                      ExpansionPayload payload);
  // Publishes a node restored from the store; returns false, with the ticket
  // still open, if the record was rejected.
  bool RestoreTicket(VisitWorker& worker, MaterializationTicketId ticket,
                     std::span<const uint8_t> record);
  void CancelTicket(VisitWorker& worker, MaterializationTicketId ticket);
  void CancelQueuedMaterializations(VisitWorker& worker);

  // AsyncJob.
  // Per worker, at most this many jobs sent and not yet finished...
  static constexpr size_t kMaxJobsInFlight = 16;
  // ...holding at most this many minibatches of materializations.
  static constexpr size_t kMaxItemsInFlightBatches = 2;
  // A job holds at most 1/kJobsPerBatch of a minibatch, so an evaluator can
  // mix several jobs into one batch.
  static constexpr size_t kJobsPerBatch = 4;

  // A visit thread builds and sends jobs; a store thread or an evaluator runs
  // them and returns them to the worker, which finishes them.
  void SendJobs(VisitWorker& worker);
  bool CanSendJob(const VisitWorker& worker) const;
  std::unique_ptr<AsyncJob> BuildJob(VisitWorker& worker) const;
  void SetWorkerProducing(VisitWorker& worker, bool producing);
  void RunLoadJob(AsyncJob& job);
  void ReturnJobToWorker(std::unique_ptr<AsyncJob> job);
  void FinishJob(VisitWorker& worker, std::unique_ptr<AsyncJob> job);
  // The controller's job: evict when the graph is over its node limit.
  void SendEvictionJob();
  void RunEvictionJob(const EvictionJob& job);

  // Batch.
  // Takes `first` and more queued jobs while they fit the minibatch.
  Batch CollectBatch(std::unique_ptr<AsyncJob> first);
  void AddJobToBatch(BackendComputation& computation, AsyncJob& job);
  void EvaluateBatch(Batch batch);
  bool IsEvaluationStarved() const;

  // Configuration, fixed for the session (constructor only).
  VisitEngine* const engine_;
  const bool has_store_;
  Backend* const backend_;
  UciResponder* const responder_;
  const Settings::Resolved settings_;
  const VisitOrigin root_;
  const uint64_t root_visits_at_start_;
  const std::chrono::steady_clock::time_point start_time_;
  const TimeManager time_manager_;
  // Visits to complete: `go nodes`, capped so root visits fit in uint32_t.
  const uint64_t node_limit_;
  // Not in infinite or ponder searches: a decided root or smart pruning may
  // stop the search before its limits.
  const bool may_stop_early_;

  // Visits and counters.
  VisitPool visits_;
  Metrics metrics_;
  SessionGate gate_;

  // Threads and the queues between them.
  std::vector<std::unique_ptr<VisitWorker>> workers_;
  WorkQueue<StoreJob> store_jobs_;
  WorkQueue<std::unique_ptr<AsyncJob>> eval_jobs_;
  std::vector<std::thread> visit_threads_;
  std::vector<std::thread> store_threads_;
  std::vector<std::thread> evaluator_threads_;
  std::thread controller_thread_;

  TicketTable tickets_;

  // AsyncJobs in flight, from all workers: queued, running, or returned but
  // not finished. Evictions are not counted; no visit waits on them.
  std::atomic<size_t> jobs_in_flight_{0};
  // Jobs held by collectors, not computing or returned. If these are all
  // remaining jobs and producers are blocked, further collection depends on
  // flushing at least one computation.
  std::atomic<size_t> collecting_jobs_{0};
  // Eviction jobs queued or running; the controller queues more only at zero.
  std::atomic<size_t> evictions_in_flight_{0};
  // A worker saw the graph over its limit and woke the controller.
  std::atomic<bool> eviction_requested_{false};

  // Lifecycle and the controller's wakeups.
  std::atomic<bool> started_{false};
  std::atomic<bool> shutdown_{false};
  std::atomic<bool> finished_{false};
  std::atomic<int64_t> unused_time_ms_{0};
  absl::Mutex controller_mutex_;
  uint64_t controller_generation_ ABSL_GUARDED_BY(controller_mutex_) = 0;
};

struct SearchSession::VisitWorker {
  explicit VisitWorker(size_t index) : index(index) {}
  bool HasQueuedIo() const { return !loads.empty() || !evals.empty(); }

  const size_t index;
  WorkerInbox inbox;
  // All fields below are touched only by this worker.
  std::deque<VisitId> runnable;
  size_t active_visits = 0;  // Owned visits: runnable or waiting on a ticket.
  std::deque<Materialization> loads;
  std::deque<Materialization> evals;
  // Wakeups for this worker's own visits, resumed in ProcessInbox: never
  // while AdvanceVisit may be stepping the visit.
  std::vector<VisitWakeup> local_wakeups;
  std::vector<std::vector<VisitWakeup>> outgoing;  // Indexed by worker.
  size_t jobs_in_flight = 0;                       // Sent and not yet finished.
  size_t items_in_flight = 0;  // The materializations in those jobs.
  bool cancelled = false;
  bool released_capacity = false;
  bool controller_event = false;
  std::atomic<bool> drained{false};
  // Conservative signal: this worker can send a job without a collected job's
  // completion. Initially true until the first work chunk is inspected.
  std::atomic<bool> producing{true};
};

}  // namespace lczero::lc5
