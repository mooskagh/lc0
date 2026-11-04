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

#include <absl/container/flat_hash_map.h>
#include <absl/synchronization/mutex.h>

#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "chess/callbacks.h"
#include "chess/uciloop.h"
#include "neural/backend.h"
#include "search/lc5/metrics.h"
#include "search/lc5/session/messages.h"
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
  enum class StopMode : uint8_t { kRunning, kRespondBestmove, kAbort };

  // A visit thread's state; defined below the class.
  struct Worker;

  // Cap on whole jobs in flight per worker; items are capped separately at
  // two minibatches.
  static constexpr size_t kBatchCredits = 16;
  static constexpr size_t kWorkChunk = 32;
  static constexpr size_t kMinActiveVisits = 16;

  // Thread entry points.
  void ControllerThread();
  void VisitThread(Worker& worker);
  void EvaluatorThread();
  void StoreThread();

  // Search: stop, output and shutdown.
  bool IsStopping() const { return stop_mode_.load() != StopMode::kRunning; }
  void RequestStop(StopMode mode);
  void RespondBestMove();
  bool IsFullyDrained();
  void JoinThreads();
  void WakeController();
  void WakeAllWorkers();
  ThinkingInfo BuildThinkingInfo(bool final) const;
  Move FallbackMove() const;
  // Whether the graph has a node limit to keep, by evicting to the store.
  bool HasNodeLimit() const {
    return has_store_ && settings_.graph_node_limit > 0;
  }

  // Admission.
  bool IsAdmissionClosed() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(admission_mutex_);
  // Live visits allowed now: grows with the root's visits, as classic search
  // grows its collision limit, so small trees are not flooded.
  size_t ActiveVisitLimit() const;
  std::optional<VisitId> AllocateVisit(const VisitOrigin& origin)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(admission_mutex_);
  void AdmitVisits(Worker& worker);
  void WakeAdmissionWaiters();

  // Worker loop.
  void ProcessInbox(Worker& worker);
  void AdvanceRunnableVisits(Worker& worker, size_t limit);
  void CancelWorkerVisits(Worker& worker);
  void PublishWorkerState(Worker& worker);
  bool HasLocalWork(const Worker& worker) const;

  // Visit.
  void AdvanceVisit(Worker& worker, VisitId id);
  void ResumeVisit(Worker& worker, const Continuation& continuation);
  bool SuspendVisit(Worker& worker, VisitPool::Slot& slot)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(slot.mutex);
  void BackupVisit(Worker& worker, VisitPool::Slot& slot)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(slot.mutex);
  void CancelVisit(Worker& worker, VisitPool::Slot& slot)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(slot.mutex);
  // Releases the visit's reservations and returns it to its origin.
  void RestartVisit(VisitPool::Slot& slot)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(slot.mutex);
  void FinishVisit(Worker& worker, VisitPool::Slot& slot, bool completed)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(slot.mutex);

  // Continuation.
  void DeliverContinuation(Worker& worker, const Continuation& continuation);
  void SendContinuations(size_t worker, std::vector<Continuation>& batch);

  // Ticket.
  void CompleteTicket(Worker& worker, MaterializationTicketId ticket,
                      ExpansionPayload payload);
  // Publishes a node restored from the store; returns false, with the ticket
  // still open, if the record was rejected.
  bool RestoreTicket(Worker& worker, MaterializationTicketId ticket,
                     std::span<const uint8_t> record);
  void CancelTicket(Worker& worker, MaterializationTicketId ticket);

  // MaterializationRequest.
  void PrepareRequest(Worker& worker, MaterializationRequest request);
  void CancelPendingRequests(Worker& worker);

  // Job.
  void SubmitJobs(Worker& worker);
  std::unique_ptr<Job> TakeJob(Worker& worker) const;
  bool HasJobCredit(const Worker& worker) const;
  void SetWorkerProducing(Worker& worker, bool producing);
  void ConsumeJob(Worker& worker, std::unique_ptr<Job> job);
  void ReturnJob(std::unique_ptr<Job> job);
  // Queues eviction jobs when the graph is over its node limit.
  void QueueEvictionJobs();
  void RunEvictionJob(Job& job);

  // Batch.
  Batch CollectBatch(std::unique_ptr<Job> first, std::unique_ptr<Job>* next);
  void AddJobInputs(BackendComputation& computation, Job& job);
  void RunBatch(Batch batch);
  bool IsEvaluationStarved() const;

  // Configuration, fixed for the session (constructor only).
  VisitEngine* const engine_;
  const bool has_store_;
  Backend* const backend_;
  UciResponder* const responder_;
  const Settings::Resolved settings_;
  const VisitOrigin root_;
  uint64_t root_visits_at_start_ = 0;
  const std::chrono::steady_clock::time_point start_time_;
  const TimeManager time_manager_;
  std::optional<uint64_t> node_limit_;
  // Not in infinite or ponder searches: a decided root or smart pruning may
  // stop the search before its limits.
  bool may_stop_early_ = true;

  // Visits and counters.
  VisitPool visits_;
  Metrics metrics_;

  // Threads and the queues between them.
  std::vector<std::unique_ptr<Worker>> workers_;
  WorkQueue<std::unique_ptr<Job>> store_jobs_;
  WorkQueue<std::unique_ptr<Job>> eval_jobs_;
  std::vector<std::thread> visit_threads_;
  std::vector<std::thread> store_threads_;
  std::vector<std::thread> evaluator_threads_;
  std::thread controller_thread_;

  // Admission and stop. admission_mutex_ serializes admission, stop requests
  // and NN authorization. Also guards Worker::admission_waiting.
  absl::Mutex admission_mutex_;
  uint64_t admitted_ ABSL_GUARDED_BY(admission_mutex_) = 0;
  size_t next_owner_ ABSL_GUARDED_BY(admission_mutex_) = 0;
  size_t next_admission_waiter_ ABSL_GUARDED_BY(admission_mutex_) = 0;
  std::atomic<StopMode> stop_mode_{StopMode::kRunning};
  std::atomic<bool> output_committed_{false};

  // Tickets.
  absl::Mutex tickets_mutex_;
  absl::flat_hash_map<NodeKey, MaterializationTicketId> ticket_by_key_
      ABSL_GUARDED_BY(tickets_mutex_);
  std::unordered_map<MaterializationTicketId, Ticket> tickets_
      ABSL_GUARDED_BY(tickets_mutex_);
  MaterializationTicketId next_ticket_ ABSL_GUARDED_BY(tickets_mutex_) = 1;

  // Jobs and evictions in flight.
  // Includes queued, executing, and returned-but-unconsumed jobs.
  std::atomic<size_t> jobs_in_flight_{0};
  // Jobs held by collectors (including retained whole jobs), not computing or
  // returned. If these are all remaining jobs and producers are blocked,
  // further collection depends on flushing at least one computation.
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

struct SearchSession::Worker {
  explicit Worker(size_t index) : index(index) {}
  bool HasQueuedIo() const {
    return !unprepared.empty() || !loads.empty() || !evals.empty();
  }

  const size_t index;
  WorkerInbox inbox;
  // All fields below are touched only by this worker.
  std::deque<VisitId> runnable;
  std::unordered_set<uint64_t> active;
  std::deque<MaterializationRequest> unprepared;  // Await terminal detection.
  std::deque<MaterializationRequest> loads;
  std::deque<MaterializationRequest> evals;
  std::vector<std::vector<Continuation>> outgoing;  // Indexed by worker.
  size_t outstanding = 0;
  size_t outstanding_items = 0;
  bool cancelled = false;
  bool released_capacity = false;
  bool controller_event = false;
  std::atomic<bool> drained{false};
  // Conservative signal: this worker can submit without a collected job's
  // completion. Initially true until the first work chunk is inspected.
  std::atomic<bool> producing{true};
  // Guarded by SearchSession::admission_mutex_, not worker-owned.
  bool admission_waiting = false;
};

}  // namespace lczero::lc5
