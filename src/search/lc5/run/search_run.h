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
#include "search/lc5/run/metrics.h"
#include "search/lc5/run/work_queue.h"
#include "search/lc5/store/node_store.h"
#include "search/lc5/sync.h"
#include "search/lc5/time_manager.h"
#include "search/lc5/visit.h"
#include "search/lc5/visit_engine.h"

namespace lczero::lc5 {

// One `go` command's search, reaching node and edge state only through a
// VisitEngine.
//
// Visit workers own their visits from admission to backup or cancellation.
// A visit that reaches an unexpanded node suspends on a materialization
// ticket; the ticket's owner submits the node to the store and/or evaluator
// as part of a whole I/O job and later publishes the expansion, resuming all
// waiters through their owners' mailboxes. The controller thread enforces
// limits, reports info and coordinates shutdown. With a store and a graph
// node limit, the controller also moves nodes out of the graph into the store
// (eviction), and a graph miss restores a stored node with its statistics.
class SearchRun {
 public:
  SearchRun(VisitEngine* engine, NodeStore* store, Backend* backend,
            UciResponder* responder, Settings::Resolved settings,
            TimeManager::Config time_management, VisitOrigin root,
            GoParams go_params,
            std::chrono::steady_clock::time_point start_time,
            int64_t saved_ms = 0);
  ~SearchRun();

  void Start();
  void Stop();
  void Abort();
  void Wait();
  bool Finished() const { return finished_.load(std::memory_order_acquire); }
  const Metrics& metrics() const { return metrics_; }
  // Time to save for the next move (TimeManager::UnusedMs at stop).
  int64_t unused_time_ms() const { return unused_time_ms_.load(); }

  // Admits a visit from an arbitrary origin, sharing the node budget with
  // root visits. May be called before Start(). Returns false if admission is
  // closed, the budget is exhausted or the pool is full.
  bool Admit(const VisitOrigin& origin);

 private:
  friend class SearchRunTestPeer;
  enum class StopMode : uint8_t { kRunning, kRespondBestmove, kAbort };

  // Tells a visit's owner how to continue it.
  struct Continuation {
    enum class Kind : uint8_t {
      kAdmit,   // Newly admitted visit.
      kSelect,  // Materialization finished; select below the node.
      // The node was restored from the store with its statistics; select
      // below it without resuming through the policy, as nothing new is known.
      kRestored,
      kBackup,  // Materialization finished; back up `value`.
      // Materialization finished and the node already holds `value`; back
      // up through the steps above it.
      kBackupAbove,
      // Materialization was cancelled on stop; cancel the visit.
      kCancel,
    };
    VisitId id;
    size_t worker;
    Kind kind = Kind::kAdmit;
    MaterializationTicketId ticket = 0;
    uint64_t generation = 0;
    SearchValue value{};
    bool exact = false;  // `value` is an exact terminal value.
    bool terminal = false;
  };
  struct Ticket {
    NodeKey key;
    Continuation owner;
    std::vector<Continuation> waiters;
  };
  // One node to materialize, travelling through load and evaluation jobs.
  struct Request {
    enum class State : uint8_t {
      kPending,    // Not (yet) processed, or abandoned on stop.
      kStoreHit,   // `record` was loaded from the store.
      kStoreMiss,  // Not in the store; needs evaluation.
      kQueued,     // Added to a computation that has not run yet.
      kEvaluated,  // `payload` holds the evaluation.
    };
    MaterializationTicketId ticket;
    NodeKey key;
    PositionHistory history;
    ExpansionPayload payload{};
    std::vector<uint8_t> record{};
    State state = State::kPending;
  };
  // kPersist jobs store evicted nodes and then erase them from the graph. The
  // controller creates them, and a store thread finishes them without
  // returning them to a worker.
  enum class JobKind { kEval, kLoad, kPersist };
  // Storage is fixed before AddInput; the whole job lives through compute
  // and completion consumption, including immediate-cache results.
  struct Job {
    JobKind kind;
    size_t worker;
    std::chrono::steady_clock::time_point queued_at;
    std::vector<Request> requests;
    std::vector<EvictCandidate> evictions;
    size_t items() const { return requests.size(); }
  };
  // A worker's inbox. Senders lock `mutex` and add to it; unlocking wakes the
  // owner if its wait condition now holds.
  struct Mailbox {
    // Makes the owner re-evaluate its wait condition after external state
    // (stop, shutdown) changed.
    void Wake() { MutexLock lock(mutex); }

    absl::Mutex mutex;
    std::vector<Continuation> continuations ABSL_GUARDED_BY(mutex);
    std::vector<std::unique_ptr<Job>> completions ABSL_GUARDED_BY(mutex);
    bool admission_ready ABSL_GUARDED_BY(mutex) = false;
  };
  struct Worker {
    explicit Worker(size_t index) : index(index) {}
    bool HasQueuedIo() const {
      return !unprepared.empty() || !loads.empty() || !evals.empty();
    }

    const size_t index;
    Mailbox mailbox;
    // All fields below are touched only by this worker.
    std::deque<VisitId> runnable;
    std::unordered_set<uint64_t> active;
    std::deque<Request> unprepared;  // Await terminal detection.
    std::deque<Request> loads;
    std::deque<Request> evals;
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
    // Guarded by SearchRun::admission_mutex_, not worker-owned.
    bool admission_waiting = false;
  };
  struct Batch {
    std::unique_ptr<BackendComputation> computation;
    std::vector<std::unique_ptr<Job>> jobs{};
    bool timed_out = false;
  };
  // Cap on whole jobs in flight per worker; items are capped separately at
  // two minibatches.
  static constexpr size_t kBatchCredits = 16;
  static constexpr size_t kWorkChunk = 32;
  static constexpr size_t kMinActiveVisits = 16;
  // Nodes per eviction job.
  static constexpr size_t kEvictionBatch = 1024;

  bool Stopping() const { return stop_mode_.load() != StopMode::kRunning; }
  void RequestStop(StopMode mode);

  // Controller.
  void Controller();
  void Respond();
  bool FullyDrained();
  void Shutdown();
  void WakeAllWorkers();
  void NotifyController();
  ThinkingInfo BuildInfo(bool final) const;
  // Whether the graph has a node limit to keep, by evicting to the store.
  bool Evicts() const { return store_ && settings_.graph_node_limit > 0; }
  // Queues eviction jobs when the graph is over its node limit.
  void QueueEvictions();
  Move FallbackMove() const;

  // Admission (visits.cc).
  bool AdmissionClosed() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(admission_mutex_);
  // Live visits allowed now: grows with the root's visits, as classic search
  // grows its collision limit, so small trees are not flooded.
  size_t ActiveLimit() const;
  std::optional<VisitId> Allocate(const VisitOrigin& origin, size_t worker)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(admission_mutex_);
  void AdmitMore(Worker& worker);
  void WakeAdmissionWaiters();

  // Visit workers (visits.cc).
  void VisitWorker(Worker& worker);
  void ProcessMailbox(Worker& worker);
  void AdvanceRunnable(Worker& worker, size_t limit);
  void CancelVisits(Worker& worker);
  void PublishWorkerState(Worker& worker);
  bool HasLocalWork(const Worker& worker) const;
  void AdvanceVisit(Worker& worker, VisitId id);
  void Backup(Worker& worker, VisitPool::Slot& slot)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(slot.mutex);
  void Cancel(Worker& worker, VisitPool::Slot& slot)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(slot.mutex);
  // Releases the visit's reservations and returns it to its origin.
  void Restart(VisitPool::Slot& slot) ABSL_EXCLUSIVE_LOCKS_REQUIRED(slot.mutex);
  void FinishVisit(Worker& worker, VisitPool::Slot& slot, bool completed)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(slot.mutex);
  void Resume(Worker& worker, const Continuation& continuation);
  void Deliver(Worker& worker, const Continuation& continuation);
  void SendContinuations(size_t worker, std::vector<Continuation>& batch);

  // Materialization and I/O (materialization.cc).
  bool SuspendForMaterialization(Worker& worker, VisitPool::Slot& slot)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(slot.mutex);
  void PrepareEvaluation(Worker& worker, Request request);
  void CompleteMaterialization(Worker& worker, MaterializationTicketId ticket,
                               ExpansionPayload payload);
  // Publishes a node restored from the store; returns false, with the ticket
  // still open, if the record was rejected.
  bool CompleteRestore(Worker& worker, MaterializationTicketId ticket,
                       std::span<const uint8_t> record);
  void CancelMaterialization(Worker& worker, MaterializationTicketId ticket);
  void CancelPendingRequests(Worker& worker);
  void SubmitJobs(Worker& worker);
  std::unique_ptr<Job> TakeJob(Worker& worker) const;
  bool HasJobCredit(const Worker& worker) const;
  void SetProducing(Worker& worker, bool producing);
  void ConsumeJob(Worker& worker, std::unique_ptr<Job> job);
  void ReturnJob(std::unique_ptr<Job> job);
  void StoreWorker();
  void Evict(Job& job);
  void EvaluatorWorker();
  Batch CollectBatch(std::unique_ptr<Job> first, std::unique_ptr<Job>* next);
  void AddInputs(BackendComputation& computation, Job& job);
  void RunBatch(Batch batch);
  bool EvaluationStarved() const;

  VisitEngine* const engine_;
  NodeStore* const store_;
  Backend* const backend_;
  UciResponder* const responder_;
  const Settings::Resolved settings_;
  const VisitOrigin root_;
  uint64_t root_visits_at_start_ = 0;
  const std::chrono::steady_clock::time_point start_time_;
  const TimeManager time_manager_;
  VisitPool visits_;
  Metrics metrics_;
  const VisitContext context_{settings_, metrics_};

  std::vector<std::unique_ptr<Worker>> workers_;
  WorkQueue<std::unique_ptr<Job>> store_jobs_;
  WorkQueue<std::unique_ptr<Job>> eval_jobs_;
  std::vector<std::thread> visit_threads_;
  std::vector<std::thread> store_threads_;
  std::vector<std::thread> evaluator_threads_;
  std::thread controller_thread_;

  absl::Mutex tickets_mutex_;
  absl::flat_hash_map<NodeKey, MaterializationTicketId> ticket_by_key_
      ABSL_GUARDED_BY(tickets_mutex_);
  std::unordered_map<MaterializationTicketId, Ticket> tickets_
      ABSL_GUARDED_BY(tickets_mutex_);
  MaterializationTicketId next_ticket_ ABSL_GUARDED_BY(tickets_mutex_) = 1;

  // Set in the constructor only.
  std::optional<uint64_t> node_limit_;
  // Not in infinite or ponder searches: a decided root or smart pruning may
  // stop the search before its limits.
  bool may_stop_early_ = true;

  // Serializes admission, stop requests and NN authorization. Also guards
  // Worker::admission_waiting.
  absl::Mutex admission_mutex_;
  uint64_t admitted_ ABSL_GUARDED_BY(admission_mutex_) = 0;
  size_t next_owner_ ABSL_GUARDED_BY(admission_mutex_) = 0;
  size_t next_admission_waiter_ ABSL_GUARDED_BY(admission_mutex_) = 0;

  std::atomic<StopMode> stop_mode_{StopMode::kRunning};
  std::atomic<bool> started_{false};
  std::atomic<bool> finished_{false};
  std::atomic<bool> output_committed_{false};
  std::atomic<bool> shutdown_{false};
  std::atomic<int64_t> unused_time_ms_{0};
  // Includes queued, executing, and returned-but-unconsumed jobs.
  std::atomic<size_t> jobs_in_flight_{0};
  // Eviction jobs queued or running; the controller queues more only at zero.
  std::atomic<size_t> evictions_in_flight_{0};
  // A worker saw the graph over its limit and woke the controller.
  std::atomic<bool> eviction_requested_{false};
  // Jobs held by collectors (including retained whole jobs), not computing or
  // returned. If these are all remaining jobs and producers are blocked,
  // further collection depends on flushing at least one computation.
  std::atomic<size_t> collecting_jobs_{0};
  absl::Mutex controller_mutex_;
  uint64_t controller_generation_ ABSL_GUARDED_BY(controller_mutex_) = 0;
};

}  // namespace lczero::lc5
