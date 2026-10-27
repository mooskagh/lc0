#pragma once

#include <absl/container/flat_hash_map.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "chess/callbacks.h"
#include "chess/uciloop.h"
#include "neural/backend.h"
#include "search/lc5/graph.h"
#include "search/lc5/metrics.h"
#include "search/lc5/node_store.h"
#include "search/lc5/time_manager.h"
#include "search/lc5/visit.h"
#include "search/lc5/work_queue.h"

namespace lczero::lc5 {

// One `go` command's search over a shared GameGraph.
//
// Visit workers own their visits from admission to backup or cancellation.
// A visit that reaches an unexpanded node suspends on a materialization
// ticket; the ticket's owner submits the node to the store and/or evaluator
// as part of a whole I/O job and later publishes the expansion, resuming all
// waiters through their owners' mailboxes. The controller thread enforces
// limits, reports info and coordinates shutdown.
class SearchRun {
 public:
  SearchRun(GameGraph* graph, NodeStore* store, Backend* backend,
            UciResponder* responder, Settings::Resolved settings,
            TimeManager::Config time_management, VisitOrigin root,
            GoParams go_params,
            std::chrono::steady_clock::time_point start_time);
  ~SearchRun();

  void Start();
  void Stop();
  void Abort();
  void Wait();
  bool Finished() const { return finished_.load(std::memory_order_acquire); }
  const Metrics& metrics() const { return metrics_; }

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
      kBackup,  // Materialization finished; back up `value`.
    };
    VisitId id;
    size_t worker;
    Kind kind = Kind::kAdmit;
    MaterializationTicketId ticket = 0;
    uint64_t generation = 0;
    SearchValue value;
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
      kStoreHit,   // `payload` was loaded from the store.
      kStoreMiss,  // Not in the store; needs evaluation.
      kQueued,     // Added to a computation that has not run yet.
      kEvaluated,  // `payload` holds the evaluation.
    };
    MaterializationTicketId ticket;
    NodeKey key;
    PositionHistory history;
    ExpansionPayload payload;
    State state = State::kPending;
  };
  enum class JobKind { kEval, kLoad, kPersist };
  // Storage is fixed before AddInput; the whole job lives through compute
  // and completion consumption, including immediate-cache results.
  struct Job {
    JobKind kind;
    size_t worker;
    std::chrono::steady_clock::time_point queued_at;
    std::vector<Request> requests;
    std::vector<StoredExpansion> entries;
    size_t items() const { return requests.size() + entries.size(); }
  };
  struct Mailbox {
    template <typename F>
    void Post(F&& mutate) {
      {
        std::lock_guard lock(mutex);
        mutate();
      }
      cv.notify_one();
    }
    // Makes the owner re-evaluate its wait predicate.
    void Wake() {
      Post([] {});
    }

    std::mutex mutex;
    std::condition_variable cv;
    std::vector<Continuation> continuations;
    std::vector<std::unique_ptr<Job>> completions;
    bool admission_ready = false;
  };
  struct Worker {
    explicit Worker(size_t index) : index(index) {}
    bool HasQueuedIo() const {
      return !unprepared.empty() || !loads.empty() || !evals.empty() ||
             !persistence.empty();
    }

    const size_t index;
    Mailbox mailbox;
    // All fields below are touched only by this worker.
    std::deque<VisitId> runnable;
    std::unordered_set<uint64_t> active;
    std::deque<Request> unprepared;  // Await terminal detection.
    std::deque<Request> loads;
    std::deque<Request> evals;
    std::deque<StoredExpansion> persistence;
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
    // Protected by admission_mutex_, not worker-owned.
    bool admission_waiting = false;
  };
  struct Batch {
    std::unique_ptr<BackendComputation> computation;
    std::vector<std::unique_ptr<Job>> jobs;
    bool timed_out = false;
  };
  // Cap on whole jobs in flight per worker; items are capped separately at
  // two minibatches.
  static constexpr size_t kBatchCredits = 16;
  static constexpr size_t kWorkChunk = 32;

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
  std::vector<Move> BuildPv(std::optional<NodeState> root) const;
  Move FallbackMove() const;

  // Admission (search_visits.cc).
  bool AdmissionClosed() const;  // Requires admission_mutex_.
  std::optional<VisitId> Allocate(const VisitOrigin& origin, size_t worker);
  void AdmitMore(Worker& worker);
  void WakeAdmissionWaiters();

  // Visit workers (search_visits.cc).
  void VisitWorker(Worker& worker);
  void ProcessMailbox(Worker& worker);
  void AdvanceRunnable(Worker& worker, size_t limit);
  void CancelVisits(Worker& worker);
  void PublishWorkerState(Worker& worker);
  bool HasLocalWork(const Worker& worker) const;
  bool PersistenceBacklogged(const Worker& worker) const;
  void AdvanceVisit(Worker& worker, VisitId id);
  void Backup(Worker& worker, VisitPool::Slot& slot);
  void Cancel(Worker& worker, VisitPool::Slot& slot);
  void FinishVisit(Worker& worker, VisitPool::Slot& slot, bool completed);
  void Resume(Worker& worker, const Continuation& continuation);
  void Deliver(Worker& worker, const Continuation& continuation);
  void SendContinuations(size_t worker, std::vector<Continuation>& batch);

  // Materialization and I/O (search_materialization.cc).
  bool SuspendForMaterialization(Worker& worker, VisitPool::Slot& slot);
  void PrepareEvaluation(Worker& worker, Request request);
  void CompleteMaterialization(Worker& worker, MaterializationTicketId ticket,
                               ExpansionPayload payload, bool store_payload);
  void CancelMaterialization(MaterializationTicketId ticket);
  void CancelPendingRequests(Worker& worker);
  void SubmitJobs(Worker& worker);
  std::unique_ptr<Job> TakeJob(Worker& worker) const;
  bool HasJobCredit(const Worker& worker) const;
  void SetProducing(Worker& worker, bool producing);
  void ConsumeJob(Worker& worker, std::unique_ptr<Job> job);
  void ReturnJob(std::unique_ptr<Job> job);
  void StoreWorker();
  void EvaluatorWorker();
  Batch CollectBatch(std::unique_ptr<Job> first, std::unique_ptr<Job>* next);
  void AddInputs(BackendComputation& computation, Job& job);
  void RunBatch(Batch batch);
  bool EvaluationStarved() const;

  GameGraph* const graph_;
  NodeStore* const store_;
  Backend* const backend_;
  UciResponder* const responder_;
  const Settings::Resolved settings_;
  const VisitOrigin root_;
  const std::chrono::steady_clock::time_point start_time_;
  const TimeManager time_manager_;
  VisitPool visits_;
  Metrics metrics_;

  std::vector<std::unique_ptr<Worker>> workers_;
  WorkQueue<std::unique_ptr<Job>> store_jobs_;
  WorkQueue<std::unique_ptr<Job>> eval_jobs_;
  std::vector<std::thread> visit_threads_;
  std::vector<std::thread> store_threads_;
  std::vector<std::thread> evaluator_threads_;
  std::thread controller_thread_;

  std::mutex tickets_mutex_;
  absl::flat_hash_map<NodeKey, MaterializationTicketId> ticket_by_key_;
  std::unordered_map<MaterializationTicketId, Ticket> tickets_;
  MaterializationTicketId next_ticket_ = 1;

  // Serializes admission, stop requests and NN authorization.
  std::mutex admission_mutex_;
  uint64_t admitted_ = 0;
  size_t next_owner_ = 0;
  size_t next_admission_waiter_ = 0;
  std::optional<uint64_t> node_limit_;

  std::atomic<StopMode> stop_mode_{StopMode::kRunning};
  std::atomic<bool> started_{false};
  std::atomic<bool> finished_{false};
  std::atomic<bool> output_committed_{false};
  std::atomic<bool> shutdown_{false};
  // Includes queued, executing, and returned-but-unconsumed jobs.
  std::atomic<size_t> jobs_in_flight_{0};
  // Jobs held by collectors (including retained whole jobs), not computing or
  // returned. If these are all remaining jobs and producers are blocked,
  // further collection depends on flushing at least one computation.
  std::atomic<size_t> collecting_jobs_{0};
  std::mutex controller_mutex_;
  std::condition_variable controller_cv_;
  uint64_t controller_generation_ = 0;  // Protected by controller_mutex_.
};

}  // namespace lczero::lc5
