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
#include "search/lc5/search_internal.h"
#include "search/lc5/time_manager.h"
#include "search/lc5/visit.h"

namespace lczero::lc5 {

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

  // Arbitrary-origin admission shares the exact node budget with root visits.
  // It may be used before Start(); false means closed, budget exhausted or
  // full.
  bool Admit(const VisitOrigin& origin);

 private:
  friend class SearchRunTestPeer;
  enum class StopMode : uint8_t { kRunning, kRespondBestmove, kAbort };
  struct Continuation {
    VisitId id;
    size_t worker;
    MaterializationTicketId ticket = 0;
    uint64_t generation = 0;
    bool backup = false;
    SearchValue value;
    bool terminal = false;
  };
  struct Ticket {
    MaterializationTicketId id;
    NodeKey key;
    Continuation owner;
    std::vector<Continuation> waiters;
  };
  struct Request {
    MaterializationTicketId ticket;
    NodeKey key;
    PositionHistory history;
    ExpansionPayload payload;
    std::optional<ExpansionPayload> loaded;
    bool load_completed = false;
    enum class Evaluation { kInvalid, kQueued, kValid };
    Evaluation evaluation = Evaluation::kInvalid;
  };
  enum class JobKind { kEval, kLoad, kPersist };
  struct Job {
    JobKind kind;
    size_t worker;
    std::chrono::steady_clock::time_point queued_at;
    // Storage is fixed before AddInput; the whole job lives through compute
    // and completion consumption, including immediate-cache results.
    std::vector<Request> requests;
    std::vector<StoredExpansion> entries;
  };
  struct Mailbox {
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<Continuation> continuations;
    std::vector<std::unique_ptr<Job>> completions;
    bool admission_ready = false;
  };
  struct Worker {
    explicit Worker(size_t index) : index(index) {}
    const size_t index;
    Mailbox mailbox;
    // All fields below are touched only by this worker.
    std::deque<VisitId> runnable;
    std::unordered_set<uint64_t> active;
    std::deque<Request> loads;
    std::deque<Request> evals;
    std::deque<StoredExpansion> persistence;
    std::vector<std::vector<Continuation>> outgoing;
    std::atomic<bool> drained{false};
    // Conservative signal: this worker can submit without a collected job's
    // completion. Initially true until the first work chunk is inspected.
    std::atomic<bool> producing{true};
    size_t outstanding = 0;
    size_t outstanding_items = 0;
    bool cancelled = false;
    bool released_capacity = false;
    bool controller_event = false;
    // Protected by admission_mutex_, not worker-owned.
    bool admission_waiting = false;
  };
  // Two minibatches of items, with a separate cap for fragmented whole jobs.
  static constexpr size_t kBatchCredits = 16;
  static constexpr size_t kWorkChunk = 32;

  void VisitWorker(Worker& worker);
  void StoreWorker();
  void EvaluatorWorker();
  void Controller();
  void AdvanceVisit(Worker& worker, VisitId id);
  bool SuspendForMaterialization(Worker& worker, VisitPool::Slot& slot);
  void Backup(Worker& worker, VisitPool::Slot& slot);
  void Cancel(Worker& worker, VisitPool::Slot& slot);
  void FinishVisit(Worker& worker, VisitPool::Slot& slot, bool completed);
  std::optional<VisitId> Allocate(const VisitOrigin& origin, size_t worker);
  void AdmitMore(Worker& worker);
  void WakeAdmissionWaiters();
  void NotifyController();
  void Resume(Worker& worker, const Continuation& continuation);
  void SendContinuations(std::vector<std::vector<Continuation>> groups);
  void ReturnJob(std::unique_ptr<Job> job);
  void ConsumeJob(Worker& worker, std::unique_ptr<Job> job);
  void SubmitJobs(Worker& worker);
  bool HasJobCredit(const Worker& worker) const;
  void SetProducing(Worker& worker, bool producing);
  bool EvaluationStarved() const;
  void PrepareEvaluation(Worker& worker, Request request);
  ExpansionPayload DetectTerminal(const PositionHistory& history) const;
  void CompleteMaterialization(Worker& worker, MaterializationTicketId ticket,
                               ExpansionPayload payload, bool store_payload);
  void CancelMaterialization(MaterializationTicketId ticket);
  void PublishEvaluation(Worker& worker, Request& request);
  ThinkingInfo BuildInfo(bool final) const;
  std::vector<Move> BuildPv(std::optional<NodeSnapshot> root) const;
  Move FallbackMove() const;
  void RequestStop(StopMode mode);

  GameGraph* graph_;
  NodeStore* store_;
  Backend* backend_;
  UciResponder* responder_;
  const Settings::Resolved settings_;
  const VisitOrigin root_;
  const std::chrono::steady_clock::time_point start_time_;
  const TimeManager time_manager_;
  VisitPool visits_;
  Metrics metrics_;

  std::vector<std::unique_ptr<Worker>> workers_;
  detail::WorkQueue<std::unique_ptr<Job>> store_jobs_;
  detail::WorkQueue<std::unique_ptr<Job>> eval_jobs_;
  std::vector<std::thread> visit_threads_;
  std::vector<std::thread> store_threads_;
  std::vector<std::thread> evaluator_threads_;
  std::thread controller_thread_;

  std::mutex tickets_mutex_;
  absl::flat_hash_map<NodeKey, MaterializationTicketId> ticket_by_key_;
  std::unordered_map<MaterializationTicketId, Ticket> tickets_;
  uint64_t next_ticket_ = 1;

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
