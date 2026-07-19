#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

#include <absl/container/flat_hash_map.h>

#include "chess/callbacks.h"
#include "chess/uciloop.h"
#include "neural/backend.h"
#include "search/lc5/graph.h"
#include "search/lc5/metrics.h"
#include "search/lc5/node_store.h"
#include "search/lc5/visit.h"

namespace lczero::lc5 {

template <typename T>
class WorkQueue {
 public:
  bool Push(T value) {
    std::lock_guard lock(mutex_);
    if (closed_) return false;
    queue_.push_back(std::move(value));
    size_.store(queue_.size(), std::memory_order_relaxed);
    cv_.notify_one();
    return true;
  }
  bool Pop(T* value) {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [&] { return closed_ || !queue_.empty(); });
    if (queue_.empty()) return false;
    *value = std::move(queue_.front());
    queue_.pop_front();
    size_.store(queue_.size(), std::memory_order_relaxed);
    return true;
  }
  bool TryPop(T* value) {
    std::lock_guard lock(mutex_);
    if (queue_.empty()) return false;
    *value = std::move(queue_.front());
    queue_.pop_front();
    size_.store(queue_.size(), std::memory_order_relaxed);
    return true;
  }
  void Close() {
    std::lock_guard lock(mutex_);
    closed_ = true;
    cv_.notify_all();
  }
  void WakeAll() { cv_.notify_all(); }
  size_t Size() const { return size_.load(std::memory_order_relaxed); }
  bool Empty() const { return Size() == 0; }

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<T> queue_;
  std::atomic<size_t> size_{0};
  bool closed_ = false;
};

class SearchRun {
 public:
  SearchRun(GameGraph* graph, NodeStore* store, Backend* backend,
            UciResponder* responder, Settings::Resolved settings,
            VisitOrigin root, GoParams go_params,
            std::chrono::steady_clock::time_point start_time);
  ~SearchRun();

  void Start();
  void Stop();
  void Abort();
  void Wait();
  bool Finished() const { return finished_.load(std::memory_order_acquire); }
  const Metrics& metrics() const { return metrics_; }

  // Internal arbitrary-origin primitive. Returns false when admission is
  // closed or the fixed VisitPool is full.
  bool Admit(const VisitOrigin& origin);

 private:
  enum class StopMode : uint8_t { kRunning, kRespondBestmove, kAbort };
  enum class TicketState : uint8_t {
    kLoadingStore,
    kWaitingForEval,
    kEvaluating,
    kCompleting,
  };
  struct Ticket {
    MaterializationTicketId id = 0;
    NodeKey key;
    TicketState state = TicketState::kLoadingStore;
    VisitId owner;
    std::vector<VisitId> waiters;
    PositionHistory owner_history;
    std::chrono::steady_clock::time_point created_at;
  };
  struct StoreRequest {
    MaterializationTicketId ticket;
    NodeKey key;
  };
  struct EvalRequest {
    MaterializationTicketId ticket;
    NodeKey key;
    PositionHistory history;
    ExpansionPayload payload;
    std::chrono::steady_clock::time_point queued_at;
  };

  void VisitWorker();
  void StoreWorker();
  void EvaluatorWorker();
  void Controller();
  void AdvanceVisit(VisitId id);
  bool SuspendForMaterialization(VisitPool::Slot& slot);
  void Backup(VisitPool::Slot& slot);
  void Cancel(VisitPool::Slot& slot);
  void FinishVisit(VisitPool::Slot& slot, bool completed);
  void AdmitMore();
  ExpansionPayload DetectTerminal(const PositionHistory& history) const;
  void CompleteMaterialization(MaterializationTicketId ticket,
                               ExpansionPayload payload, bool store_payload);
  void PublishEvaluation(std::shared_ptr<EvalRequest> request);
  void OutputInfo(bool final);
  std::vector<Move> BuildPv() const;
  Move FallbackMove() const;
  void RequestStop(StopMode mode);
  bool ShouldFlushEvaluation() const;
  void CancelAllVisits();
  void NotifyController();

  GameGraph* graph_;
  NodeStore* store_;
  Backend* backend_;
  UciResponder* responder_;
  const Settings::Resolved settings_;
  const VisitOrigin root_;
  const GoParams go_params_;
  const std::chrono::steady_clock::time_point start_time_;
  VisitPool visits_;
  Metrics metrics_;

  WorkQueue<VisitId> ready_visits_;
  WorkQueue<StoreRequest> store_requests_;
  WorkQueue<std::shared_ptr<EvalRequest>> ready_evals_;
  std::vector<std::thread> visit_threads_;
  std::vector<std::thread> store_threads_;
  std::vector<std::thread> evaluator_threads_;
  std::thread controller_thread_;

  mutable std::mutex tickets_mutex_;
  absl::flat_hash_map<NodeKey, MaterializationTicketId> ticket_by_key_;
  std::unordered_map<MaterializationTicketId, Ticket> tickets_;
  std::atomic<uint64_t> next_ticket_{1};

  std::mutex admission_mutex_;
  uint64_t admitted_ = 0;
  std::optional<uint64_t> node_limit_;
  std::optional<std::chrono::steady_clock::time_point> deadline_;

  std::atomic<StopMode> stop_mode_{StopMode::kRunning};
  std::atomic<bool> started_{false};
  std::atomic<bool> finished_{false};
  std::atomic<bool> output_committed_{false};
  std::atomic<int> advancing_visits_{0};
  std::atomic<int> evaluations_in_progress_{0};
  mutable std::mutex controller_mutex_;
  std::condition_variable controller_cv_;
};

}  // namespace lczero::lc5
