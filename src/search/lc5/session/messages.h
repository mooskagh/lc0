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
#include <absl/time/time.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#include "chess/position.h"
#include "search/lc5/graph/key.h"
#include "search/lc5/graph/node.h"
#include "search/lc5/graph/node_eval.h"
#include "search/lc5/graph/value.h"
#include "search/lc5/visit.h"

namespace lczero::lc5 {

// Inter-thread messages and signaling primitives for SearchSession.

// Notification sent to a visit worker to resume or cancel a visit that was
// waiting on node materialization.
struct VisitWakeup {
  enum class Kind : uint8_t {
    kSelect,    // Materialization finished; select below the node.
    kRestored,  // Node restored from the store; select without ResumeVisit.
    kBackup,    // Materialization finished; back up `value`.
    kCancel,    // Materialization cancelled on stop; cancel the visit.
  };
  VisitId id;     // The visit to continue.
  size_t worker;  // Owning visit worker (the only thread that may mutate it).
  Kind kind = Kind::kSelect;
  VisitValue value{};     // The leaf value; unset for kRestored and kCancel.
  bool exact = false;     // `value` is an exact terminal value.
  bool terminal = false;  // The leaf is terminal; only counted in metrics.
};

// Request and result data for populating a single graph node, either from the
// node store or via NN evaluation.
struct Materialization {
  enum class State : uint8_t {
    kPending,    // Not yet processed, or abandoned on stop.
    kStoreHit,   // `record` was loaded from the store.
    kStoreMiss,  // Not in the store; needs NN evaluation.
    kQueued,     // Added to an NN computation that has not run yet.
    kEvaluated,  // `eval` holds the NN evaluation.
  };
  MaterializationId id;     // The open materialization this completes.
  NodeKey key;              // The node to materialize.
  PositionHistory history;  // Position at the node (for terminals and NN).
  NodeEval eval{};          // Legal moves, filled with priors and value.
  std::vector<uint8_t> record{};  // Serialized node data on kStoreHit.
  State state = State::kPending;
};

// A batch of node materializations sent by a visit worker to a store or
// evaluator thread and returned to the worker's inbox upon completion.
// `materializations` must not be reallocated between AddInput and FinishJob.
struct AsyncJob {
  enum class Kind {
    kEval,  // Visit worker -> evaluator -> visit worker: run NN evaluation.
    kLoad,  // Visit worker -> store thread -> visit worker: look up in store.
  };
  Kind kind;
  size_t worker;  // Owning visit worker whose inbox receives the completed job.
  // Timestamp when queued; batch deadlines are measured from the first job.
  std::chrono::steady_clock::time_point queued_at;
  std::vector<Materialization> materializations;
  size_t items() const { return materializations.size(); }
};

// One-way request from the controller to a store thread to evict `count` nodes
// from the graph to the store.
struct EvictionJob {
  size_t count;
};

// A task queued for a store thread: either a node-load batch or an eviction.
using StoreJob = std::variant<std::unique_ptr<AsyncJob>, EvictionJob>;

// Per-visit-worker message queue for completed `AsyncJob`s, `VisitWakeup`s,
// and control signals. Locking and unlocking `mutex` wakes the worker if its
// wait condition becomes true.
struct WorkerInbox {
  // Forces the worker to re-evaluate its wait condition after external state
  // (such as stop) changes.
  void Wake() { absl::MutexLock lock(mutex); }
  // Enqueues a completed job from a store or evaluator thread.
  void AddCompletion(std::unique_ptr<AsyncJob> job) {
    absl::MutexLock lock(mutex);
    completions.push_back(std::move(job));
  }
  // Signals the worker thread to exit once the session is drained.
  void Shutdown() {
    absl::MutexLock lock(mutex);
    shutdown = true;
  }

  absl::Mutex mutex;
  // Wakeups for visits owned by this worker.
  std::vector<VisitWakeup> wakeups ABSL_GUARDED_BY(mutex);
  // Completed jobs returned from the store or evaluator.
  std::vector<std::unique_ptr<AsyncJob>> completions ABSL_GUARDED_BY(mutex);
  // True when capacity was freed and the worker may admit new visits.
  bool admission_ready ABSL_GUARDED_BY(mutex) = false;
  bool shutdown ABSL_GUARDED_BY(mutex) = false;
};

// Generation-counted signal used by workers to wake the controller thread.
// The controller reads `generation()` before checking its conditions and waits
// for it to advance, avoiding lost wakeups.
class ControllerSignal {
 public:
  void Notify() {
    absl::MutexLock lock(mutex_);
    ++generation_;
  }
  uint64_t generation() {
    absl::MutexLock lock(mutex_);
    return generation_;
  }
  // Blocks until `Notify()` is called after `generation`, or until `deadline`.
  void WaitUntil(uint64_t generation,
                 std::chrono::steady_clock::time_point deadline) {
    absl::MutexLock lock(mutex_);
    const auto notified = [&] {
      mutex_.AssertReaderHeld();
      return generation_ != generation;
    };
    mutex_.AwaitWithTimeout(
        absl::Condition(&notified),
        absl::FromChrono(deadline - std::chrono::steady_clock::now()));
  }

  // Flags that the graph exceeded its node limit. Returns true if this is the
  // first request since `TakeEvictionRequests()` (caller should `Notify()`).
  bool RequestEviction() { return !eviction_requested_.exchange(true); }
  void TakeEvictionRequests() { eviction_requested_.store(false); }

 private:
  absl::Mutex mutex_;
  uint64_t generation_ ABSL_GUARDED_BY(mutex_) = 0;
  std::atomic<bool> eviction_requested_{false};
};

}  // namespace lczero::lc5
