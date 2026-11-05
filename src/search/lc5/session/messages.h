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
#include "neural/backend.h"
#include "search/lc5/graph/expansion.h"
#include "search/lc5/graph/key.h"
#include "search/lc5/graph/node.h"
#include "search/lc5/graph/value.h"
#include "search/lc5/visit.h"

namespace lczero::lc5 {

// The messages SearchSession's threads pass to each other.

// Tells a visit's owner how to continue it.
struct VisitWakeup {
  enum class Kind : uint8_t {
    kSelect,       // Materialization finished; select below the node.
    kRestored,     // Restored from the store; select, skipping ResumeVisit.
    kBackup,       // Materialization finished; back up `value`.
    kBackupAbove,  // The node already holds `value`; back up the steps above.
    kCancel,       // Materialization cancelled on stop; cancel the visit.
  };
  VisitId id;     // The visit to continue.
  size_t worker;  // The visit's owner, the only worker that may change it.
  Kind kind = Kind::kSelect;
  MaterializationTicketId ticket = 0;  // The ticket waited on.
  uint64_t generation = 0;  // The leaf node's generation once published.
  SearchValue value{};      // The leaf value; unset for kRestored and kCancel.
  bool exact = false;       // `value` is an exact terminal value.
  bool terminal = false;    // The leaf is terminal; only counted in metrics.
};

// One node being materialized: what its ticket owner sends, in jobs, to the
// store and evaluator workers to get the node's content (its stored record,
// or else an NN evaluation).
struct Materialization {
  enum class State : uint8_t {
    kPending,    // Not (yet) processed, or abandoned on stop.
    kStoreHit,   // `record` was loaded from the store.
    kStoreMiss,  // Not in the store; needs evaluation.
    kQueued,     // Added to a computation that has not run yet.
    kEvaluated,  // `payload` holds the evaluation.
  };
  MaterializationTicketId ticket;  // The ticket this materialization completes.
  NodeKey key;                     // The node to materialize.
  PositionHistory history;     // Down to the node: for terminals and NN input.
  ExpansionPayload payload{};  // Legal moves, then priors and leaf value.
  std::vector<uint8_t> record{};  // The stored record, on kStoreHit.
  State state = State::kPending;
};

// Slow work a visit worker hands to another thread, which runs it while the
// worker carries on: a store lookup or an NN evaluation of some of its
// materializations. The job comes back to the worker, which finishes it.
// Storage is fixed before AddInput; the whole job lives through compute and
// FinishJob, including immediate-cache results.
struct AsyncJob {
  // The arrows show where a job travels.
  enum class Kind {
    kEval,  // Visit worker -> evaluator -> worker: NN-evaluate.
    kLoad,  // Visit worker -> store thread -> worker: look up in the store.
  };
  Kind kind;
  size_t worker;  // The owner, whose inbox gets the job back.
  // When queued; a batch's deadline counts from its first job's.
  std::chrono::steady_clock::time_point queued_at;
  std::vector<Materialization> materializations;
  size_t items() const { return materializations.size(); }
};

// The controller's request to a store thread: move `count` nodes from the
// graph to the store. Nothing comes back.
struct EvictionJob {
  size_t count;
};

// What store threads run.
using StoreJob = std::variant<std::unique_ptr<AsyncJob>, EvictionJob>;

struct Batch {
  std::unique_ptr<BackendComputation> computation;
  std::vector<std::unique_ptr<AsyncJob>>
      jobs{};              // Whose inputs `computation` has.
  bool timed_out = false;  // Flushed at the deadline; only counted in metrics.
};

// A worker's inbox. Senders lock `mutex` and add to it; unlocking wakes the
// owner if its wait condition now holds.
struct WorkerInbox {
  // Makes the owner re-evaluate its wait condition after external state
  // (stop, shutdown) changed.
  void Wake() { absl::MutexLock lock(mutex); }

  absl::Mutex mutex;
  // For this worker's visits.
  std::vector<VisitWakeup> wakeups ABSL_GUARDED_BY(mutex);
  // Jobs back from the store or the evaluator.
  std::vector<std::unique_ptr<AsyncJob>> completions ABSL_GUARDED_BY(mutex);
  // Capacity was freed; try to admit visits again.
  bool admission_ready ABSL_GUARDED_BY(mutex) = false;
};

// Wakes the controller. Any thread may Notify. The controller takes the
// generation before checking its conditions, then waits for a later one, so a
// notification during the check is never lost.
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
  // Waits for a Notify after `generation` was taken, or until `deadline`.
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

  // The graph is over its node limit: true for the first request since the
  // controller last took them, which is the one to Notify for.
  bool RequestEviction() { return !eviction_requested_.exchange(true); }
  void TakeEvictionRequests() { eviction_requested_.store(false); }

 private:
  absl::Mutex mutex_;
  uint64_t generation_ ABSL_GUARDED_BY(mutex_) = 0;
  std::atomic<bool> eviction_requested_{false};
};

}  // namespace lczero::lc5
