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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
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
    kAdmit,        // New visit; start descending from its origin.
    kSelect,       // Materialization finished; select below the node.
    kRestored,     // Restored from the store; select, skipping ResumeVisit.
    kBackup,       // Materialization finished; back up `value`.
    kBackupAbove,  // The node already holds `value`; back up the steps above.
    kCancel,       // Materialization cancelled on stop; cancel the visit.
  };
  VisitId id;     // The visit to continue.
  size_t worker;  // The visit's owner, the only worker that may change it.
  Kind kind = Kind::kAdmit;
  MaterializationTicketId ticket = 0;  // The ticket waited on; 0 for kAdmit.
  uint64_t generation = 0;  // The leaf node's generation once published.
  SearchValue value{};      // The leaf value; unset for kRestored and kCancel.
  bool exact = false;       // `value` is an exact terminal value.
  bool terminal = false;    // The leaf is terminal; only counted in metrics.
};

struct Ticket {
  NodeKey key;        // The node being materialized.
  VisitWakeup owner;  // The visit that opened the ticket; it does the I/O.
  std::vector<VisitWakeup> waiters;  // Later visits that reached the node.
};

// A ticket owner's request to the store and evaluator threads for one node's
// content: its stored record, or else an NN evaluation. Travels in jobs.
struct MaterializationRequest {
  enum class State : uint8_t {
    kPending,    // Not (yet) processed, or abandoned on stop.
    kStoreHit,   // `record` was loaded from the store.
    kStoreMiss,  // Not in the store; needs evaluation.
    kQueued,     // Added to a computation that has not run yet.
    kEvaluated,  // `payload` holds the evaluation.
  };
  MaterializationTicketId ticket;  // The ticket this request completes.
  NodeKey key;                     // The node to materialize.
  PositionHistory history;     // Down to the node: for terminals and NN input.
  ExpansionPayload payload{};  // Legal moves, then priors and leaf value.
  std::vector<uint8_t> record{};  // The stored record, on kStoreHit.
  State state = State::kPending;
};

// The arrows show where a job travels; only kPersist does not come back.
enum class JobKind {
  kEval,     // Visit worker -> evaluator -> worker: evaluate `requests`.
  kLoad,     // Visit worker -> store thread -> worker: load `requests`.
  kPersist,  // Controller -> store thread: evict `evict_count` nodes.
};

// Storage is fixed before AddInput; the whole job lives through compute
// and completion consumption, including immediate-cache results.
struct Job {
  JobKind kind;
  size_t worker;  // The owner, whose inbox gets the job back; not kPersist.
  // When queued; a batch's deadline counts from its first job's.
  std::chrono::steady_clock::time_point queued_at;
  std::vector<MaterializationRequest> requests;  // kEval and kLoad.
  size_t evict_count = 0;                        // kPersist.
  size_t items() const { return requests.size(); }
};

struct Batch {
  std::unique_ptr<BackendComputation> computation;
  std::vector<std::unique_ptr<Job>> jobs{};  // Whose inputs `computation` has.
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
  std::vector<std::unique_ptr<Job>> completions ABSL_GUARDED_BY(mutex);
  // Capacity was freed; try to admit visits again.
  bool admission_ready ABSL_GUARDED_BY(mutex) = false;
};

}  // namespace lczero::lc5
