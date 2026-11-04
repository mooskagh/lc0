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

// The messages SearchRun's threads pass to each other.

// Tells a visit's owner how to continue it.
struct Continuation {
  enum class Kind : uint8_t {
    kAdmit,        // Newly admitted visit.
    kSelect,       // Materialization finished; select below the node.
    kRestored,     // Restored from the store; select, skipping ResumeVisit.
    kBackup,       // Materialization finished; back up `value`.
    kBackupAbove,  // The node already holds `value`; back up the steps above.
    kCancel,       // Materialization cancelled on stop; cancel the visit.
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
// A kPersist job evicts `evict_count` nodes from the graph to the store. The
// controller creates it, and a store thread finishes it without returning it
// to a worker.
enum class JobKind { kEval, kLoad, kPersist };
// Storage is fixed before AddInput; the whole job lives through compute
// and completion consumption, including immediate-cache results.
struct Job {
  JobKind kind;
  size_t worker;
  std::chrono::steady_clock::time_point queued_at;
  std::vector<Request> requests;
  size_t evict_count = 0;
  size_t items() const { return requests.size(); }
};
// A worker's inbox. Senders lock `mutex` and add to it; unlocking wakes the
// owner if its wait condition now holds.
struct Mailbox {
  // Makes the owner re-evaluate its wait condition after external state
  // (stop, shutdown) changed.
  void Wake() { absl::MutexLock lock(mutex); }

  absl::Mutex mutex;
  std::vector<Continuation> continuations ABSL_GUARDED_BY(mutex);
  std::vector<std::unique_ptr<Job>> completions ABSL_GUARDED_BY(mutex);
  bool admission_ready ABSL_GUARDED_BY(mutex) = false;
};
struct Batch {
  std::unique_ptr<BackendComputation> computation;
  std::vector<std::unique_ptr<Job>> jobs{};
  bool timed_out = false;
};

}  // namespace lczero::lc5
