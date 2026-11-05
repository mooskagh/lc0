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
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "search/lc5/metrics.h"
#include "search/lc5/session/visit_pool.h"
#include "search/lc5/visit.h"

namespace lczero::lc5 {

// Admission control for one search session: decides whether a new visit may
// start (not stopping, node budget left, in-flight limit).
class VisitLimiter {
 public:
  struct Config {
    size_t workers;
    // Admissions allowed in total: the node budget.
    uint64_t node_limit;
    // Root visits before the search, for GetActiveVisitLimit.
    uint64_t root_visits_at_start;
    float active_visits_ratio;
  };
  // Live visits allowed while the tree is still empty; see GetActiveVisitLimit.
  static constexpr size_t kMinActiveVisits = 16;

  VisitLimiter(const Config& config, VisitPool& visits, Metrics& metrics);
  uint64_t node_limit() const { return config_.node_limit; }

  // Lock-free; for the hot path.
  bool IsStopping() const { return stopping_.load(); }
  // Closes admission of new visits and evaluations.
  void Stop();
  // Returns whether an NN batch may start computing (i.e. not stopping).
  bool CanStartEval();

  // Returns the max number of in-flight visits allowed now. Grows with the
  // tree size (as classic grows its collision limit), so small trees are not
  // flooded.
  size_t GetActiveVisitLimit() const;
  // Starts a new visit at `origin`. Returns nullopt if unsuccessful (stopping,
  // node budget spent, at the in-flight limit, etc.).
  std::optional<VisitId> AdmitVisit(const VisitOrigin& origin);
  // Starts up to `count` visits for `worker`, passing each to `on_admit`. If
  // it hits the in-flight limit, marks `worker` as waiting, so that
  // PopWorkersToWake returns it once a slot frees up.
  template <typename OnAdmit>
  void AdmitVisits(size_t worker, const VisitOrigin& origin, size_t count,
                   OnAdmit on_admit) {
    absl::MutexLock lock(mutex_);
    worker_waiting_[worker] = false;
    for (size_t i = 0; i < count; ++i) {
      const auto id = AdmitVisitLocked(origin);
      if (!id) {
        // Shares the mutex with the capacity check in PopWorkersToWake, so
        // it cannot miss a release.
        worker_waiting_[worker] = !IsAdmissionClosedLocked();
        return;
      }
      on_admit(*id);
    }
  }
  // Returns the waiting workers to wake now, at most one per free in-flight
  // slot, round-robin, and clears their waiting flag. If another worker takes
  // the slot first, the woken one just waits again.
  std::vector<size_t> PopWorkersToWake();

 private:
  bool IsAdmissionClosedLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  std::optional<VisitId> AdmitVisitLocked(const VisitOrigin& origin)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  const Config config_;
  VisitPool& visits_;
  Metrics& metrics_;

  // Guards admission and stop together: no visit is admitted after a stop.
  absl::Mutex mutex_;
  // All visits admitted in this session (not just in flight).
  uint64_t total_admitted_count_ ABSL_GUARDED_BY(mutex_) = 0;
  // Per worker: whether it is waiting for an in-flight slot.
  std::vector<bool> worker_waiting_ ABSL_GUARDED_BY(mutex_);
  // Where PopWorkersToWake starts its round-robin scan.
  size_t next_worker_to_wake_ ABSL_GUARDED_BY(mutex_) = 0;
  // Written under mutex_, read anywhere.
  std::atomic<bool> stopping_{false};
};

}  // namespace lczero::lc5
