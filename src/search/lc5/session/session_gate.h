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

// Decides which visits may start and when the search stops. One mutex orders
// admission, stop requests, the bestmove commitment and NN authorization:
// - A visit is admitted before a stop or not at all, so once the controller
//   sees the stop, the pool's active count covers every visit to drain.
//   Splitting admission from stop would lose that.
// - No NN batch starts computing after a stop.
// - An abort can suppress the bestmove only until it is committed.
class SessionGate {
 public:
  enum class StopMode : uint8_t { kRunning, kRespondBestmove, kAbort };
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

  SessionGate(const Config& config, VisitPool& visits, Metrics& metrics);

  // Lock-free; for the hot path.
  bool IsStopping() const { return stop_mode() != StopMode::kRunning; }
  StopMode stop_mode() const { return stop_mode_.load(); }
  // A bestmove request is ignored once stopping. An abort overrides it until
  // its output is committed.
  void RequestStop(StopMode mode);
  // Whether to output the bestmove: the stop is a bestmove request, and from
  // now on an abort cannot suppress it.
  bool TryCommitOutput();
  // Whether an NN batch may compute: not stopping.
  bool AuthorizeEval();

  // Live visits allowed now: grows with the root's visits, as classic search
  // grows its collision limit, so small trees are not flooded.
  size_t GetActiveVisitLimit() const;
  // A new visit from `origin`; nullopt if admission is closed (stopping, or
  // the node budget is spent), GetActiveVisitLimit() visits are alive, or the
  // pool has no free slot.
  std::optional<VisitId> Admit(const VisitOrigin& origin);
  // Admits up to `count` visits from `origin` for `worker`, passing each to
  // `on_admit`. If capacity runs out first and admission is still open,
  // `worker` waits for TakeWaitersToWake to return it.
  template <typename OnAdmit>
  void AdmitFor(size_t worker, const VisitOrigin& origin, size_t count,
                OnAdmit on_admit) {
    absl::MutexLock lock(mutex_);
    admission_waiting_[worker] = false;
    for (size_t i = 0; i < count; ++i) {
      const auto id = AdmitLocked(origin);
      if (!id) {
        // Shares the mutex with the capacity check in TakeWaitersToWake, so
        // it cannot miss a release.
        admission_waiting_[worker] = !IsAdmissionClosedLocked();
        return;
      }
      on_admit(*id);
    }
  }
  // Waiting workers, at most one per visit that may be admitted now, in turn;
  // they stop waiting. A competing allocator may still win the capacity; the
  // woken worker then waits again.
  std::vector<size_t> TakeWaitersToWake();

 private:
  bool IsAdmissionClosedLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  std::optional<VisitId> AdmitLocked(const VisitOrigin& origin)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  const Config config_;
  VisitPool& visits_;
  Metrics& metrics_;

  absl::Mutex mutex_;
  uint64_t admitted_ ABSL_GUARDED_BY(mutex_) = 0;
  std::vector<bool> admission_waiting_ ABSL_GUARDED_BY(mutex_);
  size_t next_admission_waiter_ ABSL_GUARDED_BY(mutex_) = 0;
  // Written under mutex_, read anywhere.
  std::atomic<StopMode> stop_mode_{StopMode::kRunning};
  bool output_committed_ ABSL_GUARDED_BY(mutex_) = false;
};

}  // namespace lczero::lc5
