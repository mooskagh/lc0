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

#include "search/lc5/session/visit_limiter.h"

#include <algorithm>

namespace lczero::lc5 {

VisitLimiter::VisitLimiter(const Config& config, VisitPool& visits,
                           Metrics& metrics)
    : config_(config),
      visits_(visits),
      metrics_(metrics),
      worker_waiting_(config.workers, false) {}

void VisitLimiter::Stop() {
  absl::MutexLock lock(mutex_);
  stopping_.store(true);
}

bool VisitLimiter::CanStartEval() {
  // A serial evaluator cannot check again until its computation returns.
  absl::MutexLock lock(mutex_);
  return !IsStopping();
}

size_t VisitLimiter::GetActiveVisitLimit() const {
  if (config_.active_visits_ratio <= 0.0f) return visits_.capacity();
  const double tree = static_cast<double>(config_.root_visits_at_start +
                                          metrics_.visits_completed.load());
  return std::min(visits_.capacity(),
                  kMinActiveVisits +
                      static_cast<size_t>(tree * config_.active_visits_ratio));
}

std::optional<VisitId> VisitLimiter::AdmitVisit(const VisitOrigin& origin) {
  absl::MutexLock lock(mutex_);
  return AdmitVisitLocked(origin);
}

std::vector<size_t> VisitLimiter::PopWorkersToWake() {
  std::vector<size_t> wake;
  absl::MutexLock lock(mutex_);
  if (IsAdmissionClosedLocked()) return wake;
  const size_t limit = GetActiveVisitLimit();
  const size_t active = visits_.active();
  size_t available = limit > active ? limit - active : 0;
  const size_t workers = worker_waiting_.size();
  for (size_t n = 0; n < workers && available > 0; ++n) {
    const size_t index = (next_worker_to_wake_ + n) % workers;
    if (!worker_waiting_[index]) continue;
    worker_waiting_[index] = false;
    wake.push_back(index);
    --available;
  }
  if (!wake.empty()) next_worker_to_wake_ = (wake.back() + 1) % workers;
  return wake;
}

bool VisitLimiter::IsAdmissionClosedLocked() const {
  return IsStopping() || total_admitted_count_ >= config_.node_limit;
}

std::optional<VisitId> VisitLimiter::AdmitVisitLocked(
    const VisitOrigin& origin) {
  if (IsAdmissionClosedLocked() || visits_.active() >= GetActiveVisitLimit()) {
    return std::nullopt;
  }
  const auto id = visits_.Allocate(origin);
  if (!id) return std::nullopt;
  ++total_admitted_count_;
  metrics_.visits_admitted.fetch_add(1);
  RaiseHighWater(metrics_.active_visits_high_water, visits_.active());
  return id;
}

}  // namespace lczero::lc5
