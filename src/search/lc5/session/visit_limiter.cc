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
      node_limit_(config.node_limit),
      worker_waiting_(config.workers, false) {}

void VisitLimiter::CapNodeLimit(uint64_t limit) {
  absl::MutexLock lock(mutex_);
  node_limit_.store(std::min(node_limit_.load(), limit));
}

void VisitLimiter::Stop() {
  absl::MutexLock lock(mutex_);
  stopping_.store(true);
}

size_t VisitLimiter::GetActiveVisitLimit() const {
  if (config_.active_visits_ratio <= 0.0f) return visits_.capacity();
  const double tree = static_cast<double>(config_.root_visits_at_start +
                                          metrics_.visits_completed.load());
  return std::min(visits_.capacity(),
                  kMinActiveVisits +
                      static_cast<size_t>(tree * config_.active_visits_ratio));
}

std::vector<VisitId> VisitLimiter::AdmitVisits(size_t worker,
                                               const VisitOrigin& origin,
                                               size_t count) {
  std::vector<VisitId> admitted;
  admitted.reserve(count);
  absl::MutexLock lock(mutex_);
  worker_waiting_[worker] = false;
  for (size_t i = 0; i < count; ++i) {
    if (IsAdmissionClosedLocked() ||
        visits_.active() >= GetActiveVisitLimit()) {
      // Shares the mutex with the capacity check in TakeWorkersToWake, so
      // it cannot miss a release.
      worker_waiting_[worker] = !IsAdmissionClosedLocked();
      break;
    }
    const auto id = visits_.Allocate(origin);
    if (!id) {
      worker_waiting_[worker] = true;
      break;
    }
    admitted.push_back(*id);
    ++total_admitted_count_;
  }
  if (!admitted.empty()) {
    metrics_.visits_admitted.fetch_add(admitted.size());
    RaiseHighWater(metrics_.active_visits_high_water, visits_.active());
  }
  return admitted;
}

std::vector<size_t> VisitLimiter::TakeWorkersToWake() {
  absl::MutexLock lock(mutex_);
  const size_t limit = GetActiveVisitLimit();
  const size_t active = visits_.active();
  if (IsAdmissionClosedLocked() || active >= limit) return {};

  std::vector<size_t> wake;
  for (size_t i = 0, w = next_worker_to_wake_;
       i < config_.workers && wake.size() < limit - active;
       ++i, w = (w + 1) % config_.workers) {
    if (worker_waiting_[w]) {
      worker_waiting_[w] = false;
      wake.push_back(w);
      next_worker_to_wake_ = (w + 1) % config_.workers;
    }
  }
  return wake;
}

bool VisitLimiter::IsAdmissionClosedLocked() const {
  return IsStopping() || total_admitted_count_ >= node_limit_.load();
}

}  // namespace lczero::lc5
