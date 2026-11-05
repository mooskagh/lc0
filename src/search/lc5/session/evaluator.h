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

#include <atomic>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "neural/backend.h"
#include "search/lc5/metrics.h"
#include "search/lc5/session/messages.h"
#include "search/lc5/session/visit_limiter.h"
#include "search/lc5/session/work_queue.h"
#include "search/lc5/settings.h"

namespace lczero::lc5 {

// The evaluator threads' side of the session: collects the visit workers'
// eval jobs into NN batches, computes them, and returns the jobs.
//
// A batch below the minibatch size waits for more jobs until the oldest job's
// deadline, or until it is starved: no worker can send a job without a
// collected one coming back first.
class Evaluator {
 public:
  struct Context {
    Backend& backend;
    const Settings::Resolved& settings;
    VisitLimiter& limiter;
    Metrics& metrics;
    // Every visit worker's inbox, by worker index; jobs go back there.
    std::span<WorkerInbox> inboxes;
    // All workers' AsyncJobs sent and not yet finished.
    const std::atomic<size_t>& jobs_in_flight;
  };
  explicit Evaluator(const Context& context);

  // The body of each evaluator thread: runs until Close().
  void Run();
  // For the visit workers.
  void Push(std::unique_ptr<AsyncJob> job);
  // A worker starts or stops producing: it can (or no longer can) send a
  // job without a collected job's completion. Workers start producing.
  void SetProducing(bool producing);
  // Makes waiting collectors check the stop again.
  void NotifyStop() { jobs_.NotifyPredicateChanged(); }
  void Close() { jobs_.Close(); }
  size_t queued() const { return jobs_.Size(); }

 private:
  friend class SearchSessionTestPeer;

  struct Batch {
    std::unique_ptr<BackendComputation> computation;
    std::vector<std::unique_ptr<AsyncJob>> jobs{};  // Whose inputs it has.
    bool timed_out = false;  // Flushed at the deadline; only counted.
  };

  bool IsStopping() const { return context_.limiter.IsStopping(); }
  // Takes `first` and more queued jobs while they fit the minibatch.
  Batch CollectBatch(std::unique_ptr<AsyncJob> first);
  void AddJobToBatch(BackendComputation& computation, AsyncJob& job);
  void EvaluateBatch(Batch batch);
  bool IsStarved() const;
  void ReturnJob(std::unique_ptr<AsyncJob> job);

  const Context context_;

  WorkQueue<std::unique_ptr<AsyncJob>> jobs_;
  // Jobs held by collectors, not computing or returned. If these are all
  // remaining jobs and producers are blocked, further collection depends on
  // flushing at least one computation.
  std::atomic<size_t> collecting_jobs_{0};
  // Workers that are producing.
  std::atomic<size_t> producers_;
};

}  // namespace lczero::lc5
