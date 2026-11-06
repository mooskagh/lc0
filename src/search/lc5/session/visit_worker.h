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
#include <cstdint>
#include <deque>
#include <memory>
#include <span>
#include <thread>
#include <vector>

#include "search/lc5/metrics.h"
#include "search/lc5/session/evaluator.h"
#include "search/lc5/session/materialization_table.h"
#include "search/lc5/session/messages.h"
#include "search/lc5/session/visit_limiter.h"
#include "search/lc5/session/visit_pool.h"
#include "search/lc5/session/work_queue.h"
#include "search/lc5/settings.h"
#include "search/lc5/visit_engine.h"

namespace lczero::lc5 {

// One visit thread: admits visits from the root, steps them through the
// graph, opens and closes the materializations of the leaves they reach, and
// sends those leaves to the store and the evaluators in jobs.
//
// Everything below is touched only by this worker's thread, except through
// its inbox (other threads post to it) and IsDrained.
class VisitWorker {
 public:
  // What a visit worker shares with the rest of the session.
  struct Context {
    VisitEngine& engine;
    const Settings::Resolved& settings;
    const VisitOrigin& root;
    VisitPool& visits;
    Metrics& metrics;
    VisitLimiter& limiter;
    MaterializationTable& materializations;
    // Every worker's inbox, by worker index.
    std::span<WorkerInbox> inboxes;
    Evaluator& evaluator;
    WorkQueue<StoreJob>& store_jobs;
    // All workers' AsyncJobs sent and not yet finished.
    std::atomic<size_t>& jobs_in_flight;
    ControllerSignal& controller_signal;
  };

  VisitWorker(size_t worker_id, const Context& context);

  void Start() { thread_ = std::thread(&VisitWorker::Run, this); }
  void Join() {
    if (thread_.joinable()) thread_.join();
  }
  // After a stop: no visits, materializations or jobs are left, and nothing
  // waits in the inbox.
  bool IsDrained();

 private:
  friend class SearchSessionTestPeer;

  // In one pass of its loop, a worker advances at most this many visits and
  // admits at most this many new ones, then checks its inbox again.
  static constexpr size_t kVisitsPerPass = 32;
  // At most this many jobs sent and not yet finished...
  static constexpr size_t kMaxJobsInFlight = 16;
  // ...holding at most this many minibatches of materializations.
  static constexpr size_t kMaxItemsInFlightBatches = 2;
  // A job holds at most 1/kJobsPerBatch of a minibatch, so an evaluator can
  // mix several jobs into one batch.
  static constexpr size_t kJobsPerBatch = 4;

  bool IsStopping() const { return context_.limiter.IsStopping(); }

  // The thread body: runs until the inbox gets shut down.
  void Run();
  void ProcessInbox();
  void ResumeLocalVisits();
  void AdvanceRunnableVisits(size_t limit);
  void CancelVisits();
  void PublishState();
  bool HasLocalWork() const;
  bool HasQueuedIo() const {
    return !queued_loads_.empty() || !queued_evals_.empty();
  }

  // Admission and wakeups.
  // Admits new visits from the root, up to kVisitsPerPass and this worker's
  // share of the pool.
  void AdmitNewRootVisits();
  void WakeAdmissionWaiters();
  void DeliverWakeup(const VisitWakeup& wakeup);
  void SendWakeups(size_t to, std::vector<VisitWakeup>& batch);

  // Visit steps.
  void AdvanceVisit(VisitId id);
  void ResumeVisit(const VisitWakeup& wakeup);
  void BackupVisit(Visit& visit);
  void CancelVisit(Visit& visit);
  // Releases the visit's reservations and returns it to its origin.
  void RestartVisit(Visit& visit);
  void FinishVisit(Visit& visit, bool completed);

  // Materializations.
  // Joins or opens a materialization for `visit.current_key` and suspends
  // `visit`. Returns false if the node is already expanded, so the caller can
  // descend.
  bool SuspendVisit(Visit& visit);
  void PrepareMaterialization(Materialization materialization);
  void CompleteMaterialization(MaterializationId materialization_id,
                               ExpansionPayload payload);
  // Publishes a node restored from the store; returns false, with the
  // materialization still open, if the record was rejected.
  bool RestoreMaterialization(MaterializationId materialization_id,
                              std::span<const uint8_t> record);
  void CancelMaterialization(MaterializationId materialization_id);
  void CancelQueuedMaterializations();

  // AsyncJobs. A store thread or an evaluator runs a job and returns it to
  // the inbox; FinishJob finishes it.
  void SendJobs();
  bool CanSendJob() const;
  std::unique_ptr<AsyncJob> BuildJob();
  void SetProducing(bool producing);
  void FinishJob(std::unique_ptr<AsyncJob> job);

  // Session context and configuration, fixed for the worker's lifetime.
  const size_t worker_id_;
  const Context context_;
  WorkerInbox& inbox_;  // This worker's own (`context_.inboxes[worker_id_]`).

  // Visits and wakeups.
  std::deque<VisitId> ready_visits_;
  // Owned visits: ready or waiting on a materialization.
  size_t active_visits_ = 0;
  // Wakeups for this worker's own visits (e.g. from a terminal leaf closed
  // during AdvanceVisit), buffered to resume in ResumeLocalVisits.
  std::vector<VisitWakeup> local_wakeups_;
  // Indexed by target worker.
  std::vector<std::vector<VisitWakeup>> outgoing_wakeups_;

  // Materializations and in-flight jobs.
  std::deque<Materialization> queued_loads_;
  std::deque<Materialization> queued_evals_;
  size_t jobs_in_flight_ = 0;              // Sent and not yet finished.
  size_t materializations_in_flight_ = 0;  // Materializations in those jobs.
  // Whether this worker has local work that can send more eval jobs without
  // first receiving an inbox completion. Mirrored to `evaluator_` (which starts
  // with all workers counted as producing) so it knows when to flush a partial
  // batch.
  bool producing_ = true;

  // Per-pass notifications and shutdown/drain state.
  // Set during a pass and flushed once in PublishState.
  bool notify_admission_waiters_ = false;
  bool notify_controller_ = false;
  // Set once CancelVisits has run after stop; prevents re-waking on stop and
  // gates `drained_`.
  bool visits_cancelled_ = false;
  bool shutdown_ = false;  // Seen in the inbox; exit after this pass.
  std::atomic<bool> drained_{false};
  std::thread thread_;
};

}  // namespace lczero::lc5
