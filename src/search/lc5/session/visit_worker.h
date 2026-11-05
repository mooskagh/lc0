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
#include <vector>

#include "search/lc5/metrics.h"
#include "search/lc5/session/evaluator.h"
#include "search/lc5/session/messages.h"
#include "search/lc5/session/session_gate.h"
#include "search/lc5/session/ticket_table.h"
#include "search/lc5/session/visit_pool.h"
#include "search/lc5/session/work_queue.h"
#include "search/lc5/settings.h"
#include "search/lc5/visit_engine.h"

namespace lczero::lc5 {

// One visit thread: admits visits from the root, steps them through the
// graph, opens and closes the tickets of the leaves they reach, and sends
// those leaves to the store and the evaluators in jobs.
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
    SessionGate& gate;
    TicketTable& tickets;
    // Every worker's inbox, by worker index.
    std::span<WorkerInbox> inboxes;
    Evaluator& evaluator;
    WorkQueue<StoreJob>& store_jobs;
    // All workers' AsyncJobs sent and not yet finished.
    std::atomic<size_t>& jobs_in_flight;
    ControllerSignal& controller;
  };

  VisitWorker(size_t index, const Context& context);

  // The thread body: runs until the inbox gets shut down.
  void Run();
  // After a stop: no visits, tickets or jobs are left, and nothing waits in
  // the inbox.
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

  bool IsStopping() const { return gate_.IsStopping(); }

  // The loop.
  void ProcessInbox();
  void ResumeLocalVisits();
  void AdvanceRunnableVisits(size_t limit);
  void CancelVisits();
  void PublishState();
  bool HasLocalWork() const;
  bool HasQueuedIo() const { return !loads_.empty() || !evals_.empty(); }

  // Admission.
  // Admits new visits from the root, up to kVisitsPerPass and this worker's
  // share of the pool.
  void AdmitNewRootVisits();
  void WakeAdmissionWaiters();

  // Visit.
  void AdvanceVisit(VisitId id);
  void ResumeVisit(const VisitWakeup& wakeup);
  bool SuspendVisit(Visit& visit);
  void BackupVisit(Visit& visit);
  void CancelVisit(Visit& visit);
  // Releases the visit's reservations and returns it to its origin.
  void RestartVisit(Visit& visit);
  void FinishVisit(Visit& visit, bool completed);

  // VisitWakeup.
  void DeliverWakeup(const VisitWakeup& wakeup);
  void SendWakeups(size_t to, std::vector<VisitWakeup>& batch);

  // Materialization. SuspendVisit opens a Ticket and its Materialization.
  void PrepareMaterialization(Materialization materialization);
  void CompleteTicket(MaterializationTicketId ticket, ExpansionPayload payload);
  // Publishes a node restored from the store; returns false, with the ticket
  // still open, if the record was rejected.
  bool RestoreTicket(MaterializationTicketId ticket,
                     std::span<const uint8_t> record);
  void CancelTicket(MaterializationTicketId ticket);
  void CancelQueuedMaterializations();

  // AsyncJob. A store thread or an evaluator runs a job and returns it to
  // the inbox; FinishJob finishes it.
  void SendJobs();
  bool CanSendJob() const;
  std::unique_ptr<AsyncJob> BuildJob();
  void SetProducing(bool producing);
  void FinishJob(std::unique_ptr<AsyncJob> job);

  const size_t index_;
  VisitEngine& engine_;
  const Settings::Resolved& settings_;
  const VisitOrigin& root_;
  VisitPool& visits_;
  Metrics& metrics_;
  SessionGate& gate_;
  TicketTable& tickets_;
  const std::span<WorkerInbox> inboxes_;
  WorkerInbox& inbox_;  // This worker's own.
  Evaluator& evaluator_;
  WorkQueue<StoreJob>& store_jobs_;
  std::atomic<size_t>& session_jobs_in_flight_;
  ControllerSignal& controller_;
  const bool has_store_;
  // The graph has a node limit to keep, by evicting to the store.
  const bool has_node_limit_;

  std::deque<VisitId> runnable_;
  size_t active_visits_ = 0;  // Owned visits: runnable or waiting on a ticket.
  std::deque<Materialization> loads_;
  std::deque<Materialization> evals_;
  // Wakeups for this worker's own visits, resumed in ProcessInbox: never
  // while AdvanceVisit may be stepping the visit.
  std::vector<VisitWakeup> local_wakeups_;
  std::vector<std::vector<VisitWakeup>> outgoing_;  // Indexed by worker.
  size_t jobs_in_flight_ = 0;   // Sent and not yet finished.
  size_t items_in_flight_ = 0;  // The materializations in those jobs.
  bool cancelled_ = false;
  bool shutdown_ = false;  // Seen in the inbox; exit after this pass.
  bool released_capacity_ = false;
  bool controller_event_ = false;
  std::atomic<bool> drained_{false};
  // Conservative: this worker may send a job without waiting for a collected
  // job's completion. Tells the evaluator of each change. Initially true until
  // the first work chunk is inspected.
  bool producing_ = true;
};

}  // namespace lczero::lc5
