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

#include "search/lc5/session/visit_worker.h"

#include <absl/synchronization/mutex.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <limits>
#include <utility>

#include "search/lc5/graph/prior.h"

namespace lczero::lc5 {
namespace {

// Returns a terminal payload, or a nonterminal one with the legal moves.
ExpansionPayload DetectTerminal(const PositionHistory& history) {
  ExpansionPayload payload;
  const Position& position = history.Last();
  const ChessBoard& board = position.GetBoard();
  const auto legal = board.GenerateLegalMoves();
  if (legal.empty()) {
    payload.terminal = board.IsUnderCheck() ? TerminalKind::kCheckmate
                                            : TerminalKind::kStalemate;
  } else if (!board.HasMatingMaterial()) {
    payload.terminal = TerminalKind::kInsufficientMaterial;
  } else if (position.GetRule50Ply() >= 100) {
    payload.terminal = TerminalKind::kRule50;
  } else if (position.GetRepetitions() >= 2) {
    payload.terminal = TerminalKind::kRepetition;
  } else {
    payload.moves.assign(legal.begin(), legal.end());
    return payload;
  }
  payload.leaf = TerminalValue(payload.terminal);
  return payload;
}

// Rounds priors to the precision the graph stores them in, then orders edges
// by descending prior, ties by move encoding. Sorting the rounded values keeps
// the order the graph relies on even when rounding makes two priors equal.
void SortByPrior(ExpansionPayload& payload) {
  std::vector<std::pair<float, Move>> edges;
  edges.reserve(payload.moves.size());
  for (size_t i = 0; i < payload.moves.size(); ++i) {
    edges.emplace_back(PackedPrior(payload.priors[i]), payload.moves[i]);
  }
  std::ranges::sort(edges, [](const auto& a, const auto& b) {
    if (a.first != b.first) return a.first > b.first;
    return a.second.raw_data() < b.second.raw_data();
  });
  for (size_t i = 0; i < edges.size(); ++i) {
    std::tie(payload.priors[i], payload.moves[i]) = edges[i];
  }
}

// Moves up to `limit` items from the front of `from` to the back of `to`.
template <typename T>
void MoveFront(std::deque<T>& from, std::vector<T>& to, size_t limit) {
  while (!from.empty() && to.size() < limit) {
    to.push_back(std::move(from.front()));
    from.pop_front();
  }
}

}  // namespace

VisitWorker::VisitWorker(size_t index, const Context& context)
    : index_(index),
      engine_(context.engine),
      settings_(context.settings),
      root_(context.root),
      visits_(context.visits),
      metrics_(context.metrics),
      limiter_(context.limiter),
      tickets_(context.tickets),
      inboxes_(context.inboxes),
      inbox_(context.inboxes[index]),
      evaluator_(context.evaluator),
      store_jobs_(context.store_jobs),
      session_jobs_in_flight_(context.jobs_in_flight),
      controller_(context.controller),
      has_store_(context.engine.HasStore()),
      has_node_limit_(has_store_ && settings_.graph_node_limit > 0),
      outgoing_(context.inboxes.size()) {}

bool VisitWorker::IsDrained() {
  absl::MutexLock lock(inbox_.mutex);
  return inbox_.wakeups.empty() && inbox_.completions.empty() &&
         drained_.load();
}

//////////////////////////////////////////////////////////////////////////////
// Visits: admission, the worker loop and each visit's steps.
//////////////////////////////////////////////////////////////////////////////

void VisitWorker::AdmitNewRootVisits() {
  // A modest per-worker share prevents one worker from monopolizing the pool.
  const size_t share =
      (visits_.capacity() + inboxes_.size() - 1) / inboxes_.size();
  const size_t count =
      std::min(kVisitsPerPass, share - std::min(share, active_visits_));
  const auto admitted = limiter_.AdmitVisits(index_, root_, count);
  active_visits_ += admitted.size();
  runnable_.insert(runnable_.end(), admitted.begin(), admitted.end());
  RaiseHighWater(metrics_.visits_ready_high_water, runnable_.size());
}

void VisitWorker::WakeAdmissionWaiters() {
  for (size_t index : limiter_.TakeWorkersToWake()) {
    auto& inbox = inboxes_[index];
    absl::MutexLock lock(inbox.mutex);
    inbox.admission_ready = true;
  }
}

void VisitWorker::SendWakeups(size_t to, std::vector<VisitWakeup>& batch) {
  if (batch.empty()) return;
  {
    auto& inbox = inboxes_[to];
    absl::MutexLock lock(inbox.mutex);
    inbox.wakeups.insert(inbox.wakeups.end(), batch.begin(), batch.end());
  }
  batch.clear();
  metrics_.inbox_notifications.fetch_add(1);
}

void VisitWorker::DeliverWakeup(const VisitWakeup& wakeup) {
  (wakeup.worker == index_ ? local_wakeups_ : outgoing_[wakeup.worker])
      .push_back(wakeup);
}

void VisitWorker::ResumeVisit(const VisitWakeup& wakeup) {
  // A waiting visit is on exactly one ticket, which wakes it exactly once,
  // and only that wakeup can make it ready or release it.
  Visit& visit = visits_.Get(wakeup.id);
  assert(visit.state == VisitState::kWaitingMaterialization);
  assert(visit.waiting_ticket == wakeup.ticket);
  visit.waiting_ticket = 0;
  // On stop, visits finish only when their evaluation already exists.
  if (wakeup.kind == VisitWakeup::Kind::kCancel ||
      (IsStopping() && (wakeup.kind == VisitWakeup::Kind::kSelect ||
                        wakeup.kind == VisitWakeup::Kind::kRestored))) {
    CancelVisit(visit);
    return;
  }
  visit.path.back().generation = wakeup.generation;
  visit.path.back().selected_move.reset();
  visit.result = wakeup.value;
  visit.result_exact = wakeup.exact;
  if (wakeup.kind == VisitWakeup::Kind::kBackup ||
      wakeup.kind == VisitWakeup::Kind::kBackupAbove) {
    visit.leaf_counted = wakeup.kind == VisitWakeup::Kind::kBackupAbove;
    visit.state = VisitState::kReadyBackup;
  } else if (wakeup.kind == VisitWakeup::Kind::kRestored) {
    visit.state = VisitState::kReadySelect;
    metrics_.visits_resumed.fetch_add(1);
  } else {
    switch (engine_.ResumeVisit(visit, metrics_)) {
      case VisitEngine::ResumeAction::kRestart:
        RestartVisit(visit);
        break;
      case VisitEngine::ResumeAction::kBackup:
        visit.leaf_counted = false;
        visit.state = VisitState::kReadyBackup;
        break;
      case VisitEngine::ResumeAction::kSelect:
        visit.state = VisitState::kReadySelect;
        metrics_.visits_resumed.fetch_add(1);
        break;
    }
  }
  if (wakeup.terminal) metrics_.terminal_visits.fetch_add(1);
  runnable_.push_back(wakeup.id);
}

void VisitWorker::Run() {
  for (;;) {
    ProcessInbox();
    if (IsStopping()) {
      CancelVisits();
    } else {
      AdvanceRunnableVisits(kVisitsPerPass);
      AdmitNewRootVisits();
    }
    SendJobs();
    for (size_t i = 0; i < inboxes_.size(); ++i) {
      SendWakeups(i, outgoing_[i]);
    }
    PublishState();
    if (shutdown_) break;
    // Keep going locally without a wake or controller handoff. Suspended
    // visits and the in-flight limits wait for whole completions.
    if (HasLocalWork()) continue;
    // Senders evaluate this on unlock; `cancelled` is written only by this
    // worker, never while it waits here.
    absl::MutexLock lock(inbox_.mutex);
    const auto ready = [&] {
      inbox_.mutex.AssertReaderHeld();
      return inbox_.shutdown || inbox_.admission_ready ||
             !inbox_.wakeups.empty() || !inbox_.completions.empty() ||
             (!cancelled_ && IsStopping());
    };
    inbox_.mutex.Await(absl::Condition(&ready));
  }
}

void VisitWorker::ProcessInbox() {
  std::vector<VisitWakeup> wakeups;
  std::vector<std::unique_ptr<AsyncJob>> completions;
  bool admission_ready;
  {
    absl::MutexLock lock(inbox_.mutex);
    wakeups.swap(inbox_.wakeups);
    completions.swap(inbox_.completions);
    admission_ready = std::exchange(inbox_.admission_ready, false);
    shutdown_ = inbox_.shutdown;
  }
  if (admission_ready || !wakeups.empty() || !completions.empty()) {
    // Actual incoming work can enable preparation or refill. This transition
    // disables starvation and needs no broadcast.
    SetProducing(true);
    if (drained_.exchange(false)) controller_event_ = true;
  }
  for (const auto& wakeup : wakeups) ResumeVisit(wakeup);
  for (auto& job : completions) FinishJob(std::move(job));
  ResumeLocalVisits();
}

void VisitWorker::ResumeLocalVisits() {
  // ResumeVisit wakes no visits, so this empties the queue.
  for (const auto& wakeup : local_wakeups_) ResumeVisit(wakeup);
  local_wakeups_.clear();
}

void VisitWorker::AdvanceRunnableVisits(size_t limit) {
  for (size_t i = 0; i < limit && !runnable_.empty(); ++i) {
    const VisitId id = runnable_.front();
    runnable_.pop_front();
    AdvanceVisit(id);
  }
}

void VisitWorker::CancelVisits() {
  // Every visit that is not waiting on a ticket is runnable: ready backups
  // finish, the rest are cancelled. Waiting visits stay until their ticket
  // completes or is cancelled, and come back through ResumeVisit.
  cancelled_ = true;
  AdvanceRunnableVisits(std::numeric_limits<size_t>::max());
}

void VisitWorker::PublishState() {
  const bool drained = cancelled_ && active_visits_ == 0 &&
                       jobs_in_flight_ == 0 && !HasQueuedIo() &&
                       local_wakeups_.empty();
  if (drained_.exchange(drained) != drained) {
    controller_event_ = true;
  }
  if (std::exchange(released_capacity_, false)) WakeAdmissionWaiters();
  if (std::exchange(controller_event_, false)) controller_.Notify();
  SetProducing(CanSendJob() && (!runnable_.empty() || !local_wakeups_.empty() ||
                                HasQueuedIo()));
}

bool VisitWorker::HasLocalWork() const {
  return !local_wakeups_.empty() || (!IsStopping() && !runnable_.empty()) ||
         (CanSendJob() && HasQueuedIo());
}

void VisitWorker::AdvanceVisit(VisitId id) {
  // Only this worker releases its visits, and never one that is runnable.
  Visit& visit = visits_.Get(id);
  if (visit.state == VisitState::kReadyBackup) {
    BackupVisit(visit);
    return;
  }
  if (IsStopping()) {
    CancelVisit(visit);
    return;
  }
  assert(visit.state == VisitState::kReadySelect);
  for (;;) {
    if (IsStopping()) {
      CancelVisit(visit);
      return;
    }
    if (engine_.DescendVisit(visit, metrics_) ==
        VisitEngine::DescendResult::kValuedLeaf) {
      BackupVisit(visit);
      return;
    }
    if (SuspendVisit(visit)) return;
  }
}

void VisitWorker::BackupVisit(Visit& visit) {
  engine_.BackupVisit(visit, metrics_);
  RaiseHighWater(metrics_.max_depth, visit.path.size());
  FinishVisit(visit, true);
}

void VisitWorker::CancelVisit(Visit& visit) {
  engine_.CancelVisit(visit, metrics_);
  FinishVisit(visit, false);
}

void VisitWorker::RestartVisit(Visit& visit) {
  engine_.CancelVisit(visit, metrics_);
  visit.history = visit.origin.history;
  visit.current_key = visit.origin.key;
  visit.path = visit.origin.backup_prefix;
  visit.two_fold_origin.reset();
  visit.state = VisitState::kReadySelect;
  metrics_.visits_restarted.fetch_add(1);
}

void VisitWorker::FinishVisit(Visit& visit, bool completed) {
  if (completed) {
    const uint64_t count = metrics_.visits_completed.fetch_add(1) + 1;
    if (limiter_.node_limit() == count) controller_event_ = true;
  } else {
    metrics_.visits_cancelled.fetch_add(1);
  }
  --active_visits_;
  visits_.Release(visit.id);
  // Admission and the controller are signalled once per pass, not per visit.
  released_capacity_ = true;
}

//////////////////////////////////////////////////////////////////////////////
// Materialization: tickets, materializations and jobs.
//////////////////////////////////////////////////////////////////////////////

bool VisitWorker::SuspendVisit(Visit& visit) {
  FindOrCreateResult node{};
  const auto joined = tickets_.Join(
      visit.current_key, {.id = visit.id, .worker = index_},
      [&](MaterializationTicketId id) {
        node = engine_.FindOrCreateMaterialization(visit.current_key, id);
        assert(node.lifecycle == NodeLifecycle::kExpanded || node.ticket == id);
        return node.lifecycle != NodeLifecycle::kExpanded;
      });
  if (node.created) metrics_.graph_nodes_created.fetch_add(1);
  if (!joined) return false;
  const bool owner = joined->owner;
  if (owner) {
    metrics_.tickets_created.fetch_add(1);
  } else {
    metrics_.ticket_waiters.fetch_add(1);
    RaiseHighWater(metrics_.maximum_waiters_per_ticket, joined->waiters);
  }
  visit.path.back().generation = node.generation;
  visit.waiting_ticket = joined->id;
  visit.state = VisitState::kWaitingMaterialization;
  metrics_.visits_suspended.fetch_add(1);
  // Wakes the controller once the graph outgrows its limit, so eviction keeps
  // up with fast searches between its periodic checks.
  if (owner && has_node_limit_ &&
      engine_.GetGraphSize() > settings_.graph_node_limit &&
      controller_.RequestEviction()) {
    controller_event_ = true;
  }
  if (owner) {
    Materialization materialization{.ticket = joined->id,
                                    .key = visit.current_key,
                                    .history = visit.history};
    // A stored node may hold statistics, even if terminal: look it up first.
    if (has_store_) {
      loads_.push_back(std::move(materialization));
    } else {
      PrepareMaterialization(std::move(materialization));
    }
  }
  return true;
}

void VisitWorker::PrepareMaterialization(Materialization materialization) {
  if (IsStopping()) {
    CancelTicket(materialization.ticket);
    return;
  }
  materialization.payload = DetectTerminal(materialization.history);
  if (materialization.payload.terminal != TerminalKind::kNonTerminal) {
    CompleteTicket(materialization.ticket, std::move(materialization.payload));
    return;
  }
  materialization.payload.priors.resize(materialization.payload.moves.size());
  materialization.state = Materialization::State::kPending;
  metrics_.eval_requests.fetch_add(1);
  evals_.push_back(std::move(materialization));
}

void VisitWorker::CompleteTicket(MaterializationTicketId ticket_id,
                                 ExpansionPayload payload) {
  MaterializationResult installed;
  const auto ticket = tickets_.Close(ticket_id, [&](NodeKey key) {
    installed = engine_.CompleteMaterialization(key, ticket_id, payload);
    return true;
  });
  assert(ticket->owner.worker == index_);
  RaiseHighWater(metrics_.graph_size_high_water, engine_.GetGraphSize());
  if (!installed.accepted) metrics_.publications_rejected.fetch_add(1);
  // The install already counted the leaf value on the node, so the owner
  // backs it up only through the steps above. Waiters select below the node
  // unless it is terminal, in which case they back up the whole path like a
  // later visit to a terminal. A rejected publication resumes selection only.
  const bool terminal =
      installed.accepted && payload.terminal != TerminalKind::kNonTerminal;
  const auto wake = [&](VisitWakeup wakeup, bool owner) {
    wakeup.kind = !installed.accepted ? VisitWakeup::Kind::kSelect
                  : owner             ? VisitWakeup::Kind::kBackupAbove
                  : terminal          ? VisitWakeup::Kind::kBackup
                                      : VisitWakeup::Kind::kSelect;
    wakeup.generation = installed.generation;
    wakeup.value = payload.leaf;
    wakeup.terminal = terminal;
    wakeup.exact = installed.accepted && IsExact(payload.terminal);
    DeliverWakeup(wakeup);
  };
  wake(ticket->owner, true);
  for (const auto& wakeup : ticket->waiters) wake(wakeup, false);
}

bool VisitWorker::RestoreTicket(MaterializationTicketId ticket_id,
                                std::span<const uint8_t> record) {
  MaterializationResult restored;
  const auto ticket = tickets_.Close(ticket_id, [&](NodeKey key) {
    restored = engine_.RestoreFromRecord(key, ticket_id, record);
    return restored.accepted;
  });
  if (!ticket) return false;
  assert(ticket->owner.worker == index_);
  metrics_.nodes_restored.fetch_add(1);
  RaiseHighWater(metrics_.graph_size_high_water, engine_.GetGraphSize());
  // The node has its statistics back and nothing new to back up, so the owner
  // and the waiters all select below it.
  const auto wake = [&](VisitWakeup wakeup) {
    wakeup.kind = VisitWakeup::Kind::kRestored;
    wakeup.generation = restored.generation;
    DeliverWakeup(wakeup);
  };
  wake(ticket->owner);
  for (const auto& wakeup : ticket->waiters) wake(wakeup);
  return true;
}

void VisitWorker::CancelTicket(MaterializationTicketId ticket_id) {
  // Only called on stop, for evaluations that will never be computed. The
  // owner and waiters are still alive and must be woken to cancel.
  const auto ticket = tickets_.Close(ticket_id, [&](NodeKey key) {
    engine_.CancelMaterialization(key, ticket_id);
    return true;
  });
  assert(ticket->owner.worker == index_);
  const auto wake = [&](VisitWakeup wakeup) {
    wakeup.kind = VisitWakeup::Kind::kCancel;
    DeliverWakeup(wakeup);
  };
  wake(ticket->owner);
  for (const auto& wakeup : ticket->waiters) wake(wakeup);
}

void VisitWorker::CancelQueuedMaterializations() {
  for (auto* materializations : {&loads_, &evals_}) {
    for (const auto& materialization : *materializations) {
      CancelTicket(materialization.ticket);
    }
    materializations->clear();
  }
}

bool VisitWorker::CanSendJob() const {
  return jobs_in_flight_ < kMaxJobsInFlight &&
         items_in_flight_ < kMaxItemsInFlightBatches *
                                static_cast<size_t>(settings_.minibatch_size);
}

void VisitWorker::SetProducing(bool producing) {
  if (producing_ == producing) return;
  producing_ = producing;
  evaluator_.SetProducing(producing);
}

void VisitWorker::SendJobs() {
  if (IsStopping()) CancelQueuedMaterializations();
  while (CanSendJob()) {
    if (IsStopping()) CancelQueuedMaterializations();
    auto job = BuildJob();
    if (!job) break;
    ++jobs_in_flight_;
    items_in_flight_ += job->items();
    RaiseHighWater(metrics_.io_jobs_high_water,
                   session_jobs_in_flight_.fetch_add(1) + 1);
    if (job->kind == AsyncJob::Kind::kEval) {
      evaluator_.Push(std::move(job));
    } else {
      assert(has_store_);
      store_jobs_.Push(std::move(job));
    }
  }
}

std::unique_ptr<AsyncJob> VisitWorker::BuildJob() {
  // Small whole jobs leave room to collect more cache misses without exceeding
  // the backend batch. The items-in-flight cap bounds loads and evaluations.
  const size_t batch_size = settings_.minibatch_size;
  const size_t limit =
      std::min((batch_size + kJobsPerBatch - 1) / kJobsPerBatch,
               kMaxItemsInFlightBatches * batch_size - items_in_flight_);
  auto job = std::make_unique<AsyncJob>();
  job->worker = index_;
  job->queued_at = std::chrono::steady_clock::now();
  if (!loads_.empty()) {
    job->kind = AsyncJob::Kind::kLoad;
    MoveFront(loads_, job->materializations, limit);
  } else if (!evals_.empty()) {
    job->kind = AsyncJob::Kind::kEval;
    MoveFront(evals_, job->materializations, limit);
  } else {
    return nullptr;
  }
  return job;
}

void VisitWorker::FinishJob(std::unique_ptr<AsyncJob> job) {
  assert(jobs_in_flight_ > 0);
  for (auto& materialization : job->materializations) {
    switch (materialization.state) {
      case Materialization::State::kEvaluated:
        SortByPrior(materialization.payload);
        CompleteTicket(materialization.ticket,
                       std::move(materialization.payload));
        break;
      case Materialization::State::kStoreHit:
        metrics_.node_store_hits.fetch_add(1);
        if (!RestoreTicket(materialization.ticket, materialization.record)) {
          // Not a record for this policy or format: evaluate the node.
          metrics_.restores_rejected.fetch_add(1);
          materialization.record.clear();
          PrepareMaterialization(std::move(materialization));
        }
        break;
      case Materialization::State::kStoreMiss:
        metrics_.node_store_misses.fetch_add(1);
        PrepareMaterialization(std::move(materialization));
        break;
      case Materialization::State::kPending:
      case Materialization::State::kQueued:
        CancelTicket(materialization.ticket);
        break;
    }
  }
  --jobs_in_flight_;
  assert(items_in_flight_ >= job->items());
  items_in_flight_ -= job->items();
  // Decrement last: publication is part of the job.
  session_jobs_in_flight_.fetch_sub(1);
}

}  // namespace lczero::lc5
