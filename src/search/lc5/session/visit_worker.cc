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
#include <utility>

#include "search/lc5/graph/prior.h"
#include "utils/math.h"

namespace lczero::lc5 {
namespace {

// Returns a terminal evaluation, or a nonterminal one with the legal moves.
NodeEval DetectTerminal(const PositionHistory& history) {
  NodeEval eval;
  const Position& position = history.Last();
  const ChessBoard& board = position.GetBoard();
  const auto legal = board.GenerateLegalMoves();
  if (legal.empty()) {
    eval.terminal = board.IsUnderCheck() ? TerminalKind::kCheckmate
                                         : TerminalKind::kStalemate;
  } else if (!board.HasMatingMaterial()) {
    eval.terminal = TerminalKind::kInsufficientMaterial;
  } else if (position.GetRule50Ply() >= 100) {
    eval.terminal = TerminalKind::kRule50;
  } else if (position.GetRepetitions() >= 2) {
    eval.terminal = TerminalKind::kRepetition;
  } else {
    eval.moves.assign(legal.begin(), legal.end());
    return eval;
  }
  eval.leaf = TerminalValue(eval.terminal);
  return eval;
}

// Rounds priors to the precision the graph stores them in, then orders edges
// by descending prior, ties by move encoding. Sorting the rounded values keeps
// the order the graph relies on even when rounding makes two priors equal.
void SortByPrior(NodeEval& eval) {
  std::vector<std::pair<float, Move>> edges;
  edges.reserve(eval.moves.size());
  for (size_t i = 0; i < eval.moves.size(); ++i) {
    edges.emplace_back(PackedPrior(eval.priors[i]), eval.moves[i]);
  }
  std::ranges::sort(edges, [](const auto& a, const auto& b) {
    if (a.first != b.first) return a.first > b.first;
    return a.second.raw_data() < b.second.raw_data();
  });
  for (size_t i = 0; i < edges.size(); ++i) {
    std::tie(eval.priors[i], eval.moves[i]) = edges[i];
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

VisitWorker::VisitWorker(size_t worker_id, const Context& context)
    : worker_id_(worker_id),
      context_(context),
      inbox_(context.inboxes[worker_id]),
      outgoing_wakeups_(context.inboxes.size()) {}

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
      CeilDiv(context_.visits.capacity(), context_.inboxes.size());
  const size_t count =
      std::min(kVisitsPerPass, share - std::min(share, active_visits_));
  const std::vector<VisitId> admitted =
      context_.limiter.AdmitVisits(worker_id_, context_.root, count);
  active_visits_ += admitted.size();
  ready_visits_.insert(ready_visits_.end(), admitted.begin(), admitted.end());
  RaiseHighWater(context_.metrics.visits_ready_high_water,
                 ready_visits_.size());
}

void VisitWorker::WakeAdmissionWaiters() {
  for (size_t index : context_.limiter.TakeWorkersToWake()) {
    auto& inbox = context_.inboxes[index];
    absl::MutexLock lock(inbox.mutex);
    inbox.admission_ready = true;
  }
}

void VisitWorker::SendVisitWakeups(size_t to, std::vector<VisitWakeup>& batch) {
  if (batch.empty()) return;
  {
    auto& inbox = context_.inboxes[to];
    absl::MutexLock lock(inbox.mutex);
    inbox.wakeups.insert(inbox.wakeups.end(), batch.begin(), batch.end());
  }
  batch.clear();
  context_.metrics.inbox_notifications.fetch_add(1);
}

void VisitWorker::DeliverWakeup(const VisitWakeup& wakeup) {
  (wakeup.worker == worker_id_ ? local_wakeups_
                               : outgoing_wakeups_[wakeup.worker])
      .push_back(wakeup);
}

void VisitWorker::ResumeVisit(const VisitWakeup& wakeup) {
  // Each suspended visit waits on one materialization and is woken once.
  Visit& visit = context_.visits.Get(wakeup.id);
  assert(visit.state == VisitState::kWaitingMaterialization);
  // On stop, only visits that already have a leaf value to back up may finish.
  if (wakeup.kind == VisitWakeup::Kind::kCancel ||
      (IsStopping() && (wakeup.kind == VisitWakeup::Kind::kDescend ||
                        wakeup.kind == VisitWakeup::Kind::kRestored))) {
    CancelVisit(visit);
    return;
  }
  visit.path.back().selected_move.reset();
  if (wakeup.terminal) context_.metrics.terminal_visits.fetch_add(1);
  switch (wakeup.kind) {
    case VisitWakeup::Kind::kBackup:
      visit.result = wakeup.value;
      visit.result_exact = wakeup.exact;
      BackupVisit(visit);
      return;
    case VisitWakeup::Kind::kRestored:
      // Node restored from store with its prior stats; descend below it.
      visit.state = VisitState::kReadyDescend;
      context_.metrics.visits_resumed.fetch_add(1);
      break;
    case VisitWakeup::Kind::kDescend:
      // Waiter on a newly-evaluated nonterminal leaf: reselect at the parent
      // and either back up a repetition or continue descending.
      switch (context_.engine.ResumeVisit(visit, context_.metrics)) {
        case VisitEngine::ResumeAction::kBackup:
          BackupVisit(visit);  // Reselected into 2-fold.
          return;
        case VisitEngine::ResumeAction::kDescend:
          visit.state = VisitState::kReadyDescend;
          context_.metrics.visits_resumed.fetch_add(1);
          break;
      }
      break;
    case VisitWakeup::Kind::kCancel:
      assert(false);
      break;
  }
  ready_visits_.push_back(wakeup.id);  // Queue for next AdvanceVisit pass.
}

void VisitWorker::Run() {
  for (;;) {
    ProcessInbox();  // Ingest finished store/eval jobs and visit wakeups.
    const bool stopping = IsStopping();
    if (stopping) visits_cancelled_ = true;
    // Step ready visits (up to kVisitsPerPass, or drain all when stopping).
    for (size_t i = 0;
         (stopping || i < kVisitsPerPass) && !ready_visits_.empty(); ++i) {
      const VisitId id = ready_visits_.front();
      ready_visits_.pop_front();
      AdvanceVisit(id);
    }
    if (!stopping) AdmitNewRootVisits();  // Refill with new root visits.
    SendJobs();  // Dispatch queued store loads and NN evals as AsyncJobs.
    for (size_t i = 0; i < context_.inboxes.size(); ++i) {
      SendVisitWakeups(i, outgoing_wakeups_[i]);  // Wake peers' waiting visits.
    }
    FlushNotifications();  // Signal controller, admission waiters, evaluator.
    if (shutdown_) break;
    if (HasReadyWork()) continue;  // Start next pass without locking inbox_.
    // Sleep until new inbox work, admission capacity, or stop/shutdown arrives.
    absl::MutexLock lock(inbox_.mutex);
    const auto ready = [&] {
      inbox_.mutex.AssertReaderHeld();
      return inbox_.shutdown || inbox_.admission_ready ||
             !inbox_.wakeups.empty() || !inbox_.completions.empty() ||
             (!visits_cancelled_ && IsStopping());
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
    // Evaluator needs to know when workers are stuck to flush partial batch.
    ReportProducingStatus(true);
    // Mark not drained while processing swapped-out inbox items.
    if (drained_.exchange(false)) notify_controller_ = true;
  }
  for (const auto& wakeup : wakeups) ResumeVisit(wakeup);
  for (auto& job : completions) FinishJob(std::move(job));
  for (const auto& wakeup : local_wakeups_) ResumeVisit(wakeup);
  local_wakeups_.clear();
}

void VisitWorker::FlushNotifications() {
  const bool drained = visits_cancelled_ && active_visits_ == 0 &&
                       jobs_in_flight_ == 0 && !HasQueuedIo() &&
                       local_wakeups_.empty();
  if (drained_.exchange(drained) != drained) notify_controller_ = true;
  if (std::exchange(notify_admission_waiters_, false)) WakeAdmissionWaiters();
  if (std::exchange(notify_controller_, false)) {
    context_.controller_signal.Notify();
  }
  ReportProducingStatus(CanSendJob() && HasReadyWork());
}

bool VisitWorker::HasReadyWork() const {
  return !local_wakeups_.empty() || (!IsStopping() && !ready_visits_.empty()) ||
         (CanSendJob() && HasQueuedIo());
}

void VisitWorker::AdvanceVisit(VisitId id) {
  // Only this worker releases its visits, and never one that is ready.
  Visit& visit = context_.visits.Get(id);
  if (IsStopping()) {
    CancelVisit(visit);
    return;
  }
  assert(visit.state == VisitState::kReadyDescend);
  for (;;) {
    if (IsStopping()) {
      CancelVisit(visit);
      return;
    }
    if (context_.engine.DescendVisit(visit, context_.metrics) ==
        VisitEngine::DescendResult::kValuedLeaf) {
      BackupVisit(visit);
      return;
    }
    if (SuspendVisit(visit)) return;
  }
}

void VisitWorker::BackupVisit(Visit& visit) {
  context_.engine.BackupVisit(visit, context_.metrics);
  RaiseHighWater(context_.metrics.max_depth, visit.path.size());
  FinishVisit(visit, true);
}

void VisitWorker::CancelVisit(Visit& visit) {
  context_.engine.CancelVisit(visit);
  FinishVisit(visit, false);
}

void VisitWorker::FinishVisit(Visit& visit, bool completed) {
  if (completed) {
    const uint64_t count = context_.metrics.visits_completed.fetch_add(1) + 1;
    if (count >= context_.limiter.node_limit()) {
      context_.limiter.Stop();
      notify_controller_ = true;
    } else if (count == 1) {
      notify_controller_ = true;
    }
  } else {
    context_.metrics.visits_cancelled.fetch_add(1);
  }
  --active_visits_;
  context_.visits.Release(visit.id);
  // Admission and the controller are signalled once per pass, not per visit.
  notify_admission_waiters_ = true;
}

//////////////////////////////////////////////////////////////////////////////
// Materialization: waiters, materializations and jobs.
//////////////////////////////////////////////////////////////////////////////

bool VisitWorker::SuspendVisit(Visit& visit) {
  const auto joined = context_.materializations.Join(
      visit.current_key, {.id = visit.id, .worker = worker_id_},
      [&] { return !context_.engine.ContainsNode(visit.current_key); });
  if (!joined) return false;
  const bool owner = joined->owner;
  if (owner) {
    context_.metrics.materializations_started.fetch_add(1);
  } else {
    context_.metrics.materialization_waiters.fetch_add(1);
    RaiseHighWater(context_.metrics.maximum_waiters_per_materialization,
                   joined->waiters);
  }
  // The owner's leaf is counted at install (update_node = false skips it on
  // backup); waiters keep update_node = true in case they back up a terminal
  // leaf.
  visit.path.back().update_node = !owner;
  visit.state = VisitState::kWaitingMaterialization;
  context_.metrics.visits_suspended.fetch_add(1);
  if (owner) {
    MaybeRequestEviction();
    Materialization materialization{
        .id = joined->id, .key = visit.current_key, .history = visit.history};
    // A stored node may hold statistics, even if terminal: look it up first.
    if (context_.engine.HasStore()) {
      queued_loads_.push_back(std::move(materialization));
    } else {
      PrepareMaterialization(std::move(materialization));
    }
  }
  return true;
}

void VisitWorker::PrepareMaterialization(Materialization materialization) {
  if (IsStopping()) {
    CancelMaterialization(materialization.id);
    return;
  }
  materialization.eval = DetectTerminal(materialization.history);
  if (materialization.eval.terminal != TerminalKind::kNonTerminal) {
    CompleteMaterialization(materialization.id,
                            std::move(materialization.eval));
    return;
  }
  materialization.eval.priors.resize(materialization.eval.moves.size());
  materialization.state = Materialization::State::kPending;
  context_.metrics.eval_requests.fetch_add(1);
  queued_evals_.push_back(std::move(materialization));
}

void VisitWorker::CompleteMaterialization(MaterializationId materialization_id,
                                          NodeEval eval) {
  bool accepted = false;
  bool is_root = false;
  const auto waiters =
      context_.materializations.Close(materialization_id, [&](NodeKey key) {
        is_root = key == context_.root.key;
        accepted = context_.engine.CompleteMaterialization(key, eval);
        return true;
      });
  if (is_root && accepted) notify_controller_ = true;
  assert(waiters->owner.worker == worker_id_);
  if (accepted) {
    context_.metrics.graph_nodes_created.fetch_add(1);
    RaiseHighWater(context_.metrics.graph_size_high_water,
                   context_.engine.GetGraphSize());
    MaybeRequestEviction();
  } else {
    context_.metrics.publications_rejected.fetch_add(1);
  }
  // The owner backs up the new leaf value; waiters descend below the node
  // unless it is terminal, in which case they also back up. A rejected
  // publication resumes descending only.
  const bool terminal = accepted && eval.terminal != TerminalKind::kNonTerminal;
  const auto wake = [&](VisitWakeup wakeup, bool owner) {
    wakeup.kind = accepted && (owner || terminal) ? VisitWakeup::Kind::kBackup
                                                  : VisitWakeup::Kind::kDescend;
    wakeup.value = eval.leaf;
    wakeup.terminal = terminal;
    wakeup.exact = accepted && IsExact(eval.terminal);
    DeliverWakeup(wakeup);
  };
  // Wake the owner first so ProcessInbox backs it up before any waiter resumes.
  wake(waiters->owner, true);
  for (const auto& wakeup : waiters->waiters) wake(wakeup, false);
}

bool VisitWorker::RestoreMaterialization(MaterializationId materialization_id,
                                         std::span<const uint8_t> record) {
  bool is_root = false;
  const auto waiters =
      context_.materializations.Close(materialization_id, [&](NodeKey key) {
        is_root = key == context_.root.key;
        return context_.engine.RestoreFromRecord(key, record);
      });
  if (!waiters) return false;
  if (is_root) notify_controller_ = true;
  assert(waiters->owner.worker == worker_id_);
  context_.metrics.nodes_restored.fetch_add(1);
  RaiseHighWater(context_.metrics.graph_size_high_water,
                 context_.engine.GetGraphSize());
  MaybeRequestEviction();
  // The node has its statistics back and nothing new to back up, so the owner
  // and the waiters all select below it.
  const auto wake = [&](VisitWakeup wakeup) {
    wakeup.kind = VisitWakeup::Kind::kRestored;
    DeliverWakeup(wakeup);
  };
  wake(waiters->owner);
  for (const auto& wakeup : waiters->waiters) wake(wakeup);
  return true;
}

void VisitWorker::CancelMaterialization(MaterializationId materialization_id) {
  // Only called on stop, for evaluations that will never be computed. The
  // owner and waiters are still alive and must be woken to cancel.
  const auto waiters = context_.materializations.Close(
      materialization_id, [](NodeKey) { return true; });
  assert(waiters->owner.worker == worker_id_);
  const auto wake = [&](VisitWakeup wakeup) {
    wakeup.kind = VisitWakeup::Kind::kCancel;
    DeliverWakeup(wakeup);
  };
  wake(waiters->owner);
  for (const auto& wakeup : waiters->waiters) wake(wakeup);
}

void VisitWorker::CancelQueuedMaterializations() {
  for (auto* materializations : {&queued_loads_, &queued_evals_}) {
    for (const auto& materialization : *materializations) {
      CancelMaterialization(materialization.id);
    }
    materializations->clear();
  }
}

void VisitWorker::MaybeRequestEviction() {
  // Wakes the controller once the graph outgrows its limit, so eviction keeps
  // up with fast searches between its periodic checks.
  if (context_.engine.HasStore() && context_.settings.graph_node_limit > 0 &&
      context_.engine.GetGraphSize() > context_.settings.graph_node_limit &&
      context_.controller_signal.RequestEviction()) {
    notify_controller_ = true;
  }
}

bool VisitWorker::CanSendJob() const {
  return jobs_in_flight_ < kMaxJobsInFlight &&
         materializations_in_flight_ <
             kMaxItemsInFlightBatches *
                 static_cast<size_t>(context_.settings.minibatch_size);
}

void VisitWorker::ReportProducingStatus(bool producing) {
  if (producing_ == producing) return;
  producing_ = producing;
  if (producing) {
    context_.evaluator.OnProducerActive();
  } else {
    context_.evaluator.OnProducerBlocked();
  }
}

void VisitWorker::SendJobs() {
  if (IsStopping()) CancelQueuedMaterializations();
  while (CanSendJob()) {
    if (IsStopping()) CancelQueuedMaterializations();
    auto job = BuildJob();
    if (!job) break;
    ++jobs_in_flight_;
    materializations_in_flight_ += job->items();
    RaiseHighWater(context_.metrics.io_jobs_high_water,
                   context_.jobs_in_flight.fetch_add(1) + 1);
    if (job->kind == AsyncJob::Kind::kEval) {
      context_.evaluator.Push(std::move(job));
    } else {
      assert(context_.engine.HasStore());
      context_.store_jobs.Push(std::move(job));
    }
  }
}

std::unique_ptr<AsyncJob> VisitWorker::BuildJob() {
  // Small whole jobs leave room to collect more cache misses without exceeding
  // the backend batch. The materializations-in-flight cap bounds loads and
  // evaluations.
  const size_t batch_size = context_.settings.minibatch_size;
  const size_t limit = std::min(
      CeilDiv(batch_size, kJobsPerBatch),
      kMaxItemsInFlightBatches * batch_size - materializations_in_flight_);
  auto job = std::make_unique<AsyncJob>();
  job->worker = worker_id_;
  job->queued_at = std::chrono::steady_clock::now();
  if (!queued_loads_.empty()) {
    job->kind = AsyncJob::Kind::kLoad;
    MoveFront(queued_loads_, job->materializations, limit);
  } else if (!queued_evals_.empty()) {
    job->kind = AsyncJob::Kind::kEval;
    MoveFront(queued_evals_, job->materializations, limit);
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
        SortByPrior(materialization.eval);
        CompleteMaterialization(materialization.id,
                                std::move(materialization.eval));
        break;
      case Materialization::State::kStoreHit:
        context_.metrics.node_store_hits.fetch_add(1);
        if (!RestoreMaterialization(materialization.id,
                                    materialization.record)) {
          // Rejected record: evaluate the node instead.
          context_.metrics.restores_rejected.fetch_add(1);
          materialization.record.clear();
          PrepareMaterialization(std::move(materialization));
        }
        break;
      case Materialization::State::kStoreMiss:
        context_.metrics.node_store_misses.fetch_add(1);
        PrepareMaterialization(std::move(materialization));
        break;
      case Materialization::State::kPending:
      case Materialization::State::kQueued:
        CancelMaterialization(materialization.id);
        break;
    }
  }
  --jobs_in_flight_;
  assert(materializations_in_flight_ >= job->items());
  materializations_in_flight_ -= job->items();
  // Decrement last: publication is part of the job.
  context_.jobs_in_flight.fetch_sub(1);
}

}  // namespace lczero::lc5
