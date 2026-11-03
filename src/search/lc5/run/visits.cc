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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "search/lc5/run/search_run.h"

namespace lczero::lc5 {

bool SearchRun::AdmissionClosed() const {
  return Stopping() || (node_limit_ && admitted_ >= *node_limit_);
}

std::optional<VisitId> SearchRun::Allocate(const VisitOrigin& origin) {
  // Allocate never holds pool -> slot.
  if (AdmissionClosed() || visits_.active() >= ActiveLimit()) {
    return std::nullopt;
  }
  const auto id = visits_.Allocate(origin);
  if (!id) return std::nullopt;
  ++admitted_;
  metrics_.visits_admitted.fetch_add(1);
  RaiseHighWater(metrics_.active_visits_high_water, visits_.active());
  return id;
}

size_t SearchRun::ActiveLimit() const {
  if (settings_.active_visits_ratio <= 0.0f) return visits_.capacity();
  const double tree = static_cast<double>(root_visits_at_start_ +
                                          metrics_.visits_completed.load());
  return std::min(visits_.capacity(),
                  kMinActiveVisits + static_cast<size_t>(
                                         tree * settings_.active_visits_ratio));
}

bool SearchRun::Admit(const VisitOrigin& origin) {
  size_t owner;
  std::vector<Continuation> admitted;
  {
    absl::MutexLock lock(admission_mutex_);
    owner = next_owner_++ % workers_.size();
    const auto id = Allocate(origin);
    if (!id) return false;
    admitted.push_back({.id = *id, .worker = owner});
  }
  // The allocated visit keeps active() nonzero until this is consumed, so stop
  // cannot declare a drain before the mailbox publication.
  SendContinuations(owner, admitted);
  return true;
}

void SearchRun::AdmitMore(Worker& worker) {
  absl::MutexLock lock(admission_mutex_);
  worker.admission_waiting = false;
  // A modest per-worker share prevents one worker from monopolizing the pool.
  const size_t share =
      (visits_.capacity() + workers_.size() - 1) / workers_.size();
  for (size_t i = 0; i < kWorkChunk && worker.active.size() < share; ++i) {
    const auto id = Allocate(root_);
    if (!id) {
      // The pool is full. Registration shares admission_mutex_ with the
      // capacity check in WakeAdmissionWaiters, so it cannot miss a release.
      worker.admission_waiting = !AdmissionClosed();
      break;
    }
    worker.active.insert(id->value);
    worker.runnable.push_back(*id);
  }
  RaiseHighWater(metrics_.visits_ready_high_water, worker.runnable.size());
}

void SearchRun::WakeAdmissionWaiters() {
  std::vector<size_t> wake;
  {
    absl::MutexLock lock(admission_mutex_);
    if (AdmissionClosed()) return;
    const size_t limit = ActiveLimit();
    const size_t active = visits_.active();
    size_t available = limit > active ? limit - active : 0;
    for (size_t n = 0; n < workers_.size() && available > 0; ++n) {
      const size_t index = (next_admission_waiter_ + n) % workers_.size();
      if (!workers_[index]->admission_waiting) continue;
      workers_[index]->admission_waiting = false;
      wake.push_back(index);
      --available;
    }
    if (!wake.empty()) {
      next_admission_waiter_ = (wake.back() + 1) % workers_.size();
    }
  }
  // A competing allocator may win the capacity; the woken worker then simply
  // re-registers on its next attempt.
  for (size_t index : wake) {
    auto& mailbox = workers_[index]->mailbox;
    absl::MutexLock lock(mailbox.mutex);
    mailbox.admission_ready = true;
  }
}

void SearchRun::SendContinuations(size_t worker,
                                  std::vector<Continuation>& batch) {
  if (batch.empty()) return;
  {
    auto& mailbox = workers_[worker]->mailbox;
    absl::MutexLock lock(mailbox.mutex);
    mailbox.continuations.insert(mailbox.continuations.end(), batch.begin(),
                                 batch.end());
  }
  batch.clear();
  metrics_.mailbox_notifications.fetch_add(1);
}

void SearchRun::Deliver(Worker& worker, const Continuation& continuation) {
  if (continuation.worker == worker.index) {
    Resume(worker, continuation);
  } else {
    worker.outgoing[continuation.worker].push_back(continuation);
  }
}

void SearchRun::Resume(Worker& worker, const Continuation& event) {
  VisitPool::Slot* slot = visits_.Lookup(event.id);
  if (!slot) return;
  absl::MutexLock lock(slot->mutex);
  if (!visits_.IsLiveLocked(*slot, event.id)) return;
  Visit& visit = slot->visit;
  if (event.kind == Continuation::Kind::kAdmit) {
    worker.active.insert(event.id.value);
  } else {
    if (visit.state != VisitState::kWaitingMaterialization ||
        visit.waiting_ticket != event.ticket) {
      return;
    }
    visit.waiting_ticket = 0;
    // On stop, visits finish only when their evaluation already exists.
    if (event.kind == Continuation::Kind::kCancel ||
        (Stopping() && (event.kind == Continuation::Kind::kSelect ||
                        event.kind == Continuation::Kind::kRestored))) {
      Cancel(worker, *slot);
      return;
    }
    visit.path.back().generation = event.generation;
    visit.path.back().selected_move.reset();
    visit.result = event.value;
    visit.result_exact = event.exact;
    if (event.kind == Continuation::Kind::kBackup ||
        event.kind == Continuation::Kind::kBackupAbove) {
      visit.leaf_counted = event.kind == Continuation::Kind::kBackupAbove;
      visit.state = VisitState::kReadyBackup;
    } else if (event.kind == Continuation::Kind::kRestored) {
      visit.state = VisitState::kReadySelect;
      metrics_.visits_resumed.fetch_add(1);
    } else {
      switch (engine_->OnResume(visit, metrics_)) {
        case VisitEngine::ResumeAction::kRestart:
          Restart(*slot);
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
    if (event.terminal) metrics_.terminal_visits.fetch_add(1);
  }
  worker.runnable.push_back(event.id);
}

void SearchRun::VisitWorker(Worker& worker) {
  worker.outgoing.resize(workers_.size());
  for (;;) {
    ProcessMailbox(worker);
    if (Stopping()) {
      CancelVisits(worker);
    } else {
      AdvanceRunnable(worker, kWorkChunk);
      AdmitMore(worker);
    }
    SubmitJobs(worker);
    for (size_t i = 0; i < workers_.size(); ++i) {
      SendContinuations(i, worker.outgoing[i]);
    }
    PublishWorkerState(worker);
    if (shutdown_.load()) break;
    // Keep going locally without a wake or controller handoff. Suspended
    // visits and exhausted credits wait for whole completions.
    if (HasLocalWork(worker)) continue;
    // Senders evaluate this on unlock; `cancelled` is written only by this
    // worker, never while it waits here.
    auto& mailbox = worker.mailbox;
    absl::MutexLock lock(mailbox.mutex);
    const auto ready = [&] {
      mailbox.mutex.AssertReaderHeld();
      return shutdown_.load() || mailbox.admission_ready ||
             !mailbox.continuations.empty() || !mailbox.completions.empty() ||
             (!worker.cancelled && Stopping());
    };
    mailbox.mutex.Await(absl::Condition(&ready));
  }
}

void SearchRun::ProcessMailbox(Worker& worker) {
  std::vector<Continuation> continuations;
  std::vector<std::unique_ptr<Job>> completions;
  bool admission_ready;
  {
    absl::MutexLock lock(worker.mailbox.mutex);
    continuations.swap(worker.mailbox.continuations);
    completions.swap(worker.mailbox.completions);
    admission_ready = std::exchange(worker.mailbox.admission_ready, false);
  }
  if (admission_ready || !continuations.empty() || !completions.empty()) {
    // Actual incoming work can enable preparation or refill. This transition
    // disables starvation and needs no broadcast.
    SetProducing(worker, true);
    if (worker.drained.exchange(false)) worker.controller_event = true;
  }
  for (const auto& event : continuations) Resume(worker, event);
  for (auto& job : completions) ConsumeJob(worker, std::move(job));
}

void SearchRun::AdvanceRunnable(Worker& worker, size_t limit) {
  for (size_t i = 0; i < limit && !worker.runnable.empty(); ++i) {
    const VisitId id = worker.runnable.front();
    worker.runnable.pop_front();
    AdvanceVisit(worker, id);
  }
}

void SearchRun::CancelVisits(Worker& worker) {
  // Only the owner ever cancels or mutates a visit. Epochs make later ticket
  // completions harmless even after the slot has been reused. Ready backups
  // finish; waiting visits stay until their ticket completes or is cancelled,
  // and come back through Resume.
  if (!worker.cancelled) {
    const std::vector<uint64_t> active(worker.active.begin(),
                                       worker.active.end());
    for (uint64_t value : active) AdvanceVisit(worker, VisitId{value});
    worker.runnable.clear();
    worker.cancelled = true;
  }
  // External admissions published just before stop still need cancellation.
  AdvanceRunnable(worker, std::numeric_limits<size_t>::max());
}

void SearchRun::PublishWorkerState(Worker& worker) {
  const bool drained = worker.cancelled && worker.active.empty() &&
                       worker.outstanding == 0 && !worker.HasQueuedIo();
  if (worker.drained.exchange(drained) != drained) {
    worker.controller_event = true;
  }
  if (std::exchange(worker.released_capacity, false)) WakeAdmissionWaiters();
  if (std::exchange(worker.controller_event, false)) NotifyController();
  SetProducing(worker, HasJobCredit(worker) &&
                           (!worker.runnable.empty() || worker.HasQueuedIo()));
}

bool SearchRun::HasLocalWork(const Worker& worker) const {
  return (!Stopping() && !worker.runnable.empty()) ||
         (HasJobCredit(worker) && worker.HasQueuedIo());
}

void SearchRun::AdvanceVisit(Worker& worker, VisitId id) {
  VisitPool::Slot* slot = visits_.Lookup(id);
  if (!slot) return;
  absl::MutexLock lock(slot->mutex);
  if (!visits_.IsLiveLocked(*slot, id)) return;
  Visit& visit = slot->visit;
  if (visit.state == VisitState::kReadyBackup) {
    Backup(worker, *slot);
    return;
  }
  if (visit.state == VisitState::kWaitingMaterialization) return;
  if (Stopping()) {
    Cancel(worker, *slot);
    return;
  }
  if (visit.state != VisitState::kReadySelect) return;
  for (;;) {
    if (Stopping()) {
      Cancel(worker, *slot);
      return;
    }
    if (engine_->Descend(visit, metrics_) ==
        VisitEngine::DescendResult::kValuedLeaf) {
      Backup(worker, *slot);
      return;
    }
    if (SuspendForMaterialization(worker, *slot)) return;
  }
}

void SearchRun::Backup(Worker& worker, VisitPool::Slot& slot) {
  Visit& visit = slot.visit;
  visit.state = VisitState::kBackingUp;
  engine_->Backup(visit, metrics_);
  RaiseHighWater(metrics_.max_depth, visit.path.size());
  FinishVisit(worker, slot, true);
}

void SearchRun::Cancel(Worker& worker, VisitPool::Slot& slot) {
  engine_->Cancel(slot.visit, metrics_);
  FinishVisit(worker, slot, false);
}

void SearchRun::Restart(VisitPool::Slot& slot) {
  Visit& visit = slot.visit;
  engine_->Cancel(visit, metrics_);
  visit.history = visit.origin.history;
  visit.current_key = visit.origin.key;
  visit.path = visit.origin.backup_prefix;
  visit.two_fold_origin.reset();
  visit.state = VisitState::kReadySelect;
  metrics_.visits_restarted.fetch_add(1);
}

void SearchRun::FinishVisit(Worker& worker, VisitPool::Slot& slot,
                            bool completed) {
  const VisitId id = slot.visit.id;
  if (completed) {
    const uint64_t count = metrics_.visits_completed.fetch_add(1) + 1;
    if (node_limit_ == count) worker.controller_event = true;
  } else {
    metrics_.visits_cancelled.fetch_add(1);
  }
  worker.active.erase(id.value);
  visits_.ReleaseLocked(slot, id);
  // Admission and the controller are signalled later, outside the slot lock.
  worker.released_capacity = true;
}

}  // namespace lczero::lc5
