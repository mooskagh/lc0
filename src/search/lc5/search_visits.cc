#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "search/lc5/search.h"

namespace lczero::lc5 {

std::optional<VisitId> SearchRun::Allocate(const VisitOrigin& origin,
                                           size_t worker) {
  // admission_mutex_ is held by the caller. Allocate never holds pool -> slot.
  if (stop_mode_.load() != StopMode::kRunning ||
      (node_limit_ && admitted_ >= *node_limit_))
    return std::nullopt;
  auto id = visits_.Allocate(origin, worker);
  if (!id) return std::nullopt;
  ++admitted_;
  metrics_.visits_admitted.fetch_add(1);
  detail::RaiseHighWater(metrics_.active_visits_high_water,
                         static_cast<uint64_t>(visits_.active()));
  return id;
}

bool SearchRun::Admit(const VisitOrigin& origin) {
  std::vector<std::vector<Continuation>> groups(workers_.size());
  {
    std::lock_guard lock(admission_mutex_);
    const size_t owner = next_owner_++ % workers_.size();
    auto id = Allocate(origin, owner);
    if (!id) return false;
    groups[owner].push_back({.id = *id, .worker = owner, .value = {}});
  }
  // An allocated visit keeps active() nonzero until this event is consumed,
  // so stop cannot declare a drain in the interval before mailbox publication.
  SendContinuations(std::move(groups));
  return true;
}

void SearchRun::AdmitMore(Worker& worker) {
  std::lock_guard lock(admission_mutex_);
  worker.admission_waiting = false;
  // A modest per-worker share prevents one worker from monopolizing the pool.
  const size_t share =
      (visits_.capacity() + workers_.size() - 1) / workers_.size();
  for (size_t i = 0; i < kWorkChunk && worker.active.size() < share; ++i) {
    if (stop_mode_.load() != StopMode::kRunning ||
        (node_limit_ && admitted_ >= *node_limit_))
      break;
    auto id = Allocate(root_, worker.index);
    if (!id) {
      // Allocation and registration share admission_mutex_ with the capacity
      // check in WakeAdmissionWaiters. A release is signalled after its chunk,
      // outside slot locks, so registration cannot miss available capacity.
      worker.admission_waiting = true;
      break;
    }
    worker.active.insert(id->value);
    worker.runnable.push_back(*id);
  }
  detail::RaiseHighWater(metrics_.visits_ready_high_water,
                         static_cast<uint64_t>(worker.runnable.size()));
}

void SearchRun::WakeAdmissionWaiters() {
  std::vector<size_t> wake;
  {
    std::lock_guard lock(admission_mutex_);
    if (stop_mode_.load() != StopMode::kRunning ||
        (node_limit_ && admitted_ >= *node_limit_))
      return;
    size_t available = visits_.capacity() - visits_.active();
    const size_t start = next_admission_waiter_;
    for (size_t n = 0; n < workers_.size() && available > 0; ++n) {
      const size_t index = (start + n) % workers_.size();
      if (!workers_[index]->admission_waiting) continue;
      workers_[index]->admission_waiting = false;
      wake.push_back(index);
      next_admission_waiter_ = (index + 1) % workers_.size();
      --available;
    }
  }
  // Targeted, once per releasing chunk; no graph/slot/admission lock crosses
  // the mailbox transfer. A competing allocator may win, in which case the
  // notified worker simply re-registers on its next allocation attempt.
  for (size_t index : wake) {
    auto& mailbox = workers_[index]->mailbox;
    {
      std::lock_guard lock(mailbox.mutex);
      mailbox.admission_ready = true;
    }
    mailbox.cv.notify_one();
  }
}

void SearchRun::SendContinuations(
    std::vector<std::vector<Continuation>> groups) {
  for (size_t i = 0; i < groups.size(); ++i) {
    if (groups[i].empty()) continue;
    auto& mailbox = workers_[i]->mailbox;
    {
      std::lock_guard lock(mailbox.mutex);
      mailbox.continuations.insert(mailbox.continuations.end(),
                                   groups[i].begin(), groups[i].end());
    }
    metrics_.mailbox_notifications.fetch_add(1);
    mailbox.cv.notify_one();
  }
}

void SearchRun::Resume(Worker& worker, const Continuation& event) {
  auto* slot = visits_.Lookup(event.id);
  if (!slot) return;
  std::lock_guard lock(slot->mutex);
  if (slot->epoch.load() != event.id.epoch() ||
      slot->visit.state == VisitState::kFree)
    return;
  Visit& visit = slot->visit;
  assert(visit.worker == worker.index);
  if (event.ticket == 0) {
    worker.active.insert(event.id.value);
  } else {
    if (visit.state != VisitState::kWaitingMaterialization ||
        visit.waiting_ticket != event.ticket)
      return;
    visit.waiting_ticket = 0;
    visit.path.back().generation = event.generation;
    visit.path.back().selected_move.reset();
    visit.state =
        event.backup ? VisitState::kReadyBackup : VisitState::kReadySelect;
    visit.result = event.value;
    if (event.terminal) metrics_.terminal_visits.fetch_add(1);
    if (!event.backup) metrics_.visits_resumed.fetch_add(1);
  }
  worker.runnable.push_back(event.id);
}

void SearchRun::VisitWorker(Worker& worker) {
  worker.outgoing.resize(workers_.size());
  for (;;) {
    std::vector<Continuation> continuations;
    std::vector<std::unique_ptr<Job>> completions;
    bool admission_ready;
    {
      std::lock_guard lock(worker.mailbox.mutex);
      continuations.swap(worker.mailbox.continuations);
      completions.swap(worker.mailbox.completions);
      admission_ready = worker.mailbox.admission_ready;
      worker.mailbox.admission_ready = false;
    }
    if (admission_ready || !continuations.empty() || !completions.empty()) {
      // Actual incoming work, not an idle timeout, can enable preparation or
      // refill. This transition disables starvation and needs no broadcast.
      SetProducing(worker, true);
      if (worker.drained.exchange(false)) worker.controller_event = true;
    }
    for (const auto& event : continuations) Resume(worker, event);
    for (auto& job : completions) ConsumeJob(worker, std::move(job));

    if (stop_mode_.load() != StopMode::kRunning) {
      // Only the owner ever cancels or mutates a visit. Epochs make later
      // ticket completions harmless even after the slot has been reused.
      if (!worker.cancelled) {
        std::vector<uint64_t> active(worker.active.begin(),
                                     worker.active.end());
        for (uint64_t value : active) AdvanceVisit(worker, VisitId{value});
        worker.runnable.clear();
        worker.cancelled = true;
      }
      // External admissions published just before stop still need cancellation.
      while (!worker.runnable.empty()) {
        auto id = worker.runnable.front();
        worker.runnable.pop_front();
        AdvanceVisit(worker, id);
      }
    } else if (worker.persistence.size() <
               static_cast<size_t>(settings_.minibatch_size)) {
      AdmitMore(worker);
      for (size_t i = 0; i < kWorkChunk && !worker.runnable.empty(); ++i) {
        auto id = worker.runnable.front();
        worker.runnable.pop_front();
        AdvanceVisit(worker, id);
      }
    }
    if (stop_mode_.load() == StopMode::kRunning &&
        worker.persistence.size() <
            static_cast<size_t>(settings_.minibatch_size))
      AdmitMore(worker);
    SubmitJobs(worker);
    SendContinuations(std::move(worker.outgoing));
    worker.outgoing.clear();
    worker.outgoing.resize(workers_.size());
    const bool drained = worker.cancelled && worker.active.empty() &&
                         worker.outstanding == 0 && worker.loads.empty() &&
                         worker.evals.empty() && worker.persistence.empty();
    if (worker.drained.exchange(drained) != drained)
      worker.controller_event = true;
    if (worker.released_capacity) {
      worker.released_capacity = false;
      WakeAdmissionWaiters();
    }
    if (worker.controller_event) {
      worker.controller_event = false;
      NotifyController();
    }

    SetProducing(worker,
                 HasJobCredit(worker) &&
                     (!worker.runnable.empty() || !worker.loads.empty() ||
                      !worker.evals.empty() || !worker.persistence.empty()));
    if (shutdown_.load()) break;
    // Continue gathering work locally, without a per-visit wake or controller
    // handoff. Suspended visits and full credits wait for whole completions.
    if (stop_mode_.load() == StopMode::kRunning && !worker.runnable.empty() &&
        worker.persistence.size() <
            static_cast<size_t>(settings_.minibatch_size))
      continue;
    if (HasJobCredit(worker) &&
        (!worker.loads.empty() || !worker.evals.empty() ||
         !worker.persistence.empty()))
      continue;
    std::unique_lock lock(worker.mailbox.mutex);
    worker.mailbox.cv.wait(lock, [&] {
      return shutdown_.load() || worker.mailbox.admission_ready ||
             !worker.mailbox.continuations.empty() ||
             !worker.mailbox.completions.empty() ||
             (!worker.cancelled && stop_mode_.load() != StopMode::kRunning);
    });
  }
}

void SearchRun::AdvanceVisit(Worker& worker, VisitId id) {
  VisitPool::Slot* slot = visits_.Lookup(id);
  if (!slot) return;
  std::unique_lock lock(slot->mutex);
  if (slot->epoch.load() != id.epoch() ||
      slot->visit.state == VisitState::kFree)
    return;
  assert(slot->visit.worker == worker.index);
  if (stop_mode_.load() != StopMode::kRunning) {
    Cancel(worker, *slot);
    return;
  }
  if (slot->visit.state == VisitState::kReadyBackup) {
    Backup(worker, *slot);
    return;
  }
  if (slot->visit.state != VisitState::kReadySelect) return;
  for (;;) {
    Visit& visit = slot->visit;
    if (stop_mode_.load() != StopMode::kRunning) {
      Cancel(worker, *slot);
      return;
    }
    if (visit.path.empty() || visit.path.back().key != visit.current_key)
      visit.path.push_back({.key = visit.current_key,
                            .generation = 0,
                            .selected_move = std::nullopt});
    PathStep& step = visit.path.back();
    auto snapshot = graph_->SnapshotNodeMetadata(visit.current_key);
    metrics_.selection_node_steps.fetch_add(1);
    detail::RaiseHighWater(metrics_.max_selected_depth,
                           static_cast<uint64_t>(visit.path.size()));
    if (!snapshot || snapshot->lifecycle == NodeLifecycle::kMaterializing) {
      if (SuspendForMaterialization(worker, *slot)) return;
      continue;
    }
    step.generation = snapshot->generation;
    if (snapshot->terminal != TerminalKind::kNonTerminal) {
      visit.result = detail::TerminalValue(snapshot->terminal);
      metrics_.terminal_visits.fetch_add(1);
      Backup(worker, *slot);
      return;
    }
    // Selection and propagation remain separate named policy seams.
    SelectResult selected =
        graph_->SelectAndReserve(visit.current_key, step.generation, settings_);
    if (selected.status == SelectStatus::kStale ||
        selected.status == SelectStatus::kMissing)
      continue;
    if (selected.status == SelectStatus::kMaterializing) {
      if (SuspendForMaterialization(worker, *slot)) return;
      continue;
    }
    if (selected.status == SelectStatus::kTerminal) {
      visit.result = detail::TerminalValue(snapshot->terminal);
      Backup(worker, *slot);
      return;
    }
    step.selected_move = selected.move;
    visit.history.Append(selected.move);
    visit.current_key =
        MakeNodeKey(visit.history, settings_.history_key_length);
  }
}

void SearchRun::Backup(Worker& worker, VisitPool::Slot& slot) {
  Visit& visit = slot.visit;
  visit.state = VisitState::kBackingUp;
  SearchValue value = visit.result;
  for (size_t i = visit.path.size(); i-- > 0;) {
    const PathStep& step = visit.path[i];
    const auto move =
        i + 1 < visit.path.size() ? step.selected_move : std::nullopt;
    const BackupResult result =
        graph_->BackupNode(step.key, step.generation, value, move);
    if (result.node == UpdateResult::kStale ||
        result.node == UpdateResult::kMissing)
      metrics_.stale_generation_node_updates.fetch_add(1);
    metrics_.backup_node_steps.fetch_add(1);
    if (move) {
      if (result.edge == UpdateResult::kUnderflow)
        metrics_.invariant_underflow_prevented.fetch_add(1);
      else if (result.edge != UpdateResult::kApplied)
        metrics_.stale_generation_edge_updates.fetch_add(1);
    }
    if (i == 0) break;
    value = value.Parent();
  }
  detail::RaiseHighWater(metrics_.max_depth,
                         static_cast<uint64_t>(visit.path.size()));
  FinishVisit(worker, slot, true);
}

void SearchRun::Cancel(Worker& worker, VisitPool::Slot& slot) {
  for (const PathStep& step : slot.visit.path) {
    if (!step.selected_move) continue;
    if (graph_->CancelEdge(step.key, step.generation, *step.selected_move) ==
        UpdateResult::kUnderflow)
      metrics_.invariant_underflow_prevented.fetch_add(1);
  }
  FinishVisit(worker, slot, false);
}

void SearchRun::FinishVisit(Worker& worker, VisitPool::Slot& slot,
                            bool completed) {
  const VisitId id = slot.visit.id;
  if (completed) {
    const uint64_t count = metrics_.visits_completed.fetch_add(1) + 1;
    if (node_limit_ && count == *node_limit_) worker.controller_event = true;
  } else {
    metrics_.visits_cancelled.fetch_add(1);
  }
  worker.active.erase(id.value);
  visits_.ReleaseLocked(slot, id);
  worker.released_capacity = true;
  // No admission, mailbox, or controller synchronization under the slot lock.
}

}  // namespace lczero::lc5
