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

#include "search/lc5/session/search_session.h"

#include <absl/time/time.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "search/lc5/graph/prior.h"

namespace lczero::lc5 {
namespace {

using std::chrono::steady_clock;
constexpr auto kInfoInterval = std::chrono::seconds(1);
constexpr auto kSmartPruningInterval = std::chrono::milliseconds(10);
// How often the controller checks the graph against its node limit.
constexpr auto kEvictionInterval = std::chrono::milliseconds(10);
// Root visits are counted in uint32_t; classic caps a search the same way.
constexpr uint64_t kMaxRootVisits = 4'000'000'000;

// Moves in the graph are from the side to move's perspective; UCI is not.
Move ToUci(Move move, const Position& position) {
  if (position.IsBlackToMove()) move.Flip();
  return move;
}

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

//////////////////////////////////////////////////////////////////////////////
// The session: construction, stop, the controller and output.
//////////////////////////////////////////////////////////////////////////////

SearchSession::SearchSession(VisitEngine* engine, Backend* backend,
                             UciResponder* responder,
                             Settings::Resolved settings,
                             TimeManager::Config time_management,
                             VisitOrigin root, GoParams go_params,
                             steady_clock::time_point start_time,
                             int64_t saved_ms)
    : engine_(engine),
      has_store_(engine->HasStore()),
      backend_(backend),
      responder_(responder),
      settings_(settings),
      root_(std::move(root)),
      start_time_(start_time),
      time_manager_(time_management, go_params,
                    root_.history.Last().IsBlackToMove(), start_time_,
                    saved_ms),
      visits_(settings_.max_active_visits) {
  for (int i = 0; i < settings_.threads; ++i) {
    workers_.push_back(std::make_unique<Worker>(i));
  }
  root_visits_at_start_ = engine_->GetRootStats(root_.key).visits;
  // Kept unless the search stops early with a clock budget.
  may_stop_early_ = !go_params.infinite && !go_params.ponder;
  node_limit_ =
      kMaxRootVisits - std::min(root_visits_at_start_, kMaxRootVisits);
  if (!go_params.ponder && go_params.nodes && *go_params.nodes >= 0) {
    node_limit_ =
        std::min(*node_limit_, static_cast<uint64_t>(*go_params.nodes));
  }
}

SearchSession::~SearchSession() {
  Abort();
  Wait();
}

void SearchSession::Start() {
  if (started_.exchange(true)) return;
  for (int i = 0; i < settings_.eval_threads; ++i) {
    if (has_store_)
      store_threads_.emplace_back(&SearchSession::StoreThread, this);
    evaluator_threads_.emplace_back(&SearchSession::EvaluatorThread, this);
  }
  for (auto& worker : workers_) {
    visit_threads_.emplace_back(&SearchSession::VisitThread, this,
                                std::ref(*worker));
  }
  if (node_limit_ == 0) Stop();
  controller_thread_ = std::thread(&SearchSession::ControllerThread, this);
}

void SearchSession::RequestStop(StopMode requested) {
  {
    // Admission and stop have one linearization point. Abort may override a
    // bestmove response until its output is committed.
    absl::MutexLock lock(admission_mutex_);
    if (requested == StopMode::kAbort ? !output_committed_.load()
                                      : !IsStopping()) {
      stop_mode_.store(requested);
    }
  }
  WakeController();
}

void SearchSession::Stop() { RequestStop(StopMode::kRespondBestmove); }

void SearchSession::Abort() { RequestStop(StopMode::kAbort); }

void SearchSession::Wait() {
  if (controller_thread_.joinable()) controller_thread_.join();
}

void SearchSession::WakeController() {
  absl::MutexLock lock(controller_mutex_);
  ++controller_generation_;
}

void SearchSession::WakeAllWorkers() {
  for (auto& worker : workers_) worker->inbox.Wake();
}

void SearchSession::ControllerThread() {
  auto next_info = steady_clock::now() + kInfoInterval;
  bool responded = false;
  for (;;) {
    uint64_t generation;
    {
      // Snapshot before inspecting predicates: any enabling event during
      // inspection changes the generation and prevents sleeping.
      absl::MutexLock lock(controller_mutex_);
      generation = controller_generation_;
    }
    const auto now = steady_clock::now();
    std::optional<steady_clock::time_point> next_check;
    if (!IsStopping()) {
      next_check = time_manager_.deadline();
      const RootStats root = engine_->GetRootStats(root_.key);
      if (time_manager_.ShouldStop(now) ||
          (node_limit_ && metrics_.visits_completed.load() >= *node_limit_)) {
        Stop();
      } else if (may_stop_early_ &&
                 (root.decided || (root.expanded && root.edges == 1))) {
        // The root's result is proven, or there is only one move to play.
        Stop();
      } else if (root.expanded &&
                 !time_manager_.CanAffordVisits(
                     root.best_edge_visits - root.second_edge_visits,
                     metrics_.visits_completed.load(), now)) {
        // The runner-up can't catch up within the remaining budget.
        Stop();
      } else if (may_stop_early_) {
        // Polls for a decided root, and for smart pruning.
        next_check = std::min(next_check.value_or(now + kSmartPruningInterval),
                              now + kSmartPruningInterval);
      }
      if (HasNodeLimit()) {
        QueueEvictionJobs();
        next_check = std::min(next_check.value_or(now + kEvictionInterval),
                              now + kEvictionInterval);
      }
    }
    if (!IsStopping() && now >= next_info) {
      std::vector<ThinkingInfo> infos{BuildThinkingInfo(false)};
      responder_->OutputThinkingInfo(&infos);
      next_info = now + kInfoInterval;
    }
    if (IsStopping() && !responded) {
      RespondBestMove();
      responded = true;
    }
    // Stop may arrive after the check above. Never finish draining before
    // output has been committed or suppressed and sleepers were woken.
    if (responded && IsFullyDrained()) break;
    auto wake_at = steady_clock::time_point::max();
    if (!IsStopping()) {
      wake_at = std::min(next_info, next_check.value_or(wake_at));
    }
    absl::MutexLock lock(controller_mutex_);
    const auto notified = [&] {
      controller_mutex_.AssertReaderHeld();
      return controller_generation_ != generation;
    };
    controller_mutex_.AwaitWithTimeout(
        absl::Condition(&notified),
        absl::FromChrono(wake_at - steady_clock::now()));
  }
  JoinThreads();
}

void SearchSession::RespondBestMove() {
  // The search ends here as far as the clock is concerned.
  unused_time_ms_.store(time_manager_.UnusedMs(steady_clock::now()));
  // Capture once, without waiting for any executor or eviction. Abort can
  // suppress this response until commitment, but cannot retract it.
  if (stop_mode_.load() == StopMode::kRespondBestmove) {
    std::vector<ThinkingInfo> infos{BuildThinkingInfo(true)};
    const auto& pv = infos.front().pv;
    BestMoveInfo best(pv.empty() ? FallbackMove() : pv.front(),
                      pv.size() > 1 ? pv[1] : Move{});
    bool respond;
    {
      absl::MutexLock lock(admission_mutex_);
      respond = stop_mode_.load() == StopMode::kRespondBestmove;
      if (respond) output_committed_.store(true);
    }
    if (respond) {
      responder_->OutputThinkingInfo(&infos);
      responder_->OutputBestMove(&best);
    }
  }
  // Synchronize the stop transition with sleepers only after the response.
  WakeAllWorkers();
  eval_jobs_.NotifyPredicateChanged();
}

bool SearchSession::IsFullyDrained() {
  if (visits_.active() != 0) return false;
  {
    absl::MutexLock lock(tickets_mutex_);
    if (!tickets_.empty()) return false;
  }
  if (jobs_in_flight_.load() != 0) return false;
  // Workers may hold unpublished local requests. Require their
  // explicit idle acknowledgement, not merely empty executor queues.
  for (auto& worker : workers_) {
    absl::MutexLock lock(worker->inbox.mutex);
    if (!worker->inbox.continuations.empty() ||
        !worker->inbox.completions.empty() || !worker->drained.load()) {
      return false;
    }
  }
  return true;
}

void SearchSession::JoinThreads() {
  shutdown_.store(true);
  WakeAllWorkers();
  for (auto& thread : visit_threads_) thread.join();
  store_jobs_.Close();
  eval_jobs_.Close();
  for (auto& thread : store_threads_) thread.join();
  for (auto& thread : evaluator_threads_) thread.join();
  finished_.store(true, std::memory_order_release);
}

// Deterministic legal move for when the root has no PV: it was never
// expanded, or it is terminal.
Move SearchSession::FallbackMove() const {
  const Position& position = root_.history.Last();
  const auto legal = position.GetBoard().GenerateLegalMoves();
  if (legal.empty()) return Move{};
  return ToUci(std::ranges::min(legal, {}, &Move::raw_data), position);
}

ThinkingInfo SearchSession::BuildThinkingInfo(bool final) const {
  const int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         steady_clock::now() - start_time_)
                         .count();
  const uint64_t nodes = metrics_.visits_completed.load();
  const uint64_t evals = metrics_.nn_evaluations.load();
  ThinkingInfo info;
  info.time = ms;
  info.nodes = nodes;
  info.nps = ms > 0 ? static_cast<int>(nodes * 1000 / ms) : 0;
  info.eps = ms > 0 ? static_cast<int>(evals * 1000 / ms) : 0;
  info.depth = static_cast<int>(metrics_.max_depth.load());
  info.seldepth = static_cast<int>(metrics_.max_selected_depth.load());
  engine_->FillThinkingInfo(root_, info);
  info.comment = metrics_.Format(engine_->GetGraphSize(), visits_.active(),
                                 eval_jobs_.Size());
  if (final) info.comment += " final";
  return info;
}

//////////////////////////////////////////////////////////////////////////////
// Visits: admission, the worker loop and each visit's steps.
//////////////////////////////////////////////////////////////////////////////

bool SearchSession::IsAdmissionClosed() const {
  return IsStopping() || (node_limit_ && admitted_ >= *node_limit_);
}

std::optional<VisitId> SearchSession::AllocateVisit(const VisitOrigin& origin) {
  // Allocate never holds pool -> slot.
  if (IsAdmissionClosed() || visits_.active() >= ActiveVisitLimit()) {
    return std::nullopt;
  }
  const auto id = visits_.Allocate(origin);
  if (!id) return std::nullopt;
  ++admitted_;
  metrics_.visits_admitted.fetch_add(1);
  RaiseHighWater(metrics_.active_visits_high_water, visits_.active());
  return id;
}

size_t SearchSession::ActiveVisitLimit() const {
  if (settings_.active_visits_ratio <= 0.0f) return visits_.capacity();
  const double tree = static_cast<double>(root_visits_at_start_ +
                                          metrics_.visits_completed.load());
  return std::min(visits_.capacity(),
                  kMinActiveVisits + static_cast<size_t>(
                                         tree * settings_.active_visits_ratio));
}

void SearchSession::AdmitVisits(Worker& worker) {
  absl::MutexLock lock(admission_mutex_);
  worker.admission_waiting = false;
  // A modest per-worker share prevents one worker from monopolizing the pool.
  const size_t share =
      (visits_.capacity() + workers_.size() - 1) / workers_.size();
  for (size_t i = 0; i < kWorkChunk && worker.active.size() < share; ++i) {
    const auto id = AllocateVisit(root_);
    if (!id) {
      // The pool is full. Registration shares admission_mutex_ with the
      // capacity check in WakeAdmissionWaiters, so it cannot miss a release.
      worker.admission_waiting = !IsAdmissionClosed();
      break;
    }
    worker.active.insert(id->value);
    worker.runnable.push_back(*id);
  }
  RaiseHighWater(metrics_.visits_ready_high_water, worker.runnable.size());
}

void SearchSession::WakeAdmissionWaiters() {
  std::vector<size_t> wake;
  {
    absl::MutexLock lock(admission_mutex_);
    if (IsAdmissionClosed()) return;
    const size_t limit = ActiveVisitLimit();
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
    auto& inbox = workers_[index]->inbox;
    absl::MutexLock lock(inbox.mutex);
    inbox.admission_ready = true;
  }
}

void SearchSession::SendContinuations(size_t worker,
                                      std::vector<Continuation>& batch) {
  if (batch.empty()) return;
  {
    auto& inbox = workers_[worker]->inbox;
    absl::MutexLock lock(inbox.mutex);
    inbox.continuations.insert(inbox.continuations.end(), batch.begin(),
                               batch.end());
  }
  batch.clear();
  metrics_.inbox_notifications.fetch_add(1);
}

void SearchSession::DeliverContinuation(Worker& worker,
                                        const Continuation& continuation) {
  if (continuation.worker == worker.index) {
    ResumeVisit(worker, continuation);
  } else {
    worker.outgoing[continuation.worker].push_back(continuation);
  }
}

void SearchSession::ResumeVisit(Worker& worker, const Continuation& event) {
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
        (IsStopping() && (event.kind == Continuation::Kind::kSelect ||
                          event.kind == Continuation::Kind::kRestored))) {
      CancelVisit(worker, *slot);
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
      switch (engine_->ResumeVisit(visit, metrics_)) {
        case VisitEngine::ResumeAction::kRestart:
          RestartVisit(*slot);
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

void SearchSession::VisitThread(Worker& worker) {
  worker.outgoing.resize(workers_.size());
  for (;;) {
    ProcessInbox(worker);
    if (IsStopping()) {
      CancelWorkerVisits(worker);
    } else {
      AdvanceRunnableVisits(worker, kWorkChunk);
      AdmitVisits(worker);
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
    auto& inbox = worker.inbox;
    absl::MutexLock lock(inbox.mutex);
    const auto ready = [&] {
      inbox.mutex.AssertReaderHeld();
      return shutdown_.load() || inbox.admission_ready ||
             !inbox.continuations.empty() || !inbox.completions.empty() ||
             (!worker.cancelled && IsStopping());
    };
    inbox.mutex.Await(absl::Condition(&ready));
  }
}

void SearchSession::ProcessInbox(Worker& worker) {
  std::vector<Continuation> continuations;
  std::vector<std::unique_ptr<Job>> completions;
  bool admission_ready;
  {
    absl::MutexLock lock(worker.inbox.mutex);
    continuations.swap(worker.inbox.continuations);
    completions.swap(worker.inbox.completions);
    admission_ready = std::exchange(worker.inbox.admission_ready, false);
  }
  if (admission_ready || !continuations.empty() || !completions.empty()) {
    // Actual incoming work can enable preparation or refill. This transition
    // disables starvation and needs no broadcast.
    SetWorkerProducing(worker, true);
    if (worker.drained.exchange(false)) worker.controller_event = true;
  }
  for (const auto& event : continuations) ResumeVisit(worker, event);
  for (auto& job : completions) ConsumeJob(worker, std::move(job));
}

void SearchSession::AdvanceRunnableVisits(Worker& worker, size_t limit) {
  for (size_t i = 0; i < limit && !worker.runnable.empty(); ++i) {
    const VisitId id = worker.runnable.front();
    worker.runnable.pop_front();
    AdvanceVisit(worker, id);
  }
}

void SearchSession::CancelWorkerVisits(Worker& worker) {
  // Only the owner ever cancels or mutates a visit. Epochs make later ticket
  // completions harmless even after the slot has been reused. Ready backups
  // finish; waiting visits stay until their ticket completes or is cancelled,
  // and come back through ResumeVisit.
  if (!worker.cancelled) {
    const std::vector<uint64_t> active(worker.active.begin(),
                                       worker.active.end());
    for (uint64_t value : active) AdvanceVisit(worker, VisitId{value});
    worker.runnable.clear();
    worker.cancelled = true;
  }
  // External admissions published just before stop still need cancellation.
  AdvanceRunnableVisits(worker, std::numeric_limits<size_t>::max());
}

void SearchSession::PublishWorkerState(Worker& worker) {
  const bool drained = worker.cancelled && worker.active.empty() &&
                       worker.outstanding == 0 && !worker.HasQueuedIo();
  if (worker.drained.exchange(drained) != drained) {
    worker.controller_event = true;
  }
  if (std::exchange(worker.released_capacity, false)) WakeAdmissionWaiters();
  if (std::exchange(worker.controller_event, false)) WakeController();
  SetWorkerProducing(worker,
                     HasJobCredit(worker) &&
                         (!worker.runnable.empty() || worker.HasQueuedIo()));
}

bool SearchSession::HasLocalWork(const Worker& worker) const {
  return (!IsStopping() && !worker.runnable.empty()) ||
         (HasJobCredit(worker) && worker.HasQueuedIo());
}

void SearchSession::AdvanceVisit(Worker& worker, VisitId id) {
  VisitPool::Slot* slot = visits_.Lookup(id);
  if (!slot) return;
  absl::MutexLock lock(slot->mutex);
  if (!visits_.IsLiveLocked(*slot, id)) return;
  Visit& visit = slot->visit;
  if (visit.state == VisitState::kReadyBackup) {
    BackupVisit(worker, *slot);
    return;
  }
  if (visit.state == VisitState::kWaitingMaterialization) return;
  if (IsStopping()) {
    CancelVisit(worker, *slot);
    return;
  }
  if (visit.state != VisitState::kReadySelect) return;
  for (;;) {
    if (IsStopping()) {
      CancelVisit(worker, *slot);
      return;
    }
    if (engine_->DescendVisit(visit, metrics_) ==
        VisitEngine::DescendResult::kValuedLeaf) {
      BackupVisit(worker, *slot);
      return;
    }
    if (SuspendVisit(worker, *slot)) return;
  }
}

void SearchSession::BackupVisit(Worker& worker, VisitPool::Slot& slot) {
  Visit& visit = slot.visit;
  visit.state = VisitState::kBackingUp;
  engine_->BackupVisit(visit, metrics_);
  RaiseHighWater(metrics_.max_depth, visit.path.size());
  FinishVisit(worker, slot, true);
}

void SearchSession::CancelVisit(Worker& worker, VisitPool::Slot& slot) {
  engine_->CancelVisit(slot.visit, metrics_);
  FinishVisit(worker, slot, false);
}

void SearchSession::RestartVisit(VisitPool::Slot& slot) {
  Visit& visit = slot.visit;
  engine_->CancelVisit(visit, metrics_);
  visit.history = visit.origin.history;
  visit.current_key = visit.origin.key;
  visit.path = visit.origin.backup_prefix;
  visit.two_fold_origin.reset();
  visit.state = VisitState::kReadySelect;
  metrics_.visits_restarted.fetch_add(1);
}

void SearchSession::FinishVisit(Worker& worker, VisitPool::Slot& slot,
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

//////////////////////////////////////////////////////////////////////////////
// Materialization: tickets, requests, jobs and batches.
//////////////////////////////////////////////////////////////////////////////

bool SearchSession::SuspendVisit(Worker& worker, VisitPool::Slot& slot) {
  Visit& visit = slot.visit;
  bool owner;
  MaterializationTicketId ticket_id;
  {
    // Lock order: slot -> tickets -> one graph shard. No inbox transfer here.
    absl::MutexLock lock(tickets_mutex_);
    const auto existing = ticket_by_key_.find(visit.current_key);
    owner = existing == ticket_by_key_.end();
    ticket_id = owner ? next_ticket_++ : existing->second;
    const auto node =
        engine_->FindOrCreateMaterialization(visit.current_key, ticket_id);
    if (node.created) metrics_.graph_nodes_created.fetch_add(1);
    if (node.lifecycle == NodeLifecycle::kExpanded) return false;
    assert(node.ticket == ticket_id);
    const Continuation continuation{
        .id = visit.id, .worker = worker.index, .ticket = ticket_id};
    if (owner) {
      tickets_.emplace(ticket_id, Ticket{visit.current_key, continuation, {}});
      ticket_by_key_.emplace(visit.current_key, ticket_id);
      metrics_.tickets_created.fetch_add(1);
    } else {
      auto& waiters = tickets_.at(ticket_id).waiters;
      waiters.push_back(continuation);
      metrics_.ticket_waiters.fetch_add(1);
      RaiseHighWater(metrics_.maximum_waiters_per_ticket, waiters.size());
    }
    visit.path.back().generation = node.generation;
    visit.waiting_ticket = ticket_id;
    visit.state = VisitState::kWaitingMaterialization;
  }
  metrics_.visits_suspended.fetch_add(1);
  // Wakes the controller once the graph outgrows its limit, so eviction keeps
  // up with fast searches between its periodic checks.
  if (owner && HasNodeLimit() &&
      engine_->GetGraphSize() > settings_.graph_node_limit &&
      !eviction_requested_.exchange(true)) {
    worker.controller_event = true;
  }
  if (owner) {
    // Preparation and publication happen later, outside this slot's lock.
    MaterializationRequest request{.ticket = ticket_id,
                                   .key = visit.current_key,
                                   .history = visit.history};
    (has_store_ ? worker.loads : worker.unprepared)
        .push_back(std::move(request));
  }
  return true;
}

void SearchSession::PrepareRequest(Worker& worker,
                                   MaterializationRequest request) {
  if (IsStopping()) {
    CancelTicket(worker, request.ticket);
    return;
  }
  request.payload = DetectTerminal(request.history);
  if (request.payload.terminal != TerminalKind::kNonTerminal) {
    CompleteTicket(worker, request.ticket, std::move(request.payload));
    return;
  }
  request.payload.priors.resize(request.payload.moves.size());
  request.state = MaterializationRequest::State::kPending;
  metrics_.eval_requests.fetch_add(1);
  worker.evals.push_back(std::move(request));
}

void SearchSession::CompleteTicket(Worker& worker,
                                   MaterializationTicketId ticket_id,
                                   ExpansionPayload payload) {
  Ticket ticket;
  MaterializationResult installed;
  {
    absl::MutexLock lock(tickets_mutex_);
    const auto it = tickets_.find(ticket_id);
    if (it == tickets_.end()) return;
    assert(it->second.owner.worker == worker.index);
    installed =
        engine_->CompleteMaterialization(it->second.key, ticket_id, payload);
    ticket = std::move(it->second);
    tickets_.erase(it);
    ticket_by_key_.erase(ticket.key);
  }
  RaiseHighWater(metrics_.graph_size_high_water, engine_->GetGraphSize());
  if (!installed.accepted) metrics_.publications_rejected.fetch_add(1);
  // The install already counted the leaf value on the node, so the owner
  // backs it up only through the steps above. Waiters select below the node
  // unless it is terminal, in which case they back up the whole path like a
  // later visit to a terminal. A rejected publication resumes selection only.
  const bool terminal =
      installed.accepted && payload.terminal != TerminalKind::kNonTerminal;
  const auto wake = [&](Continuation event, bool owner) {
    event.kind = !installed.accepted ? Continuation::Kind::kSelect
                 : owner             ? Continuation::Kind::kBackupAbove
                 : terminal          ? Continuation::Kind::kBackup
                                     : Continuation::Kind::kSelect;
    event.generation = installed.generation;
    event.value = payload.leaf;
    event.terminal = terminal;
    event.exact = installed.accepted && IsExact(payload.terminal);
    DeliverContinuation(worker, event);
  };
  wake(ticket.owner, true);
  for (const auto& event : ticket.waiters) wake(event, false);
}

bool SearchSession::RestoreTicket(Worker& worker,
                                  MaterializationTicketId ticket_id,
                                  std::span<const uint8_t> record) {
  Ticket ticket;
  MaterializationResult restored;
  {
    absl::MutexLock lock(tickets_mutex_);
    const auto it = tickets_.find(ticket_id);
    if (it == tickets_.end()) return true;
    assert(it->second.owner.worker == worker.index);
    restored = engine_->RestoreFromRecord(it->second.key, ticket_id, record);
    if (!restored.accepted) return false;
    ticket = std::move(it->second);
    tickets_.erase(it);
    ticket_by_key_.erase(ticket.key);
  }
  metrics_.nodes_restored.fetch_add(1);
  RaiseHighWater(metrics_.graph_size_high_water, engine_->GetGraphSize());
  // The node has its statistics back and nothing new to back up, so the owner
  // and the waiters all select below it.
  const auto wake = [&](Continuation event) {
    event.kind = Continuation::Kind::kRestored;
    event.generation = restored.generation;
    DeliverContinuation(worker, event);
  };
  wake(ticket.owner);
  for (const auto& event : ticket.waiters) wake(event);
  return true;
}

void SearchSession::CancelTicket(Worker& worker,
                                 MaterializationTicketId ticket_id) {
  // Only called on stop, for evaluations that will never be computed. The
  // owner and waiters are still alive and must be woken to cancel.
  Ticket ticket;
  {
    absl::MutexLock lock(tickets_mutex_);
    const auto it = tickets_.find(ticket_id);
    if (it == tickets_.end()) return;
    engine_->CancelMaterialization(it->second.key, ticket_id);
    ticket = std::move(it->second);
    tickets_.erase(it);
    ticket_by_key_.erase(ticket.key);
  }
  const auto wake = [&](Continuation event) {
    event.kind = Continuation::Kind::kCancel;
    DeliverContinuation(worker, event);
  };
  wake(ticket.owner);
  for (const auto& event : ticket.waiters) wake(event);
}

void SearchSession::CancelPendingRequests(Worker& worker) {
  for (auto* requests : {&worker.unprepared, &worker.loads, &worker.evals}) {
    for (const auto& request : *requests) {
      CancelTicket(worker, request.ticket);
    }
    requests->clear();
  }
}

bool SearchSession::HasJobCredit(const Worker& worker) const {
  return worker.outstanding < kBatchCredits &&
         worker.outstanding_items <
             2 * static_cast<size_t>(settings_.minibatch_size);
}

void SearchSession::SetWorkerProducing(Worker& worker, bool producing) {
  // Only true -> false enables the collectors' starvation predicate, so only
  // that edge needs the queue-mutex handshake.
  if (worker.producing.exchange(producing) && !producing)
    eval_jobs_.NotifyPredicateChanged();
}

void SearchSession::SubmitJobs(Worker& worker) {
  if (IsStopping()) CancelPendingRequests(worker);
  // Detect terminals and generate legal moves for fresh requests outside any
  // slot lock. Terminal ones publish right away; the rest join `evals`.
  while (!worker.unprepared.empty()) {
    MaterializationRequest request = std::move(worker.unprepared.front());
    worker.unprepared.pop_front();
    PrepareRequest(worker, std::move(request));
  }
  while (HasJobCredit(worker)) {
    if (IsStopping()) CancelPendingRequests(worker);
    auto job = TakeJob(worker);
    if (!job) break;
    ++worker.outstanding;
    worker.outstanding_items += job->items();
    RaiseHighWater(metrics_.io_jobs_high_water,
                   jobs_in_flight_.fetch_add(1) + 1);
    if (job->kind == JobKind::kEval) {
      eval_jobs_.Push(std::move(job));
      RaiseHighWater(metrics_.ready_eval_high_water, eval_jobs_.Size());
    } else {
      assert(has_store_);
      store_jobs_.Push(std::move(job));
    }
  }
}

std::unique_ptr<Job> SearchSession::TakeJob(Worker& worker) const {
  // Small whole jobs leave room to collect more cache misses without exceeding
  // the backend batch. Item credits bound loads and evaluations.
  const size_t batch_size = settings_.minibatch_size;
  const size_t limit =
      std::min((batch_size + 3) / 4, 2 * batch_size - worker.outstanding_items);
  auto job = std::make_unique<Job>();
  job->worker = worker.index;
  job->queued_at = std::chrono::steady_clock::now();
  if (!worker.loads.empty()) {
    job->kind = JobKind::kLoad;
    MoveFront(worker.loads, job->requests, limit);
  } else if (!worker.evals.empty()) {
    job->kind = JobKind::kEval;
    MoveFront(worker.evals, job->requests, limit);
  } else {
    return nullptr;
  }
  return job;
}

void SearchSession::ReturnJob(std::unique_ptr<Job> job) {
  {
    auto& inbox = workers_[job->worker]->inbox;
    absl::MutexLock lock(inbox.mutex);
    inbox.completions.push_back(std::move(job));
  }
  metrics_.inbox_notifications.fetch_add(1);
}

void SearchSession::ConsumeJob(Worker& worker, std::unique_ptr<Job> job) {
  assert(worker.outstanding > 0);
  for (auto& request : job->requests) {
    switch (request.state) {
      case MaterializationRequest::State::kEvaluated:
        SortByPrior(request.payload);
        CompleteTicket(worker, request.ticket, std::move(request.payload));
        break;
      case MaterializationRequest::State::kStoreHit:
        metrics_.node_store_hits.fetch_add(1);
        if (!RestoreTicket(worker, request.ticket, request.record)) {
          // Not a record for this policy or format: evaluate the node.
          metrics_.restores_rejected.fetch_add(1);
          request.record.clear();
          PrepareRequest(worker, std::move(request));
        }
        break;
      case MaterializationRequest::State::kStoreMiss:
        metrics_.node_store_misses.fetch_add(1);
        PrepareRequest(worker, std::move(request));
        break;
      case MaterializationRequest::State::kPending:
      case MaterializationRequest::State::kQueued:
        CancelTicket(worker, request.ticket);
        break;
    }
  }
  --worker.outstanding;
  assert(worker.outstanding_items >= job->items());
  worker.outstanding_items -= job->items();
  // Decrement last: publication is part of the job.
  jobs_in_flight_.fetch_sub(1);
}

void SearchSession::StoreThread() {
  std::unique_ptr<Job> job;
  while (store_jobs_.Pop(&job)) {
    if (job->kind == JobKind::kPersist) {
      RunEvictionJob(*job);
      continue;
    }
    if (!IsStopping()) {
      std::vector<NodeKey> keys;
      for (const auto& request : job->requests) keys.push_back(request.key);
      auto loaded = engine_->ReadFromStore(keys);
      assert(loaded.size() == job->requests.size());
      for (size_t i = 0; i < loaded.size(); ++i) {
        auto& request = job->requests[i];
        if (loaded[i]) {
          request.record = std::move(*loaded[i]);
          request.state = MaterializationRequest::State::kStoreHit;
        } else {
          request.state = MaterializationRequest::State::kStoreMiss;
        }
      }
      metrics_.node_store_load_batches.fetch_add(1);
      metrics_.node_store_load_keys.fetch_add(keys.size());
    }
    ReturnJob(std::move(job));
  }
}

void SearchSession::QueueEvictionJobs() {
  if (!HasNodeLimit() || IsStopping() || evictions_in_flight_.load() > 0)
    return;
  eviction_requested_.store(false);
  const size_t limit = settings_.graph_node_limit;
  const size_t size = engine_->GetGraphSize();
  if (size <= limit) return;
  // Evict down to a low-water mark, so the next round is not due right away.
  const size_t target = limit - limit / 10;
  auto job = std::make_unique<Job>();
  job->kind = JobKind::kPersist;
  job->queued_at = std::chrono::steady_clock::now();
  job->evict_count = size - target;
  evictions_in_flight_.fetch_add(1);
  RaiseHighWater(metrics_.io_jobs_high_water, jobs_in_flight_.fetch_add(1) + 1);
  store_jobs_.Push(std::move(job));
}

void SearchSession::RunEvictionJob(Job& job) {
  engine_->EvictToStore(job.evict_count, root_.key, metrics_);
  jobs_in_flight_.fetch_sub(1);
  // The controller may queue the next round, or finish draining.
  evictions_in_flight_.fetch_sub(1);
  WakeController();
}

void SearchSession::EvaluatorThread() {
  // A popped job that did not fit the previous batch. Retained jobs remain
  // counted as collecting and in flight.
  std::unique_ptr<Job> next;
  for (;;) {
    if (!next) {
      if (!eval_jobs_.Pop(&next)) break;
      collecting_jobs_.fetch_add(1);
    }
    if (IsStopping()) {
      collecting_jobs_.fetch_sub(1);
      ReturnJob(std::move(next));
      continue;
    }
    RunBatch(CollectBatch(std::move(next), &next));
  }
}

Batch SearchSession::CollectBatch(std::unique_ptr<Job> job,
                                  std::unique_ptr<Job>* next) {
  const size_t target = settings_.minibatch_size;
  const auto deadline =
      settings_.max_batch_delay_ms == 0
          ? std::chrono::steady_clock::time_point::max()
          : job->queued_at +
                std::chrono::milliseconds(settings_.max_batch_delay_ms);
  Batch batch{.computation = backend_->CreateComputation()};
  for (;;) {
    AddJobInputs(*batch.computation, *job);
    batch.jobs.push_back(std::move(job));
    const size_t used = batch.computation->UsedBatchSize();
    if (used >= target || IsStopping()) break;
    // Consume already-ready jobs first, even past the deadline, rather than
    // fragmenting an accumulated backlog into single-job batches.
    if (!eval_jobs_.TryPop(&job)) {
      // Immediate-only batches return credits instead of waiting on producers
      // that need those cache results. The oldest job's deadline is absolute;
      // zero disables it, but dependency starvation still flushes.
      if (used == 0 || IsStopping()) break;
      eval_jobs_.NotifyPredicateChanged();
      if (!eval_jobs_.PopUntil(&job, deadline, [&] {
            return IsStopping() || IsEvaluationStarved();
          })) {
        batch.timed_out = std::chrono::steady_clock::now() >= deadline;
        break;
      }
    }
    collecting_jobs_.fetch_add(1);
    // A whole job's size bounds its uncached inputs and must fit before
    // AddInput; otherwise retain it for the next batch.
    if (used + job->requests.size() > target) {
      *next = std::move(job);
      break;
    }
  }
  // Decreasing collecting_jobs_ only disables the starvation predicate; the
  // increment before PopUntil above is the enabling handshake.
  collecting_jobs_.fetch_sub(batch.jobs.size());
  return batch;
}

void SearchSession::AddJobInputs(BackendComputation& computation, Job& job) {
  for (auto& request : job.requests) {
    if (IsStopping()) break;
    auto& payload = request.payload;
    const auto result =
        computation.AddInput(EvalPosition{.pos = request.history.GetPositions(),
                                          .legal_moves = payload.moves},
                             EvalResultPtr{.q = &payload.leaf.q,
                                           .d = &payload.leaf.d,
                                           .m = &payload.leaf.m,
                                           .p = payload.priors});
    if (result == BackendComputation::FETCHED_IMMEDIATELY) {
      request.state = MaterializationRequest::State::kEvaluated;
      metrics_.cache_hits.fetch_add(1);
    } else {
      request.state = MaterializationRequest::State::kQueued;
    }
  }
}

void SearchSession::RunBatch(Batch batch) {
  const size_t used = batch.computation->UsedBatchSize();
  bool authorized = false;
  if (used > 0) {
    // The final gate shares stop's linearization point. A serial evaluator
    // cannot authorize again until this computation has returned.
    absl::MutexLock lock(admission_mutex_);
    authorized = !IsStopping();
  }
  if (authorized) {
    if (used < static_cast<size_t>(settings_.minibatch_size)) {
      (batch.timed_out ? metrics_.partial_timeout_flushes
                       : metrics_.partial_starvation_flushes)
          .fetch_add(1);
    }
    metrics_.evaluation_batches.fetch_add(1);
    metrics_.nn_evaluations.fetch_add(used);
    batch.computation->ComputeBlocking();
    for (auto& job : batch.jobs) {
      for (auto& request : job->requests) {
        if (request.state == MaterializationRequest::State::kQueued) {
          request.state = MaterializationRequest::State::kEvaluated;
        }
      }
    }
  }
  // Destroy the computation while the jobs it references are still alive.
  batch.computation.reset();
  for (auto& job : batch.jobs) ReturnJob(std::move(job));
}

bool SearchSession::IsEvaluationStarved() const {
  // Executing, store or returned jobs can still unblock a producer. Only
  // predict starvation when every outstanding job is held by a collector and
  // no worker can submit independently; the deadline bounds the uncertainty.
  for (const auto& worker : workers_) {
    if (worker->producing.load()) return false;
  }
  // Zero disables the timeout, not dependency flushing: with no deadline,
  // currently blocked producers cannot justify an unbounded wait.
  return settings_.max_batch_delay_ms == 0 ||
         jobs_in_flight_.load() <= collecting_jobs_.load();
}

}  // namespace lczero::lc5
