#include "search/lc5/search.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <unordered_set>

namespace lczero::lc5 {
namespace {

template <typename T>
void RaiseHighWater(std::atomic<T>& target, T value) {
  T old = target.load(std::memory_order_relaxed);
  while (old < value &&
         !target.compare_exchange_weak(old, value, std::memory_order_relaxed)) {
  }
}

SearchValue TerminalValue(TerminalKind terminal) {
  return terminal == TerminalKind::kCheckmate ? SearchValue{-1.0f, 0.0f, 0.0f}
                                              : SearchValue{0.0f, 1.0f, 0.0f};
}

}  // namespace

SearchRun::SearchRun(GameGraph* graph, NodeStore* store, Backend* backend,
                     UciResponder* responder, Settings::Resolved settings,
                     VisitOrigin root, GoParams go_params,
                     std::chrono::steady_clock::time_point start_time)
    : graph_(graph),
      store_(store),
      backend_(backend),
      responder_(responder),
      settings_(settings),
      root_(std::move(root)),
      go_params_(std::move(go_params)),
      start_time_(start_time),
      visits_(settings_.max_active_visits) {
  for (int i = 0; i < settings_.threads; ++i)
    workers_.push_back(std::make_unique<Worker>(i));
  if (!go_params_.ponder && go_params_.nodes && *go_params_.nodes >= 0)
    node_limit_ = static_cast<uint64_t>(*go_params_.nodes);
  if (!go_params_.ponder && go_params_.movetime)
    deadline_ = start_time_ + std::chrono::milliseconds(*go_params_.movetime);
}

SearchRun::~SearchRun() {
  Abort();
  Wait();
}

void SearchRun::Start() {
  if (started_.exchange(true)) return;
  for (int i = 0; i < settings_.eval_threads; ++i) {
    if (store_) store_threads_.emplace_back(&SearchRun::StoreWorker, this);
    evaluator_threads_.emplace_back(&SearchRun::EvaluatorWorker, this);
  }
  for (auto& worker : workers_)
    visit_threads_.emplace_back(
        [this, ptr = worker.get()] { VisitWorker(*ptr); });
  if (node_limit_ && *node_limit_ == 0) Stop();
  controller_thread_ = std::thread(&SearchRun::Controller, this);
}

void SearchRun::RequestStop(StopMode requested) {
  {
    // Admission and stop have one linearization point.
    std::lock_guard lock(admission_mutex_);
    const auto current = stop_mode_.load();
    if (requested == StopMode::kAbort) {
      if (output_committed_.load()) return;
      stop_mode_.store(StopMode::kAbort);
    } else if (current == StopMode::kRunning) {
      stop_mode_.store(StopMode::kRespondBestmove);
    }
  }
  for (auto& worker : workers_) {
    // Synchronize the predicate transition with each mailbox's wait.
    {
      std::lock_guard lock(worker->mailbox.mutex);
    }
    worker->mailbox.cv.notify_one();
  }
  eval_jobs_.WakeAll();
  NotifyController();
}

void SearchRun::Stop() { RequestStop(StopMode::kRespondBestmove); }
void SearchRun::Abort() { RequestStop(StopMode::kAbort); }

void SearchRun::Wait() {
  if (controller_thread_.joinable()) controller_thread_.join();
}

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
  RaiseHighWater(metrics_.active_visits_high_water,
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
  RaiseHighWater(metrics_.visits_ready_high_water,
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

void SearchRun::ReturnJob(std::unique_ptr<Job> job) {
  auto& mailbox = workers_[job->worker]->mailbox;
  {
    std::lock_guard lock(mailbox.mutex);
    mailbox.completions.push_back(std::move(job));
  }
  metrics_.mailbox_notifications.fetch_add(1);
  mailbox.cv.notify_one();
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
    RaiseHighWater(metrics_.max_selected_depth,
                   static_cast<uint64_t>(visit.path.size()));
    if (!snapshot || snapshot->lifecycle == NodeLifecycle::kMaterializing) {
      if (SuspendForMaterialization(worker, *slot)) return;
      continue;
    }
    step.generation = snapshot->generation;
    if (snapshot->terminal != TerminalKind::kNonTerminal) {
      visit.result = TerminalValue(snapshot->terminal);
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
      visit.result = TerminalValue(snapshot->terminal);
      Backup(worker, *slot);
      return;
    }
    step.selected_move = selected.move;
    visit.history.Append(selected.move);
    visit.current_key =
        MakeNodeKey(visit.history, settings_.history_key_length);
  }
}

bool SearchRun::SuspendForMaterialization(Worker& worker,
                                          VisitPool::Slot& slot) {
  Visit& visit = slot.visit;
  bool owner;
  MaterializationTicketId ticket_id;
  {
    // Lock order: slot -> tickets -> one graph shard. No mailbox transfer here.
    std::lock_guard lock(tickets_mutex_);
    auto existing = ticket_by_key_.find(visit.current_key);
    owner = existing == ticket_by_key_.end();
    ticket_id = owner ? next_ticket_++ : existing->second;
    auto result =
        graph_->FindOrCreateMaterializing(visit.current_key, ticket_id);
    if (result.created) metrics_.graph_nodes_created.fetch_add(1);
    if (result.lifecycle == NodeLifecycle::kExpanded) return false;
    assert(result.ticket == ticket_id);
    Continuation continuation{.id = visit.id,
                              .worker = worker.index,
                              .ticket = ticket_id,
                              .value = {}};
    if (owner) {
      tickets_.emplace(ticket_id,
                       Ticket{ticket_id, visit.current_key, continuation, {}});
      ticket_by_key_.emplace(visit.current_key, ticket_id);
      metrics_.tickets_created.fetch_add(1);
    } else {
      auto& ticket = tickets_.at(ticket_id);
      ticket.waiters.push_back(continuation);
      metrics_.ticket_waiters.fetch_add(1);
      RaiseHighWater(metrics_.maximum_waiters_per_ticket,
                     static_cast<uint64_t>(ticket.waiters.size()));
    }
    visit.path.back().generation = result.generation;
    visit.waiting_ticket = ticket_id;
    visit.state = VisitState::kWaitingMaterialization;
  }
  metrics_.visits_suspended.fetch_add(1);
  if (owner) {
    Request request{.ticket = ticket_id,
                    .key = visit.current_key,
                    .history = visit.history,
                    .payload = {},
                    .loaded = std::nullopt};
    // Terminal preparation is worker-owned even on the optional store path.
    // Publication happens outside the selecting slot's lock.
    if (store_)
      worker.loads.push_back(std::move(request));
    else
      worker.evals.push_back(std::move(request));
  }
  return true;
}

ExpansionPayload SearchRun::DetectTerminal(
    const PositionHistory& history) const {
  ExpansionPayload payload;
  const Position& position = history.Last();
  const auto legal = position.GetBoard().GenerateLegalMoves();
  if (legal.empty()) {
    payload.terminal = position.GetBoard().IsUnderCheck()
                           ? TerminalKind::kCheckmate
                           : TerminalKind::kStalemate;
  } else if (!position.GetBoard().HasMatingMaterial()) {
    payload.terminal = TerminalKind::kInsufficientMaterial;
  } else if (position.GetRule50Ply() >= 100) {
    payload.terminal = TerminalKind::kRule50;
  } else if (position.GetRepetitions() >= 2) {
    payload.terminal = TerminalKind::kRepetition;
  } else {
    payload.moves.assign(legal.begin(), legal.end());
    return payload;
  }
  auto value = TerminalValue(payload.terminal);
  payload.leaf_q = value.q;
  payload.leaf_d = value.d;
  payload.leaf_m = value.m;
  return payload;
}

void SearchRun::PrepareEvaluation(Worker& worker, Request request) {
  request.payload = DetectTerminal(request.history);
  if (request.payload.terminal != TerminalKind::kNonTerminal) {
    CompleteMaterialization(worker, request.ticket, std::move(request.payload),
                            true);
  } else {
    request.payload.priors.resize(request.payload.moves.size());
    metrics_.eval_requests.fetch_add(1);
    worker.evals.push_back(std::move(request));
  }
}

bool SearchRun::HasJobCredit(const Worker& worker) const {
  return worker.outstanding < kBatchCredits &&
         worker.outstanding_items <
             2 * static_cast<size_t>(settings_.minibatch_size);
}

void SearchRun::SetProducing(Worker& worker, bool producing) {
  // Only true -> false enables the collector's starvation predicate. Keep the
  // queue-mutex handshake for that edge; false -> true merely disables it.
  if (worker.producing.exchange(producing) && !producing) eval_jobs_.WakeAll();
}

bool SearchRun::EvaluationStarved() const {
  // Executing/store/returned jobs can still unblock a producer. Only predict
  // starvation when every outstanding job is held by a collector and no
  // worker can submit independently. Uncertainty is bounded by the deadline.
  for (const auto& worker : workers_)
    if (worker->producing.load()) return false;
  // Zero disables the timeout, not dependency flushing. Do not wait for
  // another computation/store to return: currently blocked producers cannot
  // justify an unbounded collection wait.
  return settings_.max_batch_delay_ms == 0 ||
         jobs_in_flight_.load() <= collecting_jobs_.load();
}

void SearchRun::SubmitJobs(Worker& worker) {
  // Fresh requests have no legal moves yet. Prepare them outside any slot lock.
  // Only process the current queue, since preparation appends evaluated misses.
  const size_t count = worker.evals.size();
  for (size_t i = 0; i < count; ++i) {
    Request request = std::move(worker.evals.front());
    worker.evals.pop_front();
    if (request.payload.moves.empty())
      PrepareEvaluation(worker, std::move(request));
    else
      worker.evals.push_back(std::move(request));
  }
  const size_t batch_size = settings_.minibatch_size;
  // Small whole jobs leave room to collect additional cache misses without
  // exceeding the backend limit. Item credits bound all I/O, including stores.
  const size_t job_size = (batch_size + 3) / 4;
  while (HasJobCredit(worker)) {
    const size_t limit =
        std::min(job_size, 2 * batch_size - worker.outstanding_items);
    auto job = std::make_unique<Job>();
    job->worker = worker.index;
    job->queued_at = std::chrono::steady_clock::now();
    // Persistence has priority, so fast backends cannot grow a store backlog.
    if (!worker.persistence.empty()) {
      job->kind = JobKind::kPersist;
      while (!worker.persistence.empty() && job->entries.size() < limit) {
        job->entries.push_back(std::move(worker.persistence.front()));
        worker.persistence.pop_front();
      }
    } else {
      auto* requests = !worker.loads.empty() ? &worker.loads : &worker.evals;
      if (requests->empty()) break;
      job->kind = requests == &worker.loads ? JobKind::kLoad : JobKind::kEval;
      while (!requests->empty() && job->requests.size() < limit) {
        job->requests.push_back(std::move(requests->front()));
        requests->pop_front();
      }
    }
    ++worker.outstanding;
    worker.outstanding_items += job->requests.size() + job->entries.size();
    jobs_in_flight_.fetch_add(1);
    RaiseHighWater(metrics_.io_jobs_high_water,
                   static_cast<uint64_t>(jobs_in_flight_.load()));
    if (job->kind == JobKind::kEval) {
      eval_jobs_.Push(std::move(job));
      RaiseHighWater(metrics_.ready_eval_high_water,
                     static_cast<uint64_t>(eval_jobs_.Size()));
    } else {
      assert(store_);
      store_jobs_.Push(std::move(job));
    }
  }
}

void SearchRun::ConsumeJob(Worker& worker, std::unique_ptr<Job> job) {
  assert(worker.outstanding > 0);
  if (job->kind == JobKind::kEval) {
    for (auto& request : job->requests) PublishEvaluation(worker, request);
  } else if (job->kind == JobKind::kLoad) {
    for (auto& request : job->requests) {
      if (request.loaded) {
        metrics_.node_store_hits.fetch_add(1);
        metrics_.graph_nodes_rehydrated.fetch_add(1);
        CompleteMaterialization(worker, request.ticket,
                                std::move(*request.loaded), false);
      } else {
        metrics_.node_store_misses.fetch_add(1);
        PrepareEvaluation(worker, std::move(request));
      }
    }
  }
  --worker.outstanding;
  const size_t items = job->requests.size() + job->entries.size();
  assert(worker.outstanding_items >= items);
  worker.outstanding_items -= items;
  // Decrement last: publication and persistence enqueueing are part of the job.
  jobs_in_flight_.fetch_sub(1);
}

void SearchRun::StoreWorker() {
  std::unique_ptr<Job> job;
  while (store_jobs_.Pop(&job)) {
    if (job->kind == JobKind::kLoad) {
      std::vector<NodeKey> keys;
      for (const auto& request : job->requests) keys.push_back(request.key);
      auto loaded = store_->LoadBatch(keys);
      assert(loaded.size() == job->requests.size());
      for (size_t i = 0; i < loaded.size(); ++i)
        job->requests[i].loaded = std::move(loaded[i]);
      metrics_.node_store_load_batches.fetch_add(1);
      metrics_.node_store_load_keys.fetch_add(keys.size());
    } else {
      store_->StoreBatch(job->entries);
      metrics_.node_store_store_batches.fetch_add(1);
    }
    ReturnJob(std::move(job));
  }
}

void SearchRun::EvaluatorWorker() {
  std::unique_ptr<Job> first;
  const size_t target = settings_.minibatch_size;
  // Retained whole jobs remain counted as collecting and in-flight. Their
  // storage, like all added jobs, stays fixed until completion consumption.
  for (;;) {
    if (!first) {
      if (!eval_jobs_.Pop(&first)) break;
      collecting_jobs_.fetch_add(1);
    }
    auto computation = backend_->CreateComputation();
    std::vector<std::unique_ptr<Job>> jobs;
    const auto deadline =
        settings_.max_batch_delay_ms == 0
            ? std::chrono::steady_clock::time_point::max()
            : first->queued_at +
                  std::chrono::milliseconds(settings_.max_batch_delay_ms);
    auto job = std::move(first);
    bool timed_out = false;
    for (;;) {
      for (auto& request : job->requests) {
        auto& payload = request.payload;
        if (computation->AddInput(
                EvalPosition{.pos = request.history.GetPositions(),
                             .legal_moves = payload.moves},
                EvalResultPtr{.q = &payload.leaf_q,
                              .d = &payload.leaf_d,
                              .m = &payload.leaf_m,
                              .p = payload.priors}) ==
            BackendComputation::FETCHED_IMMEDIATELY)
          metrics_.cache_hits.fetch_add(1);
      }
      jobs.push_back(std::move(job));
      const auto used = computation->UsedBatchSize();
      if (used >= target) break;
      // An overdue job may have waited behind ComputeBlocking. Always consume
      // already-ready jobs first; its deadline forbids an additional wait,
      // rather than fragmenting that accumulated backlog into single jobs.
      if (!eval_jobs_.TryPop(&job)) {
        // Immediate-only jobs must return credits rather than waiting for
        // producers that need those cache results. Stop never waits. Zero
        // disables the timeout but still flushes dependency starvation.
        // The queue CV observes producer transitions as well as arrivals, and
        // the oldest job's absolute deadline is never extended.
        if (used == 0 || stop_mode_.load() != StopMode::kRunning) break;
        eval_jobs_.WakeAll();
        if (!eval_jobs_.PopUntil(&job, deadline, [&] {
              return stop_mode_.load() != StopMode::kRunning ||
                     EvaluationStarved();
            })) {
          timed_out = std::chrono::steady_clock::now() >= deadline;
          break;
        }
      }
      collecting_jobs_.fetch_add(1);
      // Whole jobs may include cache hits, but their conservative upper bound
      // must fit before AddInput. Retain a non-fitting job for the next batch.
      if (used + job->requests.size() > target) {
        first = std::move(job);
        break;
      }
    }
    // Decreasing collecting_jobs_ only disables the starvation predicate;
    // the increment handshake before PopUntil above is the enabling wake.
    collecting_jobs_.fetch_sub(jobs.size());
    const auto used = computation->UsedBatchSize();
    if (used > 0) {
      if (used < target) {
        if (stop_mode_.load() != StopMode::kRunning)
          metrics_.partial_drain_flushes.fetch_add(1);
        else if (timed_out)
          metrics_.partial_timeout_flushes.fetch_add(1);
        else
          metrics_.partial_starvation_flushes.fetch_add(1);
      }
      metrics_.evaluation_batches.fetch_add(1);
      metrics_.nn_evaluations.fetch_add(used);
      computation->ComputeBlocking();
    }
    computation.reset();
    for (auto& completed : jobs) ReturnJob(std::move(completed));
  }
}

void SearchRun::PublishEvaluation(Worker& worker, Request& request) {
  std::vector<size_t> order(request.payload.moves.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    if (request.payload.priors[a] != request.payload.priors[b])
      return request.payload.priors[a] > request.payload.priors[b];
    return request.payload.moves[a].raw_data() <
           request.payload.moves[b].raw_data();
  });
  ExpansionPayload sorted = request.payload;
  for (size_t i = 0; i < order.size(); ++i) {
    sorted.moves[i] = request.payload.moves[order[i]];
    sorted.priors[i] = request.payload.priors[order[i]];
  }
  CompleteMaterialization(worker, request.ticket, std::move(sorted), true);
}

void SearchRun::CompleteMaterialization(Worker& worker,
                                        MaterializationTicketId ticket_id,
                                        ExpansionPayload payload,
                                        bool store_payload) {
  Ticket ticket;
  uint64_t generation;
  bool accepted;
  {
    std::lock_guard lock(tickets_mutex_);
    auto it = tickets_.find(ticket_id);
    if (it == tickets_.end()) return;
    assert(it->second.owner.worker == worker.index);
    generation =
        graph_->InstallPayload(it->second.key, ticket_id, payload, &accepted);
    ticket = std::move(it->second);
    tickets_.erase(it);
    ticket_by_key_.erase(ticket.key);
  }
  RaiseHighWater(metrics_.graph_size_high_water,
                 static_cast<uint64_t>(graph_->Size()));
  if (!accepted) metrics_.publications_rejected.fetch_add(1);
  auto wake = [&](Continuation event, bool owner) {
    event.generation = generation;
    event.terminal = accepted && payload.terminal != TerminalKind::kNonTerminal;
    event.backup = accepted && (owner || event.terminal);
    event.value = {payload.leaf_q, payload.leaf_d, payload.leaf_m};
    if (event.worker == worker.index)
      Resume(worker, event);
    else
      worker.outgoing[event.worker].push_back(event);
  };
  wake(ticket.owner, true);
  for (auto event : ticket.waiters) wake(event, false);
  // Rejected publication resumes selection; its leaf is never backed up or
  // persisted. Accepted immutable payloads drain even after cancellation.
  if (accepted && store_ && store_payload)
    worker.persistence.push_back({ticket.key, std::move(payload)});
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
  RaiseHighWater(metrics_.max_depth, static_cast<uint64_t>(visit.path.size()));
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

void SearchRun::NotifyController() {
  {
    std::lock_guard lock(controller_mutex_);
    ++controller_generation_;
  }
  controller_cv_.notify_one();
}

void SearchRun::Controller() {
  auto next_info = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  for (;;) {
    uint64_t generation;
    {
      // Snapshot before inspecting completion predicates. Any enabling event
      // during inspection changes the generation and prevents sleeping.
      std::lock_guard lock(controller_mutex_);
      generation = controller_generation_;
    }
    const auto now = std::chrono::steady_clock::now();
    if (stop_mode_.load() == StopMode::kRunning && deadline_ &&
        now >= *deadline_)
      Stop();
    if (stop_mode_.load() == StopMode::kRunning && node_limit_ &&
        metrics_.visits_completed.load() >= *node_limit_)
      Stop();
    if (stop_mode_.load() == StopMode::kRunning && now >= next_info) {
      OutputInfo(false);
      next_info = now + std::chrono::seconds(1);
    }
    if (stop_mode_.load() != StopMode::kRunning && visits_.active() == 0) {
      bool no_tickets;
      {
        std::lock_guard lock(tickets_mutex_);
        no_tickets = tickets_.empty();
      }
      // Workers may have unpublished local requests or persistence. Require
      // their explicit idle acknowledgement, not merely empty executor queues.
      if (no_tickets && jobs_in_flight_.load() == 0) {
        bool idle = true;
        for (auto& worker : workers_) {
          std::lock_guard lock(worker->mailbox.mutex);
          if (!worker->mailbox.continuations.empty() ||
              !worker->mailbox.completions.empty() || !worker->drained.load())
            idle = false;
        }
        if (idle) break;
      }
    }
    auto wake_at = std::chrono::steady_clock::time_point::max();
    if (stop_mode_.load() == StopMode::kRunning) {
      wake_at = next_info;
      if (deadline_) wake_at = std::min(wake_at, *deadline_);
    }
    std::unique_lock lock(controller_mutex_);
    controller_cv_.wait_until(
        lock, wake_at, [&] { return controller_generation_ != generation; });
  }
  shutdown_.store(true);
  for (auto& worker : workers_) {
    {
      std::lock_guard lock(worker->mailbox.mutex);
    }
    worker->mailbox.cv.notify_one();
  }
  for (auto& thread : visit_threads_) thread.join();
  store_jobs_.Close();
  eval_jobs_.Close();
  for (auto& thread : store_threads_) thread.join();
  for (auto& thread : evaluator_threads_) thread.join();
  bool respond;
  {
    std::lock_guard lock(admission_mutex_);
    respond = stop_mode_.load() == StopMode::kRespondBestmove;
    if (respond) output_committed_.store(true);
  }
  if (respond) {
    OutputInfo(true);
    auto pv = BuildPv();
    BestMoveInfo info(pv.empty() ? FallbackMove() : pv.front(),
                      pv.size() > 1 ? pv[1] : Move{});
    responder_->OutputBestMove(&info);
  }
  finished_.store(true, std::memory_order_release);
}

std::vector<Move> SearchRun::BuildPv() const {
  std::vector<Move> pv;
  PositionHistory history = root_.history;
  NodeKey key = root_.key;
  std::unordered_set<uint64_t> seen;
  for (int ply = 0; ply < 256 && seen.insert(key.hash).second; ++ply) {
    auto node = graph_->SnapshotNode(key);
    if (!node || node->lifecycle != NodeLifecycle::kExpanded ||
        node->terminal != TerminalKind::kNonTerminal)
      break;
    const EdgeState* best = nullptr;
    for (const auto& edge : node->edges) {
      if (edge.visits == 0) continue;
      if (!best || edge.visits > best->visits ||
          (edge.visits == best->visits &&
           (edge.Q() > best->Q() ||
            (edge.Q() == best->Q() &&
             (edge.prior > best->prior ||
              (edge.prior == best->prior &&
               edge.move.raw_data() < best->move.raw_data()))))))
        best = &edge;
    }
    if (!best) break;
    Move output = best->move;
    if (history.Last().IsBlackToMove()) output.Flip();
    pv.push_back(output);
    history.Append(best->move);
    key = MakeNodeKey(history, settings_.history_key_length);
  }
  return pv;
}

Move SearchRun::FallbackMove() const {
  const auto legal = root_.history.Last().GetBoard().GenerateLegalMoves();
  if (legal.empty()) return Move{};
  Move best = *std::min_element(legal.begin(), legal.end(), [](Move a, Move b) {
    return a.raw_data() < b.raw_data();
  });
  if (root_.history.Last().IsBlackToMove()) best.Flip();
  return best;
}

void SearchRun::OutputInfo(bool final) {
  const auto elapsed = std::chrono::steady_clock::now() - start_time_;
  const int64_t ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  const uint64_t nodes = metrics_.visits_completed.load();
  const uint64_t evals = metrics_.nn_evaluations.load();
  ThinkingInfo info;
  info.time = ms;
  info.nodes = nodes;
  info.nps = ms > 0 ? static_cast<int>(nodes * 1000 / ms) : 0;
  info.eps = ms > 0 ? static_cast<int>(evals * 1000 / ms) : 0;
  info.depth = static_cast<int>(metrics_.max_depth.load());
  info.seldepth = static_cast<int>(metrics_.max_selected_depth.load());
  info.pv = BuildPv();
  if (auto root = graph_->SnapshotNode(root_.key); root && root->value.visits) {
    const float q = root->value.Q();
    const float d = std::clamp(root->value.D(), 0.0f, 1.0f);
    const int draw = static_cast<int>(d * 1000.0f);
    const int win = static_cast<int>((1.0f - d + q) * 500.0f);
    info.wdl = ThinkingInfo::WDL{std::clamp(win, 0, 1000), draw,
                                 std::clamp(1000 - win - draw, 0, 1000)};
  }
  info.comment =
      metrics_.Format(graph_->Size(), visits_.active(), eval_jobs_.Size());
  if (final) info.comment += " final";
  std::vector<ThinkingInfo> infos{std::move(info)};
  responder_->OutputThinkingInfo(&infos);
}

}  // namespace lczero::lc5
