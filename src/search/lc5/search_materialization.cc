#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "search/lc5/search.h"

namespace lczero::lc5 {

void SearchRun::ReturnJob(std::unique_ptr<Job> job) {
  auto& mailbox = workers_[job->worker]->mailbox;
  {
    std::lock_guard lock(mailbox.mutex);
    mailbox.completions.push_back(std::move(job));
  }
  metrics_.mailbox_notifications.fetch_add(1);
  mailbox.cv.notify_one();
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
      detail::RaiseHighWater(metrics_.maximum_waiters_per_ticket,
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
  auto value = detail::TerminalValue(payload.terminal);
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
    detail::RaiseHighWater(metrics_.io_jobs_high_water,
                           static_cast<uint64_t>(jobs_in_flight_.load()));
    if (job->kind == JobKind::kEval) {
      eval_jobs_.Push(std::move(job));
      detail::RaiseHighWater(metrics_.ready_eval_high_water,
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
  detail::RaiseHighWater(metrics_.graph_size_high_water,
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

}  // namespace lczero::lc5
