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

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "search/lc5/prior.h"
#include "search/lc5/search.h"

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

bool SearchRun::SuspendForMaterialization(Worker& worker,
                                          VisitPool::Slot& slot) {
  Visit& visit = slot.visit;
  bool owner;
  MaterializationTicketId ticket_id;
  {
    // Lock order: slot -> tickets -> one graph shard. No mailbox transfer here.
    std::lock_guard lock(tickets_mutex_);
    const auto existing = ticket_by_key_.find(visit.current_key);
    owner = existing == ticket_by_key_.end();
    ticket_id = owner ? next_ticket_++ : existing->second;
    const auto node =
        engine_->FindOrCreateMaterializing(visit.current_key, ticket_id);
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
  if (owner) {
    // Preparation and publication happen later, outside this slot's lock.
    Request request{.ticket = ticket_id,
                    .key = visit.current_key,
                    .history = visit.history};
    (store_ ? worker.loads : worker.unprepared).push_back(std::move(request));
  }
  return true;
}

void SearchRun::PrepareEvaluation(Worker& worker, Request request) {
  if (Stopping()) {
    CancelMaterialization(worker, request.ticket);
    return;
  }
  request.payload = DetectTerminal(request.history);
  if (request.payload.terminal != TerminalKind::kNonTerminal) {
    CompleteMaterialization(worker, request.ticket, std::move(request.payload),
                            true);
    return;
  }
  request.payload.priors.resize(request.payload.moves.size());
  request.state = Request::State::kPending;
  metrics_.eval_requests.fetch_add(1);
  worker.evals.push_back(std::move(request));
}

void SearchRun::CompleteMaterialization(Worker& worker,
                                        MaterializationTicketId ticket_id,
                                        ExpansionPayload payload,
                                        bool store_payload) {
  Ticket ticket;
  InstallResult installed;
  {
    std::lock_guard lock(tickets_mutex_);
    const auto it = tickets_.find(ticket_id);
    if (it == tickets_.end()) return;
    assert(it->second.owner.worker == worker.index);
    installed = engine_->Install(it->second.key, ticket_id, payload);
    ticket = std::move(it->second);
    tickets_.erase(it);
    ticket_by_key_.erase(ticket.key);
  }
  RaiseHighWater(metrics_.graph_size_high_water, engine_->GraphSize());
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
    Deliver(worker, event);
  };
  wake(ticket.owner, true);
  for (const auto& event : ticket.waiters) wake(event, false);
  // Accepted immutable payloads are persisted even after cancellation.
  if (installed.accepted && store_ && store_payload) {
    worker.persistence.push_back({ticket.key, std::move(payload)});
  }
}

void SearchRun::CancelMaterialization(Worker& worker,
                                      MaterializationTicketId ticket_id) {
  // Only called on stop, for evaluations that will never be computed. The
  // owner and waiters are still alive and must be woken to cancel.
  Ticket ticket;
  {
    std::lock_guard lock(tickets_mutex_);
    const auto it = tickets_.find(ticket_id);
    if (it == tickets_.end()) return;
    engine_->CancelMaterialization(it->second.key, ticket_id);
    ticket = std::move(it->second);
    tickets_.erase(it);
    ticket_by_key_.erase(ticket.key);
  }
  const auto wake = [&](Continuation event) {
    event.kind = Continuation::Kind::kCancel;
    Deliver(worker, event);
  };
  wake(ticket.owner);
  for (const auto& event : ticket.waiters) wake(event);
}

void SearchRun::CancelPendingRequests(Worker& worker) {
  for (auto* requests : {&worker.unprepared, &worker.loads, &worker.evals}) {
    for (const auto& request : *requests) {
      CancelMaterialization(worker, request.ticket);
    }
    requests->clear();
  }
}

bool SearchRun::HasJobCredit(const Worker& worker) const {
  return worker.outstanding < kBatchCredits &&
         worker.outstanding_items <
             2 * static_cast<size_t>(settings_.minibatch_size);
}

void SearchRun::SetProducing(Worker& worker, bool producing) {
  // Only true -> false enables the collectors' starvation predicate, so only
  // that edge needs the queue-mutex handshake.
  if (worker.producing.exchange(producing) && !producing) eval_jobs_.WakeAll();
}

void SearchRun::SubmitJobs(Worker& worker) {
  if (Stopping()) CancelPendingRequests(worker);
  // Detect terminals and generate legal moves for fresh requests outside any
  // slot lock. Terminal ones publish right away; the rest join `evals`.
  while (!worker.unprepared.empty()) {
    Request request = std::move(worker.unprepared.front());
    worker.unprepared.pop_front();
    PrepareEvaluation(worker, std::move(request));
  }
  while (HasJobCredit(worker)) {
    if (Stopping()) CancelPendingRequests(worker);
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
      assert(store_);
      store_jobs_.Push(std::move(job));
    }
  }
}

std::unique_ptr<SearchRun::Job> SearchRun::TakeJob(Worker& worker) const {
  // Small whole jobs leave room to collect more cache misses without exceeding
  // the backend batch. Item credits bound all I/O, including stores.
  const size_t batch_size = settings_.minibatch_size;
  const size_t limit =
      std::min((batch_size + 3) / 4, 2 * batch_size - worker.outstanding_items);
  auto job = std::make_unique<Job>();
  job->worker = worker.index;
  job->queued_at = std::chrono::steady_clock::now();
  // Persistence goes first, so fast backends cannot grow a store backlog.
  if (!worker.persistence.empty()) {
    job->kind = JobKind::kPersist;
    MoveFront(worker.persistence, job->entries, limit);
  } else if (!worker.loads.empty()) {
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

void SearchRun::ReturnJob(std::unique_ptr<Job> job) {
  auto& mailbox = workers_[job->worker]->mailbox;
  mailbox.Post([&] { mailbox.completions.push_back(std::move(job)); });
  metrics_.mailbox_notifications.fetch_add(1);
}

void SearchRun::ConsumeJob(Worker& worker, std::unique_ptr<Job> job) {
  assert(worker.outstanding > 0);
  for (auto& request : job->requests) {
    switch (request.state) {
      case Request::State::kEvaluated:
        SortByPrior(request.payload);
        CompleteMaterialization(worker, request.ticket,
                                std::move(request.payload), true);
        break;
      case Request::State::kStoreHit:
        metrics_.node_store_hits.fetch_add(1);
        metrics_.graph_nodes_rehydrated.fetch_add(1);
        CompleteMaterialization(worker, request.ticket,
                                std::move(request.payload), false);
        break;
      case Request::State::kStoreMiss:
        metrics_.node_store_misses.fetch_add(1);
        PrepareEvaluation(worker, std::move(request));
        break;
      case Request::State::kPending:
      case Request::State::kQueued:
        CancelMaterialization(worker, request.ticket);
        break;
    }
  }
  --worker.outstanding;
  assert(worker.outstanding_items >= job->items());
  worker.outstanding_items -= job->items();
  // Decrement last: publication and persistence enqueueing are part of the job.
  jobs_in_flight_.fetch_sub(1);
}

void SearchRun::StoreWorker() {
  std::unique_ptr<Job> job;
  while (store_jobs_.Pop(&job)) {
    if (job->kind == JobKind::kPersist) {
      store_->StoreBatch(job->entries);
      metrics_.node_store_store_batches.fetch_add(1);
    } else if (!Stopping()) {
      std::vector<NodeKey> keys;
      for (const auto& request : job->requests) keys.push_back(request.key);
      auto loaded = store_->LoadBatch(keys);
      assert(loaded.size() == job->requests.size());
      for (size_t i = 0; i < loaded.size(); ++i) {
        auto& request = job->requests[i];
        if (loaded[i]) {
          request.payload = std::move(*loaded[i]);
          request.state = Request::State::kStoreHit;
        } else {
          request.state = Request::State::kStoreMiss;
        }
      }
      metrics_.node_store_load_batches.fetch_add(1);
      metrics_.node_store_load_keys.fetch_add(keys.size());
    }
    ReturnJob(std::move(job));
  }
}

void SearchRun::EvaluatorWorker() {
  // A popped job that did not fit the previous batch. Retained jobs remain
  // counted as collecting and in flight.
  std::unique_ptr<Job> next;
  for (;;) {
    if (!next) {
      if (!eval_jobs_.Pop(&next)) break;
      collecting_jobs_.fetch_add(1);
    }
    if (Stopping()) {
      collecting_jobs_.fetch_sub(1);
      ReturnJob(std::move(next));
      continue;
    }
    RunBatch(CollectBatch(std::move(next), &next));
  }
}

SearchRun::Batch SearchRun::CollectBatch(std::unique_ptr<Job> job,
                                         std::unique_ptr<Job>* next) {
  const size_t target = settings_.minibatch_size;
  const auto deadline =
      settings_.max_batch_delay_ms == 0
          ? std::chrono::steady_clock::time_point::max()
          : job->queued_at +
                std::chrono::milliseconds(settings_.max_batch_delay_ms);
  Batch batch{.computation = backend_->CreateComputation()};
  for (;;) {
    AddInputs(*batch.computation, *job);
    batch.jobs.push_back(std::move(job));
    const size_t used = batch.computation->UsedBatchSize();
    if (used >= target || Stopping()) break;
    // Consume already-ready jobs first, even past the deadline, rather than
    // fragmenting an accumulated backlog into single-job batches.
    if (!eval_jobs_.TryPop(&job)) {
      // Immediate-only batches return credits instead of waiting on producers
      // that need those cache results. The oldest job's deadline is absolute;
      // zero disables it, but dependency starvation still flushes.
      if (used == 0 || Stopping()) break;
      eval_jobs_.WakeAll();
      if (!eval_jobs_.PopUntil(&job, deadline, [&] {
            return Stopping() || EvaluationStarved();
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

void SearchRun::AddInputs(BackendComputation& computation, Job& job) {
  for (auto& request : job.requests) {
    if (Stopping()) break;
    auto& payload = request.payload;
    const auto result =
        computation.AddInput(EvalPosition{.pos = request.history.GetPositions(),
                                          .legal_moves = payload.moves},
                             EvalResultPtr{.q = &payload.leaf.q,
                                           .d = &payload.leaf.d,
                                           .m = &payload.leaf.m,
                                           .p = payload.priors});
    if (result == BackendComputation::FETCHED_IMMEDIATELY) {
      request.state = Request::State::kEvaluated;
      metrics_.cache_hits.fetch_add(1);
    } else {
      request.state = Request::State::kQueued;
    }
  }
}

void SearchRun::RunBatch(Batch batch) {
  const size_t used = batch.computation->UsedBatchSize();
  bool authorized = false;
  if (used > 0) {
    // The final gate shares stop's linearization point. A serial evaluator
    // cannot authorize again until this computation has returned.
    std::lock_guard lock(admission_mutex_);
    authorized = !Stopping();
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
        if (request.state == Request::State::kQueued) {
          request.state = Request::State::kEvaluated;
        }
      }
    }
  }
  // Destroy the computation while the jobs it references are still alive.
  batch.computation.reset();
  for (auto& job : batch.jobs) ReturnJob(std::move(job));
}

bool SearchRun::EvaluationStarved() const {
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
