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

#include <absl/synchronization/mutex.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace lczero::lc5 {
namespace {

using std::chrono::steady_clock;
constexpr auto kInfoInterval = std::chrono::seconds(1);
constexpr auto kSmartPruningInterval = std::chrono::milliseconds(10);
// How often the controller checks the graph against its node limit.
constexpr auto kEvictionInterval = std::chrono::milliseconds(10);
// Root visits are counted in uint32_t; classic caps a search the same way.
constexpr uint64_t kMaxRootVisits = 4'000'000'000;

// Visits a search may complete: `go nodes`, if given and not pondering, but
// never so many that the root's visit count would overflow.
uint64_t GetNodeLimit(const GoParams& go_params, uint64_t root_visits) {
  uint64_t limit = kMaxRootVisits - std::min(root_visits, kMaxRootVisits);
  if (!go_params.ponder && go_params.nodes && *go_params.nodes >= 0) {
    limit = std::min(limit, static_cast<uint64_t>(*go_params.nodes));
  }
  return limit;
}

// Moves in the graph are from the side to move's perspective; UCI is not.
Move ToUci(Move move, const Position& position) {
  if (position.IsBlackToMove()) move.Flip();
  return move;
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
      root_visits_at_start_(engine_->GetRootStats(root_.key).visits),
      start_time_(start_time),
      time_manager_(time_management, go_params,
                    root_.history.Last().IsBlackToMove(), start_time_,
                    saved_ms),
      node_limit_(GetNodeLimit(go_params, root_visits_at_start_)),
      may_stop_early_(!go_params.infinite && !go_params.ponder),
      visits_(settings_.max_active_visits),
      gate_({.workers = static_cast<size_t>(settings_.threads),
             .node_limit = node_limit_,
             .root_visits_at_start = root_visits_at_start_,
             .active_visits_ratio = settings_.active_visits_ratio},
            visits_, metrics_),
      inboxes_(settings_.threads) {
  const VisitWorker::Context context{.engine = *engine_,
                                     .settings = settings_,
                                     .root = root_,
                                     .visits = visits_,
                                     .metrics = metrics_,
                                     .gate = gate_,
                                     .tickets = tickets_,
                                     .inboxes = inboxes_,
                                     .eval_jobs = eval_jobs_,
                                     .store_jobs = store_jobs_,
                                     .jobs_in_flight = jobs_in_flight_,
                                     .controller = controller_};
  for (size_t i = 0; i < inboxes_.size(); ++i) {
    workers_.push_back(std::make_unique<VisitWorker>(i, context));
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
      store_threads_.emplace_back(&SearchSession::RunStoreWorker, this);
    evaluator_threads_.emplace_back(&SearchSession::RunEvaluatorWorker, this);
  }
  for (auto& worker : workers_) {
    visit_threads_.emplace_back(&VisitWorker::Run, worker.get());
  }
  if (node_limit_ == 0) Stop();
  controller_thread_ = std::thread(&SearchSession::RunControllerWorker, this);
}

void SearchSession::RequestStop(StopMode mode) {
  gate_.RequestStop(mode);
  controller_.Notify();
}

void SearchSession::Stop() { RequestStop(StopMode::kRespondBestmove); }

void SearchSession::Abort() { RequestStop(StopMode::kAbort); }

void SearchSession::Wait() {
  if (controller_thread_.joinable()) controller_thread_.join();
}

void SearchSession::WakeAllWorkers() {
  for (auto& inbox : inboxes_) inbox.Wake();
}

void SearchSession::RunControllerWorker() {
  auto next_info = steady_clock::now() + kInfoInterval;
  bool responded = false;
  for (;;) {
    // Before inspecting predicates: any enabling event during inspection
    // changes the generation and prevents sleeping.
    const uint64_t generation = controller_.generation();
    const auto now = steady_clock::now();
    std::optional<steady_clock::time_point> next_check;
    if (!IsStopping()) {
      next_check = time_manager_.deadline();
      const RootStats root = engine_->GetRootStats(root_.key);
      if (time_manager_.ShouldStop(now) ||
          metrics_.visits_completed.load() >= node_limit_) {
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
        SendEvictionJob();
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
    controller_.WaitUntil(generation, wake_at);
  }
  JoinThreads();
}

void SearchSession::RespondBestMove() {
  // The search ends here as far as the clock is concerned.
  unused_time_ms_.store(time_manager_.UnusedMs(steady_clock::now()));
  // Capture once, without waiting for any executor or eviction. Abort can
  // suppress this response until commitment, but cannot retract it.
  if (gate_.stop_mode() == StopMode::kRespondBestmove) {
    std::vector<ThinkingInfo> infos{BuildThinkingInfo(true)};
    const auto& pv = infos.front().pv;
    BestMoveInfo best(pv.empty() ? GetFallbackMove() : pv.front(),
                      pv.size() > 1 ? pv[1] : Move{});
    if (gate_.TryCommitOutput()) {
      responder_->OutputThinkingInfo(&infos);
      responder_->OutputBestMove(&best);
    }
  }
  // Synchronize the stop transition with sleepers only after the response.
  WakeAllWorkers();
  eval_jobs_.NotifyPredicateChanged();
}

bool SearchSession::IsFullyDrained() {
  if (visits_.active() != 0 || !tickets_.empty()) return false;
  if (jobs_in_flight_.load() != 0 || evictions_in_flight_.load() != 0) {
    return false;
  }
  // Workers may hold unpublished local materializations. Require their
  // explicit idle acknowledgement, not merely empty executor queues.
  for (auto& worker : workers_) {
    if (!worker->IsDrained()) return false;
  }
  return true;
}

void SearchSession::JoinThreads() {
  for (auto& inbox : inboxes_) inbox.Shutdown();
  for (auto& thread : visit_threads_) thread.join();
  store_jobs_.Close();
  eval_jobs_.Close();
  for (auto& thread : store_threads_) thread.join();
  for (auto& thread : evaluator_threads_) thread.join();
  finished_.store(true, std::memory_order_release);
}

// Deterministic legal move for when the root has no PV: it was never
// expanded, or it is terminal.
Move SearchSession::GetFallbackMove() const {
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
// Store threads and evaluators: they run jobs and return them.
//////////////////////////////////////////////////////////////////////////////

void SearchSession::ReturnJobToWorker(std::unique_ptr<AsyncJob> job) {
  {
    auto& inbox = inboxes_[job->worker];
    absl::MutexLock lock(inbox.mutex);
    inbox.completions.push_back(std::move(job));
  }
  metrics_.inbox_notifications.fetch_add(1);
}

void SearchSession::RunStoreWorker() {
  StoreJob job;
  while (store_jobs_.Pop(&job)) {
    if (const auto* eviction = std::get_if<EvictionJob>(&job)) {
      RunEvictionJob(*eviction);
      continue;
    }
    auto& load = std::get<std::unique_ptr<AsyncJob>>(job);
    if (!IsStopping()) RunLoadJob(*load);
    ReturnJobToWorker(std::move(load));
  }
}

void SearchSession::RunLoadJob(AsyncJob& job) {
  std::vector<NodeKey> keys;
  for (const auto& materialization : job.materializations)
    keys.push_back(materialization.key);
  auto loaded = engine_->ReadFromStore(keys);
  assert(loaded.size() == job.materializations.size());
  for (size_t i = 0; i < loaded.size(); ++i) {
    auto& materialization = job.materializations[i];
    if (loaded[i]) {
      materialization.record = std::move(*loaded[i]);
      materialization.state = Materialization::State::kStoreHit;
    } else {
      materialization.state = Materialization::State::kStoreMiss;
    }
  }
  metrics_.node_store_load_batches.fetch_add(1);
  metrics_.node_store_load_keys.fetch_add(keys.size());
}

void SearchSession::SendEvictionJob() {
  if (!HasNodeLimit() || IsStopping() || evictions_in_flight_.load() > 0)
    return;
  controller_.TakeEvictionRequests();
  const size_t limit = settings_.graph_node_limit;
  const size_t size = engine_->GetGraphSize();
  if (size <= limit) return;
  // Evict down to a low-water mark, so the next round is not due right away.
  const size_t target = limit - limit / 10;
  evictions_in_flight_.fetch_add(1);
  store_jobs_.Push(EvictionJob{.count = size - target});
}

void SearchSession::RunEvictionJob(const EvictionJob& job) {
  engine_->EvictToStore(job.count, root_.key, metrics_);
  // The controller may queue the next round, or finish draining.
  evictions_in_flight_.fetch_sub(1);
  controller_.Notify();
}

void SearchSession::RunEvaluatorWorker() {
  std::unique_ptr<AsyncJob> job;
  while (eval_jobs_.Pop(&job)) {
    if (IsStopping()) {
      ReturnJobToWorker(std::move(job));
      continue;
    }
    collecting_jobs_.fetch_add(1);
    EvaluateBatch(CollectBatch(std::move(job)));
  }
}

Batch SearchSession::CollectBatch(std::unique_ptr<AsyncJob> job) {
  const size_t target = settings_.minibatch_size;
  const auto deadline =
      settings_.max_batch_delay_ms == 0
          ? std::chrono::steady_clock::time_point::max()
          : job->queued_at +
                std::chrono::milliseconds(settings_.max_batch_delay_ms);
  Batch batch{.computation = backend_->CreateComputation()};
  for (;;) {
    AddJobToBatch(*batch.computation, *job);
    batch.jobs.push_back(std::move(job));
    const size_t used = batch.computation->UsedBatchSize();
    if (used >= target || IsStopping()) break;
    // A whole job's size bounds its uncached inputs and must fit before
    // AddInput. A job that doesn't fit stays queued and starts a later batch.
    const auto fits = [&](const std::unique_ptr<AsyncJob>& next) {
      return used + next->materializations.size() <= target;
    };
    // Consume already-ready jobs first, even past the deadline, rather than
    // fragmenting an accumulated backlog into single-job batches.
    if (!eval_jobs_.TryPopIf(&job, fits)) {
      // Immediate-only batches return their jobs instead of waiting on
      // producers that need those cache results. The oldest job's deadline is
      // absolute; zero disables it, but dependency starvation still flushes.
      if (used == 0 || IsStopping()) break;
      eval_jobs_.NotifyPredicateChanged();
      if (!eval_jobs_.PopUntil(
              &job, deadline,
              [&] { return IsStopping() || IsEvaluationStarved(); }, fits)) {
        batch.timed_out = std::chrono::steady_clock::now() >= deadline;
        break;
      }
    }
    collecting_jobs_.fetch_add(1);
  }
  // Decreasing collecting_jobs_ only disables the starvation predicate; the
  // increment before PopUntil above is the enabling handshake.
  collecting_jobs_.fetch_sub(batch.jobs.size());
  return batch;
}

void SearchSession::AddJobToBatch(BackendComputation& computation,
                                  AsyncJob& job) {
  for (auto& materialization : job.materializations) {
    if (IsStopping()) break;
    auto& payload = materialization.payload;
    const auto result = computation.AddInput(
        EvalPosition{.pos = materialization.history.GetPositions(),
                     .legal_moves = payload.moves},
        EvalResultPtr{.q = &payload.leaf.q,
                      .d = &payload.leaf.d,
                      .m = &payload.leaf.m,
                      .p = payload.priors});
    if (result == BackendComputation::FETCHED_IMMEDIATELY) {
      materialization.state = Materialization::State::kEvaluated;
      metrics_.cache_hits.fetch_add(1);
    } else {
      materialization.state = Materialization::State::kQueued;
    }
  }
}

void SearchSession::EvaluateBatch(Batch batch) {
  const size_t used = batch.computation->UsedBatchSize();
  if (used > 0 && gate_.AuthorizeEval()) {
    if (used < static_cast<size_t>(settings_.minibatch_size)) {
      (batch.timed_out ? metrics_.partial_timeout_flushes
                       : metrics_.partial_starvation_flushes)
          .fetch_add(1);
    }
    metrics_.evaluation_batches.fetch_add(1);
    metrics_.nn_evaluations.fetch_add(used);
    batch.computation->ComputeBlocking();
    for (auto& job : batch.jobs) {
      for (auto& materialization : job->materializations) {
        if (materialization.state == Materialization::State::kQueued) {
          materialization.state = Materialization::State::kEvaluated;
        }
      }
    }
  }
  // Destroy the computation while the jobs it references are still alive.
  batch.computation.reset();
  for (auto& job : batch.jobs) ReturnJobToWorker(std::move(job));
}

bool SearchSession::IsEvaluationStarved() const {
  // Executing, store or returned jobs can still unblock a producer. Only
  // predict starvation when every outstanding job is held by a collector and
  // no worker can send a job independently; the deadline bounds the
  // uncertainty.
  for (const auto& worker : workers_) {
    if (worker->IsProducing()) return false;
  }
  // Zero disables the timeout, not dependency flushing: with no deadline,
  // currently blocked producers cannot justify an unbounded wait.
  return settings_.max_batch_delay_ms == 0 ||
         jobs_in_flight_.load() <= collecting_jobs_.load();
}

}  // namespace lczero::lc5
