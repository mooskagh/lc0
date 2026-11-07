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
// never so many that the root's visit count would overflow, and at least one if
// the root is not yet expanded.
uint64_t GetNodeLimit(const GoParams& go_params, const RootStats& root) {
  uint64_t limit = kMaxRootVisits - std::min(root.visits, kMaxRootVisits);
  if (!go_params.ponder && go_params.nodes && *go_params.nodes >= 0) {
    limit = std::min(limit, static_cast<uint64_t>(*go_params.nodes));
  }
  if (!root.expanded) limit = std::max<uint64_t>(limit, 1);
  return limit;
}

VisitLimiter::Config MakeLimiterConfig(const Settings::Resolved& settings,
                                       const GoParams& go_params,
                                       const RootStats& root) {
  return {.workers = static_cast<size_t>(settings.threads),
          .node_limit = GetNodeLimit(go_params, root),
          .root_visits_at_start = root.visits,
          .active_visits_ratio = settings.active_visits_ratio};
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
      backend_(backend),
      responder_(responder),
      settings_(settings),
      root_(std::move(root)),
      start_time_(start_time),
      time_manager_(time_management, go_params,
                    root_.history.Last().IsBlackToMove(), start_time_,
                    saved_ms),
      may_stop_early_(!go_params.infinite && !go_params.ponder),
      visit_pool_(settings_.max_active_visits),
      limiter_(MakeLimiterConfig(settings_, go_params,
                                 engine_->GetRootStats(root_.key)),
               visit_pool_, metrics_),
      inboxes_(settings_.threads),
      evaluator_({.backend = *backend_,
                  .settings = settings_,
                  .limiter = limiter_,
                  .metrics = metrics_,
                  .inboxes = inboxes_,
                  .jobs_in_flight = jobs_in_flight_}) {
  const VisitWorker::Context context{.engine = *engine_,
                                     .settings = settings_,
                                     .root = root_,
                                     .visits = visit_pool_,
                                     .metrics = metrics_,
                                     .limiter = limiter_,
                                     .materializations = materializations_,
                                     .inboxes = inboxes_,
                                     .evaluator = evaluator_,
                                     .store_jobs = store_jobs_,
                                     .jobs_in_flight = jobs_in_flight_,
                                     .controller_signal = controller_signal_};
  for (size_t i = 0; i < inboxes_.size(); ++i) {
    visit_workers_.push_back(std::make_unique<VisitWorker>(i, context));
  }
}

SearchSession::~SearchSession() {
  Abort();
  Wait();
}

void SearchSession::Start() {
  if (started_.exchange(true)) return;
  if (limiter_.node_limit() == 0 || time_manager_.ShouldStop(start_time_)) {
    Stop();
  }
  if (engine_->HasStore()) {
    for (int i = 0; i < settings_.eval_threads; ++i) {
      store_threads_.emplace_back(&SearchSession::RunStoreWorker, this);
    }
  }
  evaluator_.Start();
  for (auto& worker : visit_workers_) worker->Start();
  controller_thread_ = std::thread(&SearchSession::RunControllerWorker, this);
}

void SearchSession::Stop() {
  stop_requested_.store(true);
  if (metrics_.visits_completed.load() > 0 ||
      engine_->GetRootStats(root_.key).expanded) {
    limiter_.Stop();            // Close visit admission and set IsStopping().
  } else {
    limiter_.CapNodeLimit(1);   // Admit only the root visit before stopping.
  }
  controller_signal_.Notify();  // Wake controller to emit bestmove and drain.
}

void SearchSession::Abort() {
  aborted_.store(true);
  limiter_.Stop();
  controller_signal_.Notify();
}

void SearchSession::Wait() {
  if (controller_thread_.joinable()) controller_thread_.join();
}

void SearchSession::WakeAllVisitWorkers() {
  for (auto& inbox : inboxes_) inbox.Wake();
}

void SearchSession::RunControllerWorker() {
  auto next_info = steady_clock::now() + kInfoInterval;
  bool responded = false;
  for (;;) {
    // Before inspecting predicates: any enabling event during inspection
    // changes the generation and prevents sleeping.
    const uint64_t generation = controller_signal_.generation();
    const auto now = steady_clock::now();

    if (!IsStopping() &&
        (stop_requested_.load() || time_manager_.ShouldStop(now))) {
      limiter_.CapNodeLimit(1);
    }
    if (!IsStopping() && ShouldStop(now)) limiter_.Stop();
    SendEvictionJob();
    if (!IsStopping()) MaybeOutputInfo(now, next_info);
    if (IsStopping() && !responded) {
      RespondBestMove();
      responded = true;
    }
    // Stop may arrive after the check above. Never finish draining before
    // output has been committed or suppressed and sleepers were woken.
    if (responded && IsFullyDrained()) break;

    controller_signal_.WaitUntil(generation, NextWakeup(now, next_info));
  }
  JoinThreads();
}

void SearchSession::MaybeOutputInfo(steady_clock::time_point now,
                                    steady_clock::time_point& next_info) {
  const RootStats root = engine_->GetRootStats(root_.key);
  const int depth = static_cast<int>(metrics_.max_depth.load());
  const int seldepth = static_cast<int>(metrics_.max_selected_depth.load());
  const bool changed =
      root.expanded && (depth > 0 || seldepth > 0) &&
      (depth != last_depth_ || seldepth != last_seldepth_ ||
       root.best_move != last_best_move_);
  if (!changed && now < next_info) return;

  std::vector<ThinkingInfo> infos{BuildThinkingInfo(false)};
  last_depth_ = infos.front().depth;
  last_seldepth_ = infos.front().seldepth;
  last_best_move_ = root.best_move;
  responder_->OutputThinkingInfo(&infos);
  next_info = now + kInfoInterval;
}

bool SearchSession::ShouldStop(steady_clock::time_point now) const {
  const RootStats root = engine_->GetRootStats(root_.key);
  if (!root.expanded && metrics_.visits_completed.load() == 0) return false;
  if (stop_requested_.load() || time_manager_.ShouldStop(now) ||
      metrics_.visits_completed.load() >= limiter_.node_limit()) {
    return true;
  }
  // The root's result is proven, or there is only one move to play.
  if (may_stop_early_ && (root.decided || root.edges == 1)) {
    return true;
  }
  // The runner-up can't catch up within the remaining budget.
  return !time_manager_.CanAffordVisits(
      root.best_edge_visits - root.second_edge_visits,
      metrics_.visits_completed.load(), now);
}

steady_clock::time_point SearchSession::NextWakeup(
    steady_clock::time_point now, steady_clock::time_point next_info) const {
  if (IsStopping()) return steady_clock::time_point::max();
  auto wake_at = next_info;
  if (const auto deadline = time_manager_.deadline();
      deadline && *deadline > now) {
    wake_at = std::min(wake_at, *deadline);
  }
  if (may_stop_early_) wake_at = std::min(wake_at, now + kSmartPruningInterval);
  if (HasNodeLimit()) wake_at = std::min(wake_at, now + kEvictionInterval);
  return wake_at;
}

void SearchSession::RespondBestMove() {
  // The search ends here as far as the clock is concerned.
  unused_time_ms_.store(time_manager_.UnusedMs(steady_clock::now()));
  // Capture once, without waiting for any executor or eviction. Abort can
  // suppress this response before it is sent.
  if (!aborted_.load()) {
    std::vector<ThinkingInfo> infos{BuildThinkingInfo(true)};
    const auto& pv = infos.front().pv;
    BestMoveInfo best(pv.empty() ? GetFallbackMove() : pv.front(),
                      pv.size() > 1 ? pv[1] : Move{});
    if (!aborted_.load()) {
      responder_->OutputThinkingInfo(&infos);
      responder_->OutputBestMove(&best);
    }
  }
  // Synchronize the stop transition with sleepers only after the response.
  WakeAllVisitWorkers();
  evaluator_.NotifyStop();
}

bool SearchSession::IsFullyDrained() {
  if (visit_pool_.active() != 0 || !materializations_.empty()) return false;
  if (jobs_in_flight_.load() != 0 || evictions_in_flight_.load() != 0) {
    return false;
  }
  // Workers may hold unpublished local materializations. Require their
  // explicit idle acknowledgement, not merely empty executor queues.
  for (auto& worker : visit_workers_) {
    if (!worker->IsDrained()) return false;
  }
  return true;
}

void SearchSession::JoinThreads() {
  for (auto& inbox : inboxes_) inbox.Shutdown();
  for (auto& worker : visit_workers_) worker->Join();
  store_jobs_.Close();
  for (auto& thread : store_threads_) thread.join();
  evaluator_.Join();
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
  info.nodes = limiter_.root_visits_at_start() + nodes;
  info.nps = ms > 0 ? static_cast<int>(nodes * 1000 / ms) : 0;
  info.eps = ms > 0 ? static_cast<int>(evals * 1000 / ms) : 0;
  info.depth = static_cast<int>(metrics_.max_depth.load());
  info.seldepth = static_cast<int>(metrics_.max_selected_depth.load());
  engine_->FillThinkingInfo(root_, info);
  info.comment = metrics_.Format(engine_->GetGraphSize(), visit_pool_.active(),
                                 evaluator_.queue_size());
  if (final) info.comment += " final";
  return info;
}

//////////////////////////////////////////////////////////////////////////////
// Store threads: they run load jobs and return them, and evictions.
//////////////////////////////////////////////////////////////////////////////

void SearchSession::ReturnJobToWorker(std::unique_ptr<AsyncJob> job) {
  inboxes_[job->worker].AddCompletion(std::move(job));
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
  controller_signal_.TakeEvictionRequests();
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
  controller_signal_.Notify();
}

}  // namespace lczero::lc5
