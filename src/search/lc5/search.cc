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

#include "search/lc5/search.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

namespace lczero::lc5 {
namespace {

using std::chrono::steady_clock;
constexpr auto kInfoInterval = std::chrono::seconds(1);
constexpr auto kSmartPruningInterval = std::chrono::milliseconds(10);
// Visits to complete before the visit rate is trusted for early stopping.
constexpr uint64_t kSmartPruningMinVisits = 100;

// Moves in the graph are from the side to move's perspective; UCI is not.
Move ToUci(Move move, const Position& position) {
  if (position.IsBlackToMove()) move.Flip();
  return move;
}

// The PV edge: most visits, then best Q, then the usual tie-break.
const EdgeState* PvEdge(const NodeState& node) {
  const EdgeState* best = nullptr;
  for (const auto& edge : node.edges) {
    if (edge.value.visits == 0) continue;
    const auto rank = [](const EdgeState& e) {
      return std::pair(e.value.visits, e.Q());
    };
    if (!best || rank(edge) > rank(*best) ||
        (rank(edge) == rank(*best) && PreferOnTie(edge, *best))) {
      best = &edge;
    }
  }
  return best;
}

ThinkingInfo::WDL WdlFromValue(float q, float d) {
  d = std::clamp(d, 0.0f, 1.0f);
  const int draw = static_cast<int>(d * 1000.0f);
  const int win = static_cast<int>((1.0f - d + q) * 500.0f);
  return {std::clamp(win, 0, 1000), draw,
          std::clamp(1000 - win - draw, 0, 1000)};
}

int CentipawnsFromQ(float q) {
  return static_cast<int>(90 * std::tan(1.5637541897 * q));
}

}  // namespace

SearchRun::SearchRun(GameGraph* graph, NodeStore* store, Backend* backend,
                     UciResponder* responder, Settings::Resolved settings,
                     TimeManager::Config time_management, VisitOrigin root,
                     GoParams go_params, steady_clock::time_point start_time,
                     int64_t saved_ms)
    : graph_(graph),
      store_(store),
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
  if (!go_params.ponder && go_params.nodes && *go_params.nodes >= 0) {
    node_limit_ = static_cast<uint64_t>(*go_params.nodes);
  }
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
  for (auto& worker : workers_) {
    visit_threads_.emplace_back(&SearchRun::VisitWorker, this,
                                std::ref(*worker));
  }
  if (node_limit_ == 0) Stop();
  controller_thread_ = std::thread(&SearchRun::Controller, this);
}

void SearchRun::RequestStop(StopMode requested) {
  {
    // Admission and stop have one linearization point. Abort may override a
    // bestmove response until its output is committed.
    std::lock_guard lock(admission_mutex_);
    if (requested == StopMode::kAbort ? !output_committed_.load()
                                      : !Stopping()) {
      stop_mode_.store(requested);
    }
  }
  NotifyController();
}

void SearchRun::Stop() { RequestStop(StopMode::kRespondBestmove); }

void SearchRun::Abort() { RequestStop(StopMode::kAbort); }

void SearchRun::Wait() {
  if (controller_thread_.joinable()) controller_thread_.join();
}

void SearchRun::NotifyController() {
  {
    std::lock_guard lock(controller_mutex_);
    ++controller_generation_;
  }
  controller_cv_.notify_one();
}

void SearchRun::WakeAllWorkers() {
  for (auto& worker : workers_) worker->mailbox.Wake();
}

void SearchRun::Controller() {
  auto next_info = steady_clock::now() + kInfoInterval;
  bool responded = false;
  for (;;) {
    uint64_t generation;
    {
      // Snapshot before inspecting predicates: any enabling event during
      // inspection changes the generation and prevents sleeping.
      std::lock_guard lock(controller_mutex_);
      generation = controller_generation_;
    }
    const auto now = steady_clock::now();
    std::optional<steady_clock::time_point> next_check;
    if (!Stopping()) {
      const auto decision = time_manager_.Evaluate(now);
      next_check = decision.next_check;
      if (decision.should_stop ||
          (node_limit_ && metrics_.visits_completed.load() >= *node_limit_)) {
        Stop();
      } else if (time_manager_.smart_pruning_factor() > 0.0f) {
        if (BestMoveDecided(now)) {
          unused_time_ms_.store(
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  *time_manager_.deadline() - now)
                  .count());
          Stop();
        } else {
          next_check = std::min(*next_check, now + kSmartPruningInterval);
        }
      }
    }
    if (!Stopping() && now >= next_info) {
      std::vector<ThinkingInfo> infos{BuildInfo(false)};
      responder_->OutputThinkingInfo(&infos);
      next_info = now + kInfoInterval;
    }
    if (Stopping() && !responded) {
      Respond();
      responded = true;
    }
    // Stop may arrive after the check above. Never finish draining before
    // output has been committed or suppressed and sleepers were woken.
    if (responded && FullyDrained()) break;
    auto wake_at = steady_clock::time_point::max();
    if (!Stopping()) {
      wake_at = std::min(next_info, next_check.value_or(wake_at));
    }
    std::unique_lock lock(controller_mutex_);
    controller_cv_.wait_until(
        lock, wake_at, [&] { return controller_generation_ != generation; });
  }
  Shutdown();
}

// Whether the most visited root edge stays ahead even if all remaining visits,
// estimated from this run's rate so far, go to the runner-up.
bool SearchRun::BestMoveDecided(steady_clock::time_point now) const {
  const auto root = graph_->SnapshotNode(root_.key);
  if (!root || root->lifecycle != NodeLifecycle::kExpanded) return false;
  if (root->edges.size() == 1) return true;
  const uint64_t visits = metrics_.visits_completed.load();
  const auto elapsed = now - start_time_;
  if (visits < kSmartPruningMinVisits || elapsed.count() <= 0) return false;
  const double remaining =
      static_cast<double>(visits) * (*time_manager_.deadline() - now).count() /
      elapsed.count() / time_manager_.smart_pruning_factor();
  uint64_t best = 0;
  uint64_t second = 0;
  for (const auto& edge : root->edges) {
    if (edge.value.visits > best) {
      second = best;
      best = edge.value.visits;
    } else {
      second = std::max(second, edge.value.visits);
    }
  }
  return best - second > remaining;
}

void SearchRun::Respond() {
  // Capture once, without waiting for any executor or persistence. Abort can
  // suppress this response until commitment, but cannot retract it.
  if (stop_mode_.load() == StopMode::kRespondBestmove) {
    std::vector<ThinkingInfo> infos{BuildInfo(true)};
    const auto& pv = infos.front().pv;
    BestMoveInfo best(pv.empty() ? FallbackMove() : pv.front(),
                      pv.size() > 1 ? pv[1] : Move{});
    bool respond;
    {
      std::lock_guard lock(admission_mutex_);
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
  eval_jobs_.WakeAll();
}

bool SearchRun::FullyDrained() {
  if (visits_.active() != 0) return false;
  {
    std::lock_guard lock(tickets_mutex_);
    if (!tickets_.empty()) return false;
  }
  if (jobs_in_flight_.load() != 0) return false;
  // Workers may hold unpublished local requests or persistence. Require their
  // explicit idle acknowledgement, not merely empty executor queues.
  for (auto& worker : workers_) {
    std::lock_guard lock(worker->mailbox.mutex);
    if (!worker->mailbox.continuations.empty() ||
        !worker->mailbox.completions.empty() || !worker->drained.load()) {
      return false;
    }
  }
  return true;
}

void SearchRun::Shutdown() {
  shutdown_.store(true);
  WakeAllWorkers();
  for (auto& thread : visit_threads_) thread.join();
  store_jobs_.Close();
  eval_jobs_.Close();
  for (auto& thread : store_threads_) thread.join();
  for (auto& thread : evaluator_threads_) thread.join();
  finished_.store(true, std::memory_order_release);
}

std::vector<Move> SearchRun::BuildPv(std::optional<NodeState> node) const {
  constexpr int kMaxPvLength = 256;
  std::vector<Move> pv;
  PositionHistory history = root_.history;
  std::unordered_set<uint64_t> seen{root_.key.hash};
  while (node && pv.size() < kMaxPvLength &&
         node->lifecycle == NodeLifecycle::kExpanded &&
         node->terminal == TerminalKind::kNonTerminal) {
    const EdgeState* edge = PvEdge(*node);
    if (!edge) break;
    pv.push_back(ToUci(edge->move, history.Last()));
    history.Append(edge->move);
    const NodeKey key = MakeNodeKey(history, settings_.history_key_length);
    if (!seen.insert(key.hash).second) break;
    node = graph_->SnapshotNode(key);
  }
  return pv;
}

// Deterministic legal move for when no root edge has completed a visit.
Move SearchRun::FallbackMove() const {
  const Position& position = root_.history.Last();
  const auto legal = position.GetBoard().GenerateLegalMoves();
  if (legal.empty()) return Move{};
  return ToUci(std::ranges::min(legal, {}, &Move::raw_data), position);
}

ThinkingInfo SearchRun::BuildInfo(bool final) const {
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
  auto root = graph_->SnapshotNode(root_.key);
  if (root && root->value.visits) {
    info.score = CentipawnsFromQ(root->value.Q());
    info.wdl = WdlFromValue(root->value.Q(), root->value.D());
  }
  info.pv = BuildPv(std::move(root));
  info.comment =
      metrics_.Format(graph_->Size(), visits_.active(), eval_jobs_.Size());
  if (final) info.comment += " final";
  return info;
}

}  // namespace lczero::lc5
