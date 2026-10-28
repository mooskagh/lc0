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
#include <limits>
#include <utility>

namespace lczero::lc5 {
namespace {

using std::chrono::steady_clock;
constexpr auto kInfoInterval = std::chrono::seconds(1);
// Longest wait between stopper checks.
constexpr int64_t kMaxStopperIntervalMs = 10;

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
                     std::unique_ptr<classic::SearchStopper> stopper,
                     VisitOrigin root, GoParams go_params,
                     steady_clock::time_point start_time)
    : graph_(graph),
      store_(store),
      backend_(backend),
      responder_(responder),
      settings_(settings),
      root_(std::move(root)),
      start_time_(start_time),
      stopper_(std::move(stopper)),
      visits_(settings_.max_active_visits) {
  for (int i = 0; i < settings_.threads; ++i) {
    workers_.push_back(std::make_unique<Worker>(i));
  }
  if (const auto node = graph_->SnapshotNode(root_.key)) {
    root_visits_at_start_ = node->value.visits;
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
      if (node_limit_ && metrics_.visits_completed.load() >= *node_limit_) {
        Stop();
      } else if (stopper_) {
        const auto stats = PopulateIterationStats(now);
        // As in classic, stoppers wait until the root has been evaluated.
        int64_t wait_ms = kMaxStopperIntervalMs;
        if (stats.total_nodes > 0) {
          hints_.Reset();
          if (stopper_->ShouldStop(stats, &hints_)) {
            Stop();
          } else {
            wait_ms = std::clamp<int64_t>(hints_.GetEstimatedRemainingTimeMs(),
                                          1, kMaxStopperIntervalMs);
          }
        }
        next_check = now + std::chrono::milliseconds(wait_ms);
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

classic::IterationStats SearchRun::PopulateIterationStats(
    steady_clock::time_point now) const {
  classic::IterationStats stats;
  stats.time_since_movestart =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_)
          .count();
  const int64_t first_visit_ms = first_visit_ms_.load();
  if (first_visit_ms >= 0) {
    stats.time_since_first_batch =
        std::max<int64_t>(0, stats.time_since_movestart - first_visit_ms);
  }
  const uint64_t completed = metrics_.visits_completed.load();
  stats.nodes_since_movestart = completed;
  stats.batches_since_movestart = metrics_.evaluation_batches.load();
  stats.average_depth = static_cast<int>(metrics_.cumulative_depth.load() /
                                         std::max<uint64_t>(1, completed));
  // No proven results yet, so no mates, wins or losing edges are reported.
  const auto root = graph_->SnapshotNode(root_.key);
  if (!root) return stats;
  stats.total_nodes = root->value.visits;
  stats.edge_n.reserve(root->edges.size());
  for (const auto& edge : root->edges) {
    stats.edge_n.push_back(
        static_cast<uint32_t>(std::min<uint64_t>(
        edge.value.visits, std::numeric_limits<uint32_t>::max())));
  }
  // Ask for more time while the most visited move is not the best by Q.
  if (const EdgeState* pv = PvEdge(*root)) {
    for (const auto& edge : root->edges) {
      if (edge.value.visits > 0 && edge.Q() > pv->Q()) {
        stats.time_usage_hint_ =
            classic::IterationStats::TimeUsageHint::kNeedMoreTime;
        break;
      }
    }
  }
  return stats;
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
      // Lets the time manager bank the time this move left unused.
      if (stopper_) stopper_->OnSearchDone(PopulateIterationStats(steady_clock::now()));
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
