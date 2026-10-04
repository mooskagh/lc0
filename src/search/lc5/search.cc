#include "search/lc5/search.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace lczero::lc5 {

SearchRun::SearchRun(GameGraph* graph, NodeStore* store, Backend* backend,
                     UciResponder* responder, Settings::Resolved settings,
                     TimeManager::Config time_management, VisitOrigin root,
                     GoParams go_params,
                     std::chrono::steady_clock::time_point start_time)
    : graph_(graph),
      store_(store),
      backend_(backend),
      responder_(responder),
      settings_(settings),
      root_(std::move(root)),
      start_time_(start_time),
      time_manager_(time_management, go_params,
                    root_.history.Last().IsBlackToMove(), start_time_),
      visits_(settings_.max_active_visits) {
  for (int i = 0; i < settings_.threads; ++i)
    workers_.push_back(std::make_unique<Worker>(i));
  if (!go_params.ponder && go_params.nodes && *go_params.nodes >= 0)
    node_limit_ = static_cast<uint64_t>(*go_params.nodes);
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

void SearchRun::Controller() {
  auto next_info = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  bool stop_notified = false;
  for (;;) {
    uint64_t generation;
    {
      // Snapshot before inspecting completion predicates. Any enabling event
      // during inspection changes the generation and prevents sleeping.
      std::lock_guard lock(controller_mutex_);
      generation = controller_generation_;
    }
    const auto now = std::chrono::steady_clock::now();
    std::optional<std::chrono::steady_clock::time_point> next_check;
    if (stop_mode_.load() == StopMode::kRunning) {
      const auto decision = time_manager_.Evaluate(now);
      next_check = decision.next_check;
      if (decision.should_stop) Stop();
    }
    if (stop_mode_.load() == StopMode::kRunning && node_limit_ &&
        metrics_.visits_completed.load() >= *node_limit_)
      Stop();
    if (stop_mode_.load() == StopMode::kRunning && now >= next_info) {
      std::vector<ThinkingInfo> infos{BuildInfo(false)};
      responder_->OutputThinkingInfo(&infos);
      next_info = now + std::chrono::seconds(1);
    }
    if (stop_mode_.load() != StopMode::kRunning && !stop_notified) {
      // Capture once, without waiting for any executor or persistence. Abort
      // can suppress this snapshot until commitment, but cannot retract it.
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
      stop_notified = true;
      // Synchronize predicate changes with sleepers only after the response.
      for (auto& worker : workers_) {
        {
          std::lock_guard lock(worker->mailbox.mutex);
        }
        worker->mailbox.cv.notify_one();
      }
      eval_jobs_.WakeAll();
    }
    // Stop may arrive after the stop-phase check above. Never finish a drain
    // until that phase has committed/suppressed output and synchronized wakes.
    if (stop_notified && visits_.active() == 0) {
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
      if (next_check) wake_at = std::min(wake_at, *next_check);
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
  finished_.store(true, std::memory_order_release);
}

std::vector<Move> SearchRun::BuildPv(std::optional<NodeSnapshot> root) const {
  std::vector<Move> pv;
  PositionHistory history = root_.history;
  NodeKey key = root_.key;
  std::unordered_set<uint64_t> seen;
  for (int ply = 0; ply < 256 && seen.insert(key.hash).second; ++ply) {
    auto node = ply == 0 ? std::move(root) : graph_->SnapshotNode(key);
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

ThinkingInfo SearchRun::BuildInfo(bool final) const {
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
  auto root = graph_->SnapshotNode(root_.key);
  info.pv = BuildPv(root);
  if (root && root->value.visits) {
    const float q = root->value.Q();
    info.score = static_cast<int>(90 * std::tan(1.5637541897 * q));
    const float d = std::clamp(root->value.D(), 0.0f, 1.0f);
    const int draw = static_cast<int>(d * 1000.0f);
    const int win = static_cast<int>((1.0f - d + q) * 500.0f);
    info.wdl = ThinkingInfo::WDL{std::clamp(win, 0, 1000), draw,
                                 std::clamp(1000 - win - draw, 0, 1000)};
  }
  info.comment =
      metrics_.Format(graph_->Size(), visits_.active(), eval_jobs_.Size());
  if (final) info.comment += " final";
  return info;
}

}  // namespace lczero::lc5
