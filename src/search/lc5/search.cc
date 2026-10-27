#include "search/lc5/search.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

namespace lczero::lc5 {
namespace {

using std::chrono::steady_clock;
constexpr auto kInfoInterval = std::chrono::seconds(1);

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
                     GoParams go_params, steady_clock::time_point start_time)
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
    if (!Stopping()) wake_at = std::min(next_info, next_check.value_or(wake_at));
    std::unique_lock lock(controller_mutex_);
    controller_cv_.wait_until(
        lock, wake_at, [&] { return controller_generation_ != generation; });
  }
  Shutdown();
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
