#include "search/lc5/search.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <sstream>
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
  if (!go_params_.ponder && go_params_.nodes && *go_params_.nodes >= 0) {
    node_limit_ = static_cast<uint64_t>(*go_params_.nodes);
  }
  if (!go_params_.ponder && go_params_.movetime) {
    deadline_ = start_time_ + std::chrono::milliseconds(*go_params_.movetime);
  }
}

SearchRun::~SearchRun() {
  Abort();
  Wait();
}

void SearchRun::Start() {
  if (started_.exchange(true)) return;
  for (int i = 0; i < settings_.threads; ++i) {
    visit_threads_.emplace_back(&SearchRun::VisitWorker, this);
  }
  for (int i = 0; i < settings_.eval_threads; ++i) {
    store_threads_.emplace_back(&SearchRun::StoreWorker, this);
    evaluator_threads_.emplace_back(&SearchRun::EvaluatorWorker, this);
  }
  AdmitMore();
  controller_thread_ = std::thread(&SearchRun::Controller, this);
}

void SearchRun::RequestStop(StopMode requested) {
  StopMode current = stop_mode_.load(std::memory_order_relaxed);
  for (;;) {
    StopMode desired = current;
    if (requested == StopMode::kAbort) {
      if (output_committed_.load(std::memory_order_acquire)) return;
      desired = StopMode::kAbort;
    } else if (current == StopMode::kRunning) {
      desired = StopMode::kRespondBestmove;
    }
    if (desired == current || stop_mode_.compare_exchange_weak(current, desired))
      break;
  }
  ready_visits_.WakeAll();
  store_requests_.WakeAll();
  ready_evals_.WakeAll();
  NotifyController();
}

void SearchRun::Stop() { RequestStop(StopMode::kRespondBestmove); }
void SearchRun::Abort() { RequestStop(StopMode::kAbort); }

void SearchRun::Wait() {
  if (!started_) return;
  if (controller_thread_.joinable()) controller_thread_.join();
  for (auto& thread : visit_threads_)
    if (thread.joinable()) thread.join();
  for (auto& thread : store_threads_)
    if (thread.joinable()) thread.join();
  for (auto& thread : evaluator_threads_)
    if (thread.joinable()) thread.join();
}

bool SearchRun::Admit(const VisitOrigin& origin) {
  if (stop_mode_.load(std::memory_order_acquire) != StopMode::kRunning)
    return false;
  auto id = visits_.Allocate(origin);
  if (!id) return false;
  metrics_.visits_admitted.fetch_add(1);
  RaiseHighWater(metrics_.active_visits_high_water,
                 static_cast<uint64_t>(visits_.active()));
  ready_visits_.Push(*id);
  RaiseHighWater(metrics_.visits_ready_high_water,
                 static_cast<uint64_t>(ready_visits_.Size()));
  return true;
}

void SearchRun::AdmitMore() {
  std::lock_guard lock(admission_mutex_);
  while (stop_mode_.load(std::memory_order_relaxed) == StopMode::kRunning &&
         visits_.active() < visits_.capacity() &&
         (!node_limit_ || admitted_ < *node_limit_)) {
    auto id = visits_.Allocate(root_);
    if (!id) break;
    ++admitted_;
    metrics_.visits_admitted.fetch_add(1);
    RaiseHighWater(metrics_.active_visits_high_water,
                   static_cast<uint64_t>(visits_.active()));
    ready_visits_.Push(*id);
    RaiseHighWater(metrics_.visits_ready_high_water,
                   static_cast<uint64_t>(ready_visits_.Size()));
  }
  if (node_limit_ && admitted_ == 0 && *node_limit_ == 0) {
    RequestStop(StopMode::kRespondBestmove);
  }
}

void SearchRun::VisitWorker() {
  VisitId id;
  while (ready_visits_.Pop(&id)) AdvanceVisit(id);
}

void SearchRun::AdvanceVisit(VisitId id) {
  VisitPool::Slot* slot = visits_.Lookup(id);
  if (!slot) return;
  std::unique_lock lock(slot->mutex);
  if (slot->epoch.load(std::memory_order_relaxed) != id.epoch() ||
      slot->visit.state == VisitState::kFree)
    return;
  if (stop_mode_.load(std::memory_order_acquire) != StopMode::kRunning ||
      slot->visit.state == VisitState::kCancelling) {
    slot->visit.state = VisitState::kCancelling;
    Cancel(*slot);
    return;
  }
  if (slot->visit.state == VisitState::kReadyBackup) {
    Backup(*slot);
    return;
  }
  if (slot->visit.state != VisitState::kReadySelect) return;

  advancing_visits_.fetch_add(1);
  for (;;) {
    Visit& visit = slot->visit;
    if (stop_mode_.load(std::memory_order_relaxed) != StopMode::kRunning) {
      advancing_visits_.fetch_sub(1);
      visit.state = VisitState::kCancelling;
      Cancel(*slot);
      return;
    }
    if (visit.path.empty() || visit.path.back().key != visit.current_key) {
      visit.path.push_back(PathStep{.key = visit.current_key,
                                    .generation = 0,
                                    .selected_move = std::nullopt});
    }
    PathStep& step = visit.path.back();
    auto snapshot = graph_->SnapshotNode(visit.current_key);
    metrics_.selection_node_steps.fetch_add(1);
    RaiseHighWater(metrics_.max_selected_depth,
                   static_cast<uint64_t>(visit.path.size()));
    if (!snapshot || snapshot->lifecycle == NodeLifecycle::kMaterializing) {
      if (snapshot) step.generation = snapshot->generation;
      advancing_visits_.fetch_sub(1);
      if (SuspendForMaterialization(*slot)) NotifyController();
      return;
    }
    step.generation = snapshot->generation;
    if (snapshot->terminal != TerminalKind::kNonTerminal) {
      visit.result = TerminalValue(snapshot->terminal);
      visit.state = VisitState::kReadyBackup;
      metrics_.terminal_visits.fetch_add(1);
      advancing_visits_.fetch_sub(1);
      Backup(*slot);
      return;
    }
    SelectResult selected = graph_->SelectAndReserve(
        visit.current_key, step.generation, settings_);
    if (selected.status == SelectStatus::kStale ||
        selected.status == SelectStatus::kMissing) {
      step.generation = selected.generation;
      continue;
    }
    if (selected.status == SelectStatus::kMaterializing) {
      advancing_visits_.fetch_sub(1);
      SuspendForMaterialization(*slot);
      return;
    }
    if (selected.status == SelectStatus::kTerminal) {
      visit.result = TerminalValue(snapshot->terminal);
      visit.state = VisitState::kReadyBackup;
      advancing_visits_.fetch_sub(1);
      Backup(*slot);
      return;
    }
    step.selected_move = selected.move;
    visit.history.Append(selected.move);
    visit.current_key = MakeNodeKey(visit.history, settings_.history_key_length);
  }
}

bool SearchRun::SuspendForMaterialization(VisitPool::Slot& slot) {
  Visit& visit = slot.visit;
  // Serialize graph reconciliation and waiter registration with publication.
  // The caller holds slot.mutex; completion releases tickets_mutex_ before wakeup.
  std::lock_guard tickets_lock(tickets_mutex_);
  auto existing = ticket_by_key_.find(visit.current_key);
  const bool owner = existing == ticket_by_key_.end();
  const MaterializationTicketId ticket_id =
      owner ? next_ticket_.fetch_add(1) : existing->second;
  const FindOrCreateResult graph_result =
      graph_->FindOrCreateMaterializing(visit.current_key, ticket_id);
  if (graph_result.created) metrics_.graph_nodes_created.fetch_add(1);
  if (graph_result.lifecycle == NodeLifecycle::kExpanded) {
    // The selection snapshot predates publication. No ticket or store request
    // is needed; select again from the published graph.
    visit.state = VisitState::kReadySelect;
    ready_visits_.Push(visit.id);
    return false;
  }
  assert(graph_result.ticket == ticket_id);
  if (owner) {
    Ticket ticket{.id = ticket_id,
                  .key = visit.current_key,
                  .owner = visit.id,
                  .waiters = {},
                  .owner_history = visit.history,
                  .created_at = std::chrono::steady_clock::now()};
    tickets_.emplace(ticket_id, std::move(ticket));
    ticket_by_key_.emplace(visit.current_key, ticket_id);
    metrics_.tickets_created.fetch_add(1);
  } else {
    auto& ticket = tickets_.at(ticket_id);
    ticket.waiters.push_back(visit.id);
    metrics_.ticket_waiters.fetch_add(1);
    RaiseHighWater(metrics_.maximum_waiters_per_ticket,
                   static_cast<uint64_t>(ticket.waiters.size()));
  }
  visit.path.back().generation = graph_result.generation;
  visit.waiting_ticket = ticket_id;
  visit.state = VisitState::kWaitingMaterialization;
  metrics_.visits_suspended.fetch_add(1);
  if (owner) store_requests_.Push({ticket_id, visit.current_key});
  return true;
}

ExpansionPayload SearchRun::DetectTerminal(
    const PositionHistory& history) const {
  ExpansionPayload payload;
  const Position& position = history.Last();
  const auto legal = position.GetBoard().GenerateLegalMoves();
  const GameResult result = history.ComputeGameResult();
  if (result == GameResult::UNDECIDED) {
    payload.moves.assign(legal.begin(), legal.end());
    return payload;
  }
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
  }
  if (result == GameResult::DRAW) {
    payload.leaf_q = 0.0f;
    payload.leaf_d = 1.0f;
  } else {
    const bool side_to_move_won =
        (result == GameResult::WHITE_WON) != position.IsBlackToMove();
    payload.leaf_q = side_to_move_won ? 1.0f : -1.0f;
    payload.leaf_d = 0.0f;
  }
  payload.leaf_m = 0.0f;
  return payload;
}

void SearchRun::StoreWorker() {
  StoreRequest first;
  while (store_requests_.Pop(&first)) {
    std::vector<StoreRequest> requests{first};
    StoreRequest next;
    while (requests.size() < 256 && store_requests_.TryPop(&next))
      requests.push_back(next);
    std::vector<NodeKey> keys;
    keys.reserve(requests.size());
    for (const auto& request : requests) keys.push_back(request.key);
    metrics_.node_store_load_batches.fetch_add(1);
    metrics_.node_store_load_keys.fetch_add(keys.size());
    auto loaded = store_->LoadBatch(keys);
    for (size_t i = 0; i < requests.size(); ++i) {
      if (loaded[i]) {
        metrics_.node_store_hits.fetch_add(1);
        metrics_.graph_nodes_rehydrated.fetch_add(1);
        CompleteMaterialization(requests[i].ticket, std::move(*loaded[i]), false);
        continue;
      }
      metrics_.node_store_misses.fetch_add(1);
      PositionHistory history;
      {
        std::lock_guard lock(tickets_mutex_);
        auto ticket = tickets_.find(requests[i].ticket);
        if (ticket == tickets_.end()) continue;
        history = ticket->second.owner_history;
      }
      ExpansionPayload payload = DetectTerminal(history);
      if (payload.terminal != TerminalKind::kNonTerminal) {
        CompleteMaterialization(requests[i].ticket, std::move(payload), true);
      } else {
        auto request = std::make_shared<EvalRequest>(EvalRequest{
            .ticket = requests[i].ticket,
            .key = requests[i].key,
            .history = std::move(history),
            .payload = std::move(payload),
            .queued_at = std::chrono::steady_clock::now()});
        {
          std::lock_guard lock(tickets_mutex_);
          auto ticket = tickets_.find(request->ticket);
          if (ticket == tickets_.end()) continue;
          ticket->second.state = TicketState::kWaitingForEval;
        }
        metrics_.eval_requests.fetch_add(1);
        ready_evals_.Push(std::move(request));
        RaiseHighWater(metrics_.ready_eval_high_water,
                       static_cast<uint64_t>(ready_evals_.Size()));
      }
    }
  }
}

bool SearchRun::ShouldFlushEvaluation() const {
  return (ready_visits_.Empty() &&
          advancing_visits_.load(std::memory_order_relaxed) == 0) ||
         stop_mode_.load(std::memory_order_relaxed) != StopMode::kRunning;
}

void SearchRun::EvaluatorWorker() {
  std::shared_ptr<EvalRequest> first;
  while (ready_evals_.Pop(&first)) {
    auto computation = backend_->CreateComputation();
    std::vector<std::shared_ptr<EvalRequest>> pending;
    auto request = std::move(first);
    const auto collection_start = std::chrono::steady_clock::now();
    while (request) {
      request->payload.priors.resize(request->payload.moves.size());
      const auto immediate = computation->AddInput(
          EvalPosition{.pos = request->history.GetPositions(),
                       .legal_moves = request->payload.moves},
          EvalResultPtr{.q = &request->payload.leaf_q,
                        .d = &request->payload.leaf_d,
                        .m = &request->payload.leaf_m,
                        .p = request->payload.priors});
      if (immediate == BackendComputation::FETCHED_IMMEDIATELY) {
        metrics_.cache_hits.fetch_add(1);
        PublishEvaluation(std::move(request));
      } else {
        pending.push_back(std::move(request));
      }
      if (computation->UsedBatchSize() >=
          static_cast<size_t>(settings_.minibatch_size))
        break;
      request.reset();
      if (ready_evals_.TryPop(&request)) continue;
      const bool timed_out =
          settings_.max_batch_delay_ms > 0 &&
          std::chrono::steady_clock::now() - collection_start >=
              std::chrono::milliseconds(settings_.max_batch_delay_ms);
      if (ShouldFlushEvaluation() || timed_out) break;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      ready_evals_.TryPop(&request);
    }
    if (computation->UsedBatchSize() > 0) {
      evaluations_in_progress_.fetch_add(1);
      metrics_.evaluation_batches.fetch_add(1);
      metrics_.nn_evaluations.fetch_add(computation->UsedBatchSize());
      computation->ComputeBlocking();
      evaluations_in_progress_.fetch_sub(1);
    }
    for (auto& item : pending) PublishEvaluation(std::move(item));
    NotifyController();
  }
}

void SearchRun::PublishEvaluation(std::shared_ptr<EvalRequest> request) {
  std::vector<size_t> order(request->payload.moves.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    if (request->payload.priors[a] != request->payload.priors[b])
      return request->payload.priors[a] > request->payload.priors[b];
    return request->payload.moves[a].raw_data() <
           request->payload.moves[b].raw_data();
  });
  ExpansionPayload sorted = request->payload;
  for (size_t i = 0; i < order.size(); ++i) {
    sorted.moves[i] = request->payload.moves[order[i]];
    sorted.priors[i] = request->payload.priors[order[i]];
  }
  CompleteMaterialization(request->ticket, std::move(sorted), true);
}

void SearchRun::CompleteMaterialization(MaterializationTicketId ticket_id,
                                        ExpansionPayload payload,
                                        bool store_payload) {
  Ticket ticket;
  uint64_t generation;
  {
    std::lock_guard lock(tickets_mutex_);
    auto it = tickets_.find(ticket_id);
    if (it == tickets_.end()) return;
    // Keep the ticket discoverable until the expanded graph is published.
    generation = graph_->InstallPayload(it->second.key, ticket_id, payload);
    ticket = std::move(it->second);
    tickets_.erase(it);
    ticket_by_key_.erase(ticket.key);
  }
  RaiseHighWater(metrics_.graph_size_high_water,
                 static_cast<uint64_t>(graph_->Size()));
  auto wake = [&](VisitId id, bool owner) {
    VisitPool::Slot* slot = visits_.Lookup(id);
    if (!slot) return;
    std::lock_guard lock(slot->mutex);
    if (slot->epoch.load(std::memory_order_relaxed) != id.epoch() ||
        slot->visit.state != VisitState::kWaitingMaterialization ||
        slot->visit.waiting_ticket != ticket.id)
      return;
    Visit& visit = slot->visit;
    visit.path.back().generation = generation;
    visit.path.back().selected_move.reset();
    visit.waiting_ticket = 0;
    if (payload.terminal != TerminalKind::kNonTerminal || owner) {
      visit.result = {payload.leaf_q, payload.leaf_d, payload.leaf_m};
      visit.state = VisitState::kReadyBackup;
      if (payload.terminal != TerminalKind::kNonTerminal)
        metrics_.terminal_visits.fetch_add(1);
    } else {
      visit.state = VisitState::kReadySelect;
      metrics_.visits_resumed.fetch_add(1);
    }
    ready_visits_.Push(id);
  };
  wake(ticket.owner, true);
  for (VisitId waiter : ticket.waiters) wake(waiter, false);
  // Store only immutable payload, after Visits have been made runnable.
  if (store_payload) {
    StoredExpansion entry{ticket.key, payload};
    store_->StoreBatch(std::span<const StoredExpansion>(&entry, 1));
    metrics_.node_store_store_batches.fetch_add(1);
  }
  NotifyController();
}

void SearchRun::Backup(VisitPool::Slot& slot) {
  Visit& visit = slot.visit;
  visit.state = VisitState::kBackingUp;
  SearchValue value = visit.result;
  for (size_t i = visit.path.size(); i-- > 0;) {
    const PathStep& step = visit.path[i];
    UpdateResult node_result =
        graph_->UpdateNodeValue(step.key, step.generation, value);
    if (node_result == UpdateResult::kStale ||
        node_result == UpdateResult::kMissing)
      metrics_.stale_generation_node_updates.fetch_add(1);
    metrics_.backup_node_steps.fetch_add(1);
    if (i == 0) break;
    SearchValue parent_value = value.Parent();
    const PathStep& parent = visit.path[i - 1];
    if (parent.selected_move) {
      UpdateResult edge_result = graph_->CompleteEdge(
          parent.key, parent.generation, *parent.selected_move, parent_value);
      if (edge_result == UpdateResult::kUnderflow)
        metrics_.invariant_underflow_prevented.fetch_add(1);
      else if (edge_result != UpdateResult::kApplied)
        metrics_.stale_generation_edge_updates.fetch_add(1);
    }
    value = parent_value;
  }
  RaiseHighWater(metrics_.max_depth, static_cast<uint64_t>(visit.path.size()));
  FinishVisit(slot, true);
}

void SearchRun::Cancel(VisitPool::Slot& slot) {
  Visit& visit = slot.visit;
  for (const PathStep& step : visit.path) {
    if (!step.selected_move) continue;
    UpdateResult result =
        graph_->CancelEdge(step.key, step.generation, *step.selected_move);
    if (result == UpdateResult::kUnderflow)
      metrics_.invariant_underflow_prevented.fetch_add(1);
  }
  FinishVisit(slot, false);
}

void SearchRun::FinishVisit(VisitPool::Slot& slot, bool completed) {
  const VisitId id = slot.visit.id;
  if (completed) {
    const uint64_t count = metrics_.visits_completed.fetch_add(1) + 1;
    if (node_limit_ && count >= *node_limit_)
      RequestStop(StopMode::kRespondBestmove);
  } else {
    metrics_.visits_cancelled.fetch_add(1);
  }
  visits_.ReleaseLocked(slot, id);
  NotifyController();
}

void SearchRun::CancelAllVisits() {
  for (VisitId id : visits_.ActiveIds()) {
    VisitPool::Slot* slot = visits_.Lookup(id);
    if (!slot) continue;
    bool enqueue = false;
    {
      std::lock_guard lock(slot->mutex);
      if (slot->epoch.load(std::memory_order_relaxed) == id.epoch() &&
          slot->visit.state != VisitState::kFree &&
          slot->visit.state != VisitState::kCancelling) {
        slot->visit.state = VisitState::kCancelling;
        enqueue = true;
      }
    }
    if (enqueue) ready_visits_.Push(id);
  }
}

void SearchRun::NotifyController() { controller_cv_.notify_all(); }

void SearchRun::Controller() {
  auto next_info = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  for (;;) {
    const auto now = std::chrono::steady_clock::now();
    if (stop_mode_.load() == StopMode::kRunning && deadline_ && now >= *deadline_)
      RequestStop(StopMode::kRespondBestmove);
    if (stop_mode_.load() == StopMode::kRunning) AdmitMore();
    if (stop_mode_.load() == StopMode::kRunning && now >= next_info) {
      OutputInfo(false);
      next_info = now + std::chrono::seconds(1);
    }
    if (stop_mode_.load() != StopMode::kRunning) CancelAllVisits();
    bool no_tickets;
    {
      std::lock_guard lock(tickets_mutex_);
      no_tickets = tickets_.empty();
    }
    if (stop_mode_.load() != StopMode::kRunning && visits_.active() == 0 &&
        no_tickets && store_requests_.Empty() && ready_evals_.Empty() &&
        evaluations_in_progress_.load() == 0) {
      break;
    }
    std::unique_lock lock(controller_mutex_);
    controller_cv_.wait_for(lock, std::chrono::milliseconds(10));
  }
  ready_visits_.Close();
  store_requests_.Close();
  ready_evals_.Close();
  const StopMode final_mode = stop_mode_.load();
  if (final_mode == StopMode::kRespondBestmove &&
      !output_committed_.exchange(true)) {
    OutputInfo(true);
    std::vector<Move> pv = BuildPv();
    Move best = pv.empty() ? FallbackMove() : pv.front();
    Move ponder = pv.size() > 1 ? pv[1] : Move{};
    BestMoveInfo info(best, ponder);
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
  Move best = *std::min_element(legal.begin(), legal.end(),
                               [](Move a, Move b) {
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
  info.comment = metrics_.Format(graph_->Size(), visits_.active(),
                                 ready_evals_.Size());
  if (final) info.comment += " final";
  std::vector<ThinkingInfo> infos{std::move(info)};
  responder_->OutputThinkingInfo(&infos);
}

}  // namespace lczero::lc5
