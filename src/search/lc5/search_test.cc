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

#include <gtest/gtest.h>

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "search/lc5/engine.h"
#include "search/lc5/graph/node_record.h"
#include "search/lc5/policies/policy_engine.h"
#include "search/lc5/policies/puct.h"
#include "search/lc5/policies/puct_dag.h"
#include "search/lc5/policies/registry.h"
#include "search/lc5/session/search_session.h"
#include "search/lc5/store/memory_node_store.h"

namespace lczero::lc5 {

class SearchSessionTestPeer {
 public:
  // Admits a visit from an arbitrary origin, sharing the node budget with
  // root visits. May be called before Start(). Returns false if admission is
  // closed, the budget is exhausted or the pool is full.
  static bool AdmitVisit(SearchSession& session, const VisitOrigin& origin) {
    size_t owner;
    std::vector<VisitWakeup> admitted;
    {
      absl::MutexLock lock(session.admission_mutex_);
      owner = session.next_owner_++ % session.workers_.size();
      const auto id = session.AllocateVisit(origin);
      if (!id) return false;
      admitted.push_back({.id = *id, .worker = owner});
    }
    // The allocated visit keeps active() nonzero until this is consumed, so
    // stop cannot declare a drain before the inbox publication.
    session.SendWakeups(owner, admitted);
    return true;
  }
  // Before Start only: make the entire backlog ready before collectors session.
  static void PrepareAdmissions(SearchSession& session)
      ABSL_NO_THREAD_SAFETY_ANALYSIS {
    for (auto& ptr : session.workers_) {
      auto& worker = *ptr;
      worker.outgoing.resize(session.workers_.size());
      for (const auto& wakeup : worker.inbox.wakeups)
        session.ResumeVisit(worker, wakeup);
      worker.inbox.wakeups.clear();
      while (!worker.runnable.empty()) {
        auto id = worker.runnable.front();
        worker.runnable.pop_front();
        session.AdvanceVisit(worker, id);
      }
      session.SubmitJobs(worker);
      session.SetWorkerProducing(worker, false);
    }
  }
  // Inspect owner-local state only after Wait's full join barrier.
  static void ExpectDrained(const SearchSession& session)
      ABSL_NO_THREAD_SAFETY_ANALYSIS {
    EXPECT_EQ(session.visits_.active(), 0u);
    EXPECT_TRUE(session.tickets_.empty());
    EXPECT_TRUE(session.ticket_by_key_.empty());
    EXPECT_EQ(session.jobs_in_flight_.load(), 0u);
    EXPECT_EQ(session.collecting_jobs_.load(), 0u);
    EXPECT_EQ(session.eval_jobs_.Size(), 0u);
    EXPECT_EQ(session.store_jobs_.Size(), 0u);
    for (const auto& worker : session.workers_) {
      EXPECT_TRUE(worker->drained.load());
      EXPECT_TRUE(worker->active.empty());
      EXPECT_TRUE(worker->runnable.empty());
      EXPECT_TRUE(worker->unprepared.empty());
      EXPECT_TRUE(worker->loads.empty());
      EXPECT_TRUE(worker->evals.empty());
      EXPECT_EQ(worker->jobs_in_flight, 0u);
      EXPECT_EQ(worker->items_in_flight, 0u);
      EXPECT_TRUE(worker->inbox.wakeups.empty());
      EXPECT_TRUE(worker->inbox.completions.empty());
    }
  }
  static void ExpectDrained(const Lc5Engine& engine) {
    ASSERT_TRUE(engine.session_);
    ASSERT_TRUE(engine.session_->IsFinished());
    ExpectDrained(*engine.session_);
    ASSERT_TRUE(engine.visit_engine_);
    for (const auto& [key, node] :
         static_cast<PolicyEngine<PuctAveraging>&>(*engine.visit_engine_)
             .graph()
             .SnapshotAllForTesting()) {
      EXPECT_EQ(node.lifecycle, NodeLifecycle::kExpanded);
      EXPECT_EQ(node.ticket, 0u);
      for (const auto& tried : node.tried) EXPECT_EQ(tried.in_flight, 0u);
    }
  }
};

namespace {

// The statistics of edge `i`; an untried edge has none, so it reads as empty.
const TriedEdge<PuctAveraging>& TriedAt(const NodeState<PuctAveraging>& node,
                                        size_t i) {
  static const TriedEdge<PuctAveraging> kUntried;
  return i < node.tried.size() ? node.tried[i] : kUntried;
}

Settings::Resolved TestSettings(int threads = 1, int capacity = 4) {
  return {.threads = threads,
          .eval_threads = 1,
          .minibatch_size = 4,
          .max_active_visits = capacity,
          .max_batch_delay_ms = 0,
          .active_visits_ratio = 0.0f};
}

PuctAveraging::Params TestParams() {
  return {.cpuct = 1.0f,
          .cpuct_base = 100.0f,
          .cpuct_factor = 0.0f,
          .fpu_strategy = FpuStrategy::kAbsolute,
          .fpu_value = 0.0f};
}

TimeManager::Config TestTimeManagement() {
  return {.move_overhead_ms = 200, .alphazero_time_pct = 3.0f};
}

VisitOrigin Origin(std::string_view fen = ChessBoard::kStartposFen) {
  PositionHistory history;
  history.Reset(Position::FromFen(fen));
  return {.key = MakeNodeKey(history, 7),
          .history = std::move(history),
          .backup_prefix = {}};
}

// Only read after Wait(), except for the atomic bestmove count while blocked.
class RecordingResponder final : public UciResponder {
 public:
  void OutputBestMove(BestMoveInfo* info) override {
    {
      std::lock_guard lock(mutex_);
      bestmoves.push_back(*info);
      bestmove_count.fetch_add(1);
    }
    cv_.notify_all();
  }
  bool WaitForBestmove() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, std::chrono::seconds(5),
                        [&] { return bestmove_count.load() != 0; });
  }
  void OutputThinkingInfo(std::vector<ThinkingInfo>* infos) override {
    thinking.insert(thinking.end(), infos->begin(), infos->end());
  }
  std::atomic<size_t> bestmove_count{0};
  std::vector<BestMoveInfo> bestmoves;
  std::vector<ThinkingInfo> thinking;

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
};

// One-shot barrier: arrival is observable before return; release is untimed.
class OperationBarrier {
 public:
  void ArriveAndWait() {
    std::unique_lock lock(mutex_);
    arrived_ = true;
    cv_.notify_all();
    cv_.wait(lock, [&] { return released_; });
  }
  bool WaitForArrival() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, std::chrono::seconds(5),
                        [&] { return arrived_; });
  }
  void Release() {
    {
      std::lock_guard lock(mutex_);
      released_ = true;
    }
    cv_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool arrived_ = false;
  bool released_ = false;
};

class ControlledBackend final : public Backend {
 public:
  enum class Cache { kNone, kAll, kAlternating };
  explicit ControlledBackend(Cache cache = Cache::kNone, bool blocked = false,
                             int maximum_batch_size = 4,
                             size_t evaluator_count = 1)
      : cache_(cache),
        maximum_batch_size_(maximum_batch_size),
        evaluator_count_(evaluator_count),
        released_(!blocked) {}

  BackendAttributes GetAttributes() const override {
    return {.has_mlh = true,
            .has_wdl = true,
            .runs_on_cpu = true,
            .suggested_num_search_threads = 1,
            .recommended_batch_size = 4,
            .maximum_batch_size = maximum_batch_size_};
  }
  std::unique_ptr<BackendComputation> CreateComputation() override {
    {
      // Each evaluator has already popped a job. Hold the initial cohort here
      // so one fast evaluator cannot consume all work before its peers start.
      std::unique_lock lock(mutex_);
      evaluator_ids.insert(std::this_thread::get_id());
      cv_.notify_all();
      cv_.wait(lock, [&] {
        return evaluator_ids.size() >= evaluator_count_ || released_;
      });
    }
    return std::make_unique<Computation>(*this);
  }
  bool WaitForInputs(size_t count) {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, std::chrono::seconds(5),
                        [&] { return history_sizes.size() >= count; });
  }
  bool WaitForCompute(size_t count = 1) {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, std::chrono::seconds(5),
                        [&] { return batch_sizes.size() >= count; });
  }
  void Release() {
    {
      std::lock_guard lock(mutex_);
      released_ = true;
    }
    input.Release();
    cv_.notify_all();
  }

  // Configure before Start. Pause the numbered AddInput before returning.
  size_t blocked_input = 0;
  OperationBarrier input;
  float q = 0.25f;
  std::atomic<size_t> inputs{0};
  std::atomic<size_t> computes{0};
  // Protected by mutex_ while writing; tests read after the session joins.
  std::vector<size_t> history_sizes;
  std::vector<size_t> batch_sizes;
  std::unordered_map<std::thread::id, size_t> calls_by_evaluator;
  std::unordered_set<std::thread::id> evaluator_ids;

 private:
  void Fill(EvalResultPtr result) {
    *result.q = q;
    *result.d = 0.5f;
    *result.m = 3.0f;
    std::fill(result.p.begin(), result.p.end(), 1.0f / result.p.size());
  }
  class Computation final : public BackendComputation {
   public:
    explicit Computation(ControlledBackend& backend) : backend_(backend) {}
    size_t UsedBatchSize() const override { return pending_.size(); }
    AddInputResult AddInput(const EvalPosition& pos,
                            EvalResultPtr result) override {
      const size_t index = backend_.inputs.fetch_add(1);
      {
        std::lock_guard lock(backend_.mutex_);
        backend_.history_sizes.push_back(pos.pos.size());
        backend_.cv_.notify_all();
      }
      if (index + 1 == backend_.blocked_input) backend_.input.ArriveAndWait();
      if (backend_.cache_ == Cache::kAll ||
          (backend_.cache_ == Cache::kAlternating && index % 2 == 0)) {
        backend_.Fill(result);
        return FETCHED_IMMEDIATELY;
      }
      // Keep the input spans too: read them at compute time to exercise job
      // storage lifetime, not just the output pointers.
      positions_.push_back(pos);
      pending_.push_back(result);
      return ENQUEUED_FOR_EVAL;
    }
    void ComputeBlocking() override {
      {
        std::unique_lock lock(backend_.mutex_);
        backend_.batch_sizes.push_back(pending_.size());
        ++backend_.calls_by_evaluator[std::this_thread::get_id()];
        backend_.cv_.notify_all();
        backend_.cv_.wait(lock, [&] { return backend_.released_; });
      }
      backend_.computes.fetch_add(1);
      for (size_t i = 0; i < pending_.size(); ++i) {
        EXPECT_FALSE(positions_[i].pos.empty());
        EXPECT_EQ(positions_[i].legal_moves.size(), pending_[i].p.size());
        backend_.Fill(pending_[i]);
      }
    }

   private:
    ControlledBackend& backend_;
    std::vector<EvalPosition> positions_;
    std::vector<EvalResultPtr> pending_;
  };
  const Cache cache_;
  const int maximum_batch_size_;
  const size_t evaluator_count_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool released_;
};

class RecordingStore final : public NodeStore {
 public:
  std::vector<std::optional<std::vector<uint8_t>>> LoadBatch(
      std::span<const NodeKey> keys) override {
    std::lock_guard lock(mutex_);
    std::vector<std::optional<std::vector<uint8_t>>> result;
    for (auto key : keys) {
      auto it = entries.find(key);
      result.push_back(it == entries.end() ? std::nullopt
                                           : std::optional(it->second));
    }
    return result;
  }
  void StoreBatch(std::span<const StoredNode> batch) override {
    std::lock_guard lock(mutex_);
    for (const auto& node : batch) {
      entries.insert_or_assign(node.key, node.record);
      ++stored;
    }
  }
  void EraseBatch(std::span<const NodeKey> keys) override {
    std::lock_guard lock(mutex_);
    for (const auto key : keys) entries.erase(key);
  }
  void Clear() override {
    std::lock_guard lock(mutex_);
    entries.clear();
  }
  // Seed before Start(), inspect after Wait().
  absl::flat_hash_map<NodeKey, std::vector<uint8_t>> entries;
  size_t stored = 0;

 private:
  std::mutex mutex_;
};

class BlockingStore final : public NodeStore {
 public:
  explicit BlockingStore(size_t blocked_load = 1, bool block_persistence = true)
      : blocked_load_(blocked_load), block_persistence_(block_persistence) {}
  std::vector<std::optional<std::vector<uint8_t>>> LoadBatch(
      std::span<const NodeKey> keys) override {
    if (load_calls_.fetch_add(1) + 1 == blocked_load_) load.ArriveAndWait();
    return recording.LoadBatch(keys);
  }
  void StoreBatch(std::span<const StoredNode> nodes) override {
    if (block_persistence_) persist.ArriveAndWait();
    recording.StoreBatch(nodes);
  }
  void EraseBatch(std::span<const NodeKey> keys) override {
    recording.EraseBatch(keys);
  }
  void Clear() override { recording.Clear(); }
  void Release() {
    load.Release();
    persist.Release();
  }
  OperationBarrier load;
  OperationBarrier persist;
  RecordingStore recording;

 private:
  const size_t blocked_load_;
  const bool block_persistence_;
  std::atomic<size_t> load_calls_{0};
};

// Records every expanded node of `graph` in `store`.
template <typename P>
void SeedStore(const GameGraph<P>& graph, RecordingStore& store) {
  for (const auto& [key, node] : graph.SnapshotAllForTesting()) {
    if (node.lifecycle == NodeLifecycle::kExpanded) {
      store.entries.insert_or_assign(key, EncodeNode(node));
    }
  }
}

// Gives `engine` a new store and returns it.
template <typename S, typename... Args>
S& InstallStore(VisitEngine& engine, Args&&... args) {
  auto store = std::make_unique<S>(std::forward<Args>(args)...);
  S& result = *store;
  engine.SetStore(std::move(store));
  return result;
}

template <typename P>
void ExpectDrained(const SearchSession& session, const GameGraph<P>& graph) {
  ASSERT_TRUE(session.IsFinished());
  SearchSessionTestPeer::ExpectDrained(session);
  const auto& m = session.metrics();
  EXPECT_EQ(m.visits_admitted.load(),
            m.visits_completed.load() + m.visits_cancelled.load());
  EXPECT_EQ(m.invariant_underflow_prevented.load(), 0u);
  EXPECT_EQ(m.publications_rejected.load(), 0u);
  for (const auto& [key, node] : graph.SnapshotAllForTesting()) {
    SCOPED_TRACE(key.hash);
    EXPECT_EQ(node.lifecycle, NodeLifecycle::kExpanded);
    EXPECT_EQ(node.ticket, 0u);
    for (const auto& tried : node.tried) EXPECT_EQ(tried.in_flight, 0u);
  }
}

void ExpectFinalOutput(const RecordingResponder& responder, uint64_t nodes) {
  ASSERT_EQ(responder.bestmoves.size(), 1u);
  ASSERT_FALSE(responder.thinking.empty());
  const auto& final = responder.thinking.back();
  EXPECT_LE(final.nodes, nodes);
  EXPECT_NE(final.comment.find(" active="), std::string::npos);
  EXPECT_NE(final.comment.find(" ready_eval="), std::string::npos);
  EXPECT_TRUE(final.comment.ends_with(" final"));
}

void ExpectLegalBestmove(const RecordingResponder& responder,
                         const VisitOrigin& root) {
  ASSERT_EQ(responder.bestmoves.size(), 1u);
  auto move = responder.bestmoves[0].bestmove;
  if (root.history.Last().IsBlackToMove()) move.Flip();
  const auto legal = root.history.Last().GetBoard().GenerateLegalMoves();
  EXPECT_NE(std::find(legal.begin(), legal.end(), move), legal.end());
}

TEST(Lc5SearchTest, ExpiredTimeBudgetStopsAndDrains) {
  for (const auto& params :
       {GoParams{.movetime = 0}, GoParams{.movetime = -1},
        GoParams{.wtime = 0, .btime = 0}, GoParams{.movetime = 60000}}) {
    PolicyEngine<PuctAveraging> engine(TestParams());
    auto& graph = engine.graph();
    auto& store = InstallStore<RecordingStore>(engine);
    ControlledBackend backend;
    RecordingResponder responder;
    const auto root = Origin();
    // A synthetic old origin also verifies that preparation time is charged.
    SearchSession session(
        &engine, &backend, &responder, TestSettings(), TestTimeManagement(),
        root, params,
        std::chrono::steady_clock::now() - std::chrono::minutes(2));
    ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, root));
    session.Start();
    session.Wait();
    // Startup can race expiration; neither completed visits nor evaluations
    // are required to be zero. All admitted work must still drain.
    ExpectDrained(session, graph);
    ExpectFinalOutput(responder, session.metrics().visits_completed.load());
    ExpectLegalBestmove(responder, root);
    // Nothing is evicted without a graph node limit.
    EXPECT_EQ(store.stored, 0u);
  }
}

TEST(Lc5SearchTest, AbortSuppressesOutputWithExpiredTimeBudget) {
  PolicyEngine<PuctAveraging> engine(TestParams());
  auto& graph = engine.graph();
  InstallStore<RecordingStore>(engine);
  ControlledBackend backend;
  RecordingResponder responder;
  const auto root = Origin();
  SearchSession session(
      &engine, &backend, &responder, TestSettings(), TestTimeManagement(), root,
      GoParams{.movetime = 0},
      std::chrono::steady_clock::now() - std::chrono::minutes(2));
  ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, root));
  // Abort before startup so output commitment cannot race the assertion.
  session.Abort();
  session.Start();
  session.Wait();
  ExpectDrained(session, graph);
  EXPECT_TRUE(responder.bestmoves.empty());
  EXPECT_TRUE(responder.thinking.empty());
}

TEST(Lc5SearchTest, ExplicitMovetimeAndInfiniteOverrideExhaustedClock) {
  for (const auto& params :
       {GoParams{.wtime = 0, .btime = 0, .nodes = 1, .movetime = 60000},
        GoParams{.wtime = 0, .btime = 0, .nodes = 1, .infinite = true},
        GoParams{.wtime = 0,
                 .btime = 0,
                 .nodes = 1,
                 .movetime = 60000,
                 .infinite = true}}) {
    for (const auto* fen :
         {ChessBoard::kStartposFen,
          "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"}) {
      SCOPED_TRACE(fen);
      SCOPED_TRACE(params.infinite);
      SCOPED_TRACE(params.movetime.value_or(-1));
      PolicyEngine<PuctAveraging> engine(TestParams());
      auto& graph = engine.graph();
      ControlledBackend backend;
      RecordingResponder responder;
      const auto root = Origin(fen);
      SearchSession session(&engine, &backend, &responder, TestSettings(),
                            TestTimeManagement(), root, params,
                            std::chrono::steady_clock::now());
      session.Start();
      session.Wait();
      ExpectDrained(session, graph);
      EXPECT_EQ(session.metrics().visits_admitted.load(), 1u);
      EXPECT_EQ(session.metrics().visits_completed.load(), 1u);
      EXPECT_EQ(session.metrics().visits_cancelled.load(), 0u);
      ExpectFinalOutput(responder, 1);
      ExpectLegalBestmove(responder, root);
    }
  }
}

TEST(Lc5SearchTest, PonderIgnoresZeroNodeAndTimeLimits) {
  PolicyEngine<PuctAveraging> engine(TestParams());
  auto& graph = engine.graph();
  ControlledBackend backend(ControlledBackend::Cache::kNone, true);
  RecordingResponder responder;
  const auto root = Origin();
  SearchSession session(
      &engine, &backend, &responder, TestSettings(1, 1), TestTimeManagement(),
      root,
      GoParams{
          .wtime = 0, .btime = 0, .nodes = 0, .movetime = 0, .ponder = true},
      std::chrono::steady_clock::now() - std::chrono::minutes(2));
  session.Start();
  const bool computing = backend.WaitForCompute();
  if (computing) {
    EXPECT_FALSE(session.IsFinished());
    EXPECT_EQ(responder.bestmove_count.load(), 0u);
  } else {
    session.Abort();
  }
  // Release before fatal assertions or destruction, including failure paths.
  backend.Release();
  const bool continued = computing && backend.WaitForInputs(2);
  session.Stop();
  session.Wait();
  ASSERT_TRUE(computing) << "Ponder must admit work despite zero limits";
  ASSERT_TRUE(continued) << "Ponder must continue after the first evaluation";
  ExpectDrained(session, graph);
  EXPECT_GE(session.metrics().visits_admitted.load(), 2u);
  ExpectFinalOutput(responder, session.metrics().visits_completed.load());
  ExpectLegalBestmove(responder, root);
}

TEST(Lc5SearchTest, ExactNodeBudgetWithMoreWorkersThanCapacity) {
  for (const int64_t nodes : {0, 1, 7}) {
    SCOPED_TRACE(nodes);
    PolicyEngine<PuctAveraging> engine(TestParams());
    auto& graph = engine.graph();
    ControlledBackend backend;
    RecordingResponder responder;
    auto root = Origin();
    SearchSession session(&engine, &backend, &responder, TestSettings(8, 2),
                          TestTimeManagement(), root, GoParams{.nodes = nodes},
                          std::chrono::steady_clock::now());
    session.Start();
    session.Wait();
    ExpectDrained(session, graph);
    EXPECT_EQ(session.metrics().visits_admitted.load(), nodes);
    EXPECT_EQ(session.metrics().visits_completed.load(), nodes);
    EXPECT_EQ(session.metrics().visits_cancelled.load(), 0u);
    EXPECT_LE(session.metrics().active_visits_high_water.load(), 2u);
    EXPECT_FALSE(SearchSessionTestPeer::AdmitVisit(session, root));
    ExpectFinalOutput(responder, nodes);
    if (nodes == 0) {
      EXPECT_FALSE(responder.thinking.back().score.has_value());
      EXPECT_FALSE(responder.thinking.back().wdl.has_value());
    }
    const auto legal = root.history.Last().GetBoard().GenerateLegalMoves();
    EXPECT_NE(
        std::find(legal.begin(), legal.end(), responder.bestmoves[0].bestmove),
        legal.end());
    // Each completed visit counts once at the root, including the one that
    // expanded it.
    const auto node = graph.SnapshotNode(root.key);
    EXPECT_EQ(node ? node->stats.visits : 0u, static_cast<uint64_t>(nodes));
  }
}

TEST(Lc5SearchTest, ReportsCentipawnsAndWdlFromRootSideToMove) {
  struct Case {
    float q;
    int score;
    int win;
    int loss;
  };
  for (const auto* fen :
       {ChessBoard::kStartposFen,
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"}) {
    SCOPED_TRACE(fen);
    for (const auto& test :
         {Case{0.25f, 37, 375, 125}, Case{-0.25f, -37, 125, 375},
          Case{0.0f, 0, 250, 250}}) {
      SCOPED_TRACE(test.q);
      PolicyEngine<PuctAveraging> engine(TestParams());
      auto& graph = engine.graph();
      ControlledBackend backend;
      backend.q = test.q;
      RecordingResponder responder;
      SearchSession session(
          &engine, &backend, &responder, TestSettings(), TestTimeManagement(),
          Origin(fen), GoParams{.nodes = 1}, std::chrono::steady_clock::now());
      session.Start();
      session.Wait();
      ExpectDrained(session, graph);
      ExpectFinalOutput(responder, 1);
      const auto& final = responder.thinking.back();
      ASSERT_TRUE(final.score.has_value());
      EXPECT_EQ(*final.score, test.score);
      ASSERT_TRUE(final.wdl.has_value());
      EXPECT_EQ(final.wdl->w, test.win);
      EXPECT_EQ(final.wdl->d, 500);
      EXPECT_EQ(final.wdl->l, test.loss);
    }
  }
}

TEST(Lc5SearchTest, OwnerBacksUpLeafWhileWaitersResumeSelection) {
  PolicyEngine<PuctAveraging> engine(TestParams());
  auto& graph = engine.graph();
  ControlledBackend backend;
  RecordingResponder responder;
  auto root = Origin();
  SearchSession session(&engine, &backend, &responder, TestSettings(),
                        TestTimeManagement(), root, GoParams{.nodes = 4},
                        std::chrono::steady_clock::now());
  // One worker consumes all four admissions before submitting the root job.
  for (int i = 0; i < 4; ++i)
    ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, root));
  EXPECT_FALSE(SearchSessionTestPeer::AdmitVisit(session, root));
  session.Start();
  session.Wait();
  ExpectDrained(session, graph);
  ExpectFinalOutput(responder, 4);
  EXPECT_EQ(session.metrics().visits_completed.load(), 4u);
  EXPECT_GE(session.metrics().ticket_waiters.load(), 3u);
  EXPECT_GE(session.metrics().visits_resumed.load(), 3u);
  const auto node = graph.SnapshotNode(root.key);
  ASSERT_TRUE(node);
  EXPECT_EQ(node->stats.visits, 4u);
  uint64_t edge_visits = 0;
  for (const auto& tried : node->tried) edge_visits += tried.stats.visits;
  // Only the owner backs up at the root leaf. Each waiter must traverse an
  // edge.
  EXPECT_EQ(edge_visits, 3u);
}

TEST(Lc5SearchTest, WaitersDrainAndCountOnceInEveryResumeMode) {
  for (const auto mode : {WaiterResume::kSelect, WaiterResume::kReselect,
                          WaiterResume::kRestart}) {
    for (const int threads : {1, 2}) {
      SCOPED_TRACE(static_cast<int>(mode));
      SCOPED_TRACE(threads);
      auto params = TestParams();
      params.waiter_resume = mode;
      PolicyEngine<PuctAveraging> engine(params);
      auto& graph = engine.graph();
      ControlledBackend backend;
      RecordingResponder responder;
      auto root = Origin();
      constexpr int64_t kNodes = 64;
      SearchSession session(&engine, &backend, &responder,
                            TestSettings(threads, 16), TestTimeManagement(),
                            root, GoParams{.nodes = kNodes},
                            std::chrono::steady_clock::now());
      session.Start();
      session.Wait();
      ExpectDrained(session, graph);
      ExpectFinalOutput(responder, kNodes);
      EXPECT_EQ(session.metrics().visits_completed.load(), kNodes);
      const auto node = graph.SnapshotNode(root.key);
      ASSERT_TRUE(node);
      EXPECT_EQ(node->stats.visits, static_cast<uint64_t>(kNodes));
    }
  }
}

TEST(Lc5SearchTest, PuctDagDrainsAndKeepsNodeMeans) {
  // Kings and blocked pawns: with HistoryKeyLength 0, king routes that
  // commute share a node from the fourth ply.
  auto settings = TestSettings();
  PositionHistory history;
  history.Reset(Position::FromFen("k7/p7/P7/8/8/8/8/7K w - - 0 1"));
  const VisitOrigin root{.key = MakeNodeKey(history, 0),
                         .history = std::move(history),
                         .backup_prefix = {}};
  for (const int threads : {1, 2}) {
    SCOPED_TRACE(threads);
    settings.threads = threads;
    settings.max_active_visits = 16;
    PolicyEngine<PuctDag> engine(TestParams(), /*history_key_length=*/0);
    auto& graph = engine.graph();
    ControlledBackend backend;
    RecordingResponder responder;
    constexpr int64_t kNodes = 2000;
    SearchSession session(&engine, &backend, &responder, settings,
                          TestTimeManagement(), root, GoParams{.nodes = kNodes},
                          std::chrono::steady_clock::now());
    session.Start();
    session.Wait();
    ExpectDrained(session, graph);
    ExpectFinalOutput(responder, kNodes);
    ExpectLegalBestmove(responder, root);
    const auto node = graph.SnapshotNode(root.key);
    ASSERT_TRUE(node);
    EXPECT_EQ(node->stats.visits, static_cast<uint64_t>(kNodes));
    // A node's visits are its own sample plus its edges' visits, whose cached
    // values its mean weighs.
    const auto snapshot = graph.SnapshotAllForTesting();
    std::unordered_map<uint64_t, const NodeState<PuctDag>*> nodes;
    for (const auto& [key, n] : snapshot) {
      nodes[key.hash] = &n;
      EXPECT_LE(std::abs(n.stats.Q()), 1.0f);
      if (n.terminal != TerminalKind::kNonTerminal) continue;
      uint32_t edge_visits = 0;
      for (const auto& tried : n.tried) edge_visits += tried.stats.visits;
      EXPECT_EQ(n.stats.visits, 1 + edge_visits);
    }
    // Transpositions: some node is the child of more than one visited edge.
    std::unordered_map<uint64_t, int> parents;
    std::unordered_set<uint64_t> seen{root.key.hash};
    std::vector<PositionHistory> pending{root.history};
    while (!pending.empty()) {
      const PositionHistory parent = std::move(pending.back());
      pending.pop_back();
      const auto it = nodes.find(MakeNodeKey(parent, 0).hash);
      if (it == nodes.end()) continue;
      const auto& n = *it->second;
      for (size_t i = 0; i < n.tried.size(); ++i) {
        if (n.tried[i].stats.visits == 0) continue;
        PositionHistory child = parent;
        child.Append(n.edges[i].move);
        const uint64_t key = MakeNodeKey(child, 0).hash;
        ++parents[key];
        if (seen.insert(key).second) pending.push_back(std::move(child));
      }
    }
    EXPECT_TRUE(std::any_of(parents.begin(), parents.end(),
                            [](const auto& p) { return p.second > 1; }));
  }
}

TEST(Lc5SearchTest, StickyEndgamesPlaysMateAndStopsEarly) {
  // Mate in one: d1d8.
  const auto root = Origin("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1");
  const Move mate = Move::White(Square(kFileD, kRank1), Square(kFileD, kRank8));
  for (const bool sticky : {false, true}) {
    SCOPED_TRACE(sticky);
    auto params = TestParams();
    params.sticky_endgames = sticky;
    PolicyEngine<PuctAveraging> engine(params);
    auto& graph = engine.graph();
    ControlledBackend backend;
    RecordingResponder responder;
    // Without the option the search runs to the limit, so keep it small.
    const int nodes = sticky ? 1'000'000 : 500;
    SearchSession session(&engine, &backend, &responder, TestSettings(),
                          TestTimeManagement(), root, GoParams{.nodes = nodes},
                          std::chrono::steady_clock::now());
    session.Start();
    session.Wait();
    ExpectDrained(session, graph);
    ExpectFinalOutput(responder, nodes);
    const auto& final = responder.thinking.back();
    if (sticky) {
      EXPECT_EQ(responder.bestmoves[0].bestmove, mate);
      EXPECT_EQ(final.mate, 1);
      EXPECT_FALSE(final.score);
      EXPECT_LT(session.metrics().visits_completed.load(),
                static_cast<uint64_t>(nodes));
    } else {
      EXPECT_FALSE(final.mate);
      EXPECT_TRUE(final.score);
      EXPECT_EQ(session.metrics().visits_completed.load(),
                static_cast<uint64_t>(nodes));
    }
  }
}

TEST(Lc5SearchTest, ContainedTwoFoldDrawStopsAtFirstOccurrence) {
  // Path: nodes 0..3, then a two-fold leaf repeating node 1.
  constexpr size_t kNodes = 4;
  constexpr size_t kOrigin = 1;
  const Move move = Move::White(Square(kFileA, kRank2), Square(kFileA, kRank3));
  const SearchValue draw{.q = 0.0f, .d = 1.0f, .m = 0.0f};
  for (const bool contained : {false, true}) {
    SCOPED_TRACE(contained);
    PolicyEngine<PuctAveraging> engine(TestParams());
    auto& graph = engine.graph();
    Visit visit;
    for (size_t i = 0; i < kNodes; ++i) {
      const NodeKey key{100 + i};
      const auto created = graph.FindOrCreateMaterialization(key, 1);
      graph.CompleteMaterialization(key, 1,
                                    {.leaf = {.q = 0.5f, .d = 0.0f, .m = 10.0f},
                                     .moves = {move},
                                     .priors = {1.0f}});
      ASSERT_EQ(graph.SelectAndReserve(key, TestParams()).status,
                SelectStatus::kSelected);
      visit.path.push_back({.key = key,
                            .generation = created.generation,
                            .selected_move = move});
    }
    visit.path.push_back({.key = NodeKey{200}});
    visit.result = draw;
    if (contained) visit.two_fold_origin = kOrigin;
    Metrics metrics;
    engine.BackupVisit(visit, metrics);

    // Nodes from the first occurrence down, and their edges, take the draw;
    // its M grows by one per ply up.
    for (size_t i = kOrigin; i < kNodes; ++i) {
      SCOPED_TRACE(i);
      const auto node = graph.SnapshotNode(NodeKey{100 + i});
      ASSERT_TRUE(node);
      const float m = static_cast<float>(kNodes - i);
      EXPECT_EQ(node->stats.visits, 2u);
      EXPECT_FLOAT_EQ(node->stats.Q(), 0.25f);
      EXPECT_FLOAT_EQ(node->stats.D(), 0.5f);
      EXPECT_FLOAT_EQ(node->stats.M(), (10.0f + m) / 2);
      EXPECT_EQ(TriedAt(*node, 0).in_flight, 0u);
      EXPECT_FLOAT_EQ(TriedAt(*node, 0).stats.Q(), 0.0f);
      EXPECT_FLOAT_EQ(TriedAt(*node, 0).stats.D(), 1.0f);
      EXPECT_FLOAT_EQ(TriedAt(*node, 0).stats.M(), m);
    }
    // Above it, the edge into the first occurrence takes that node's value
    // (Q 0.25, D 0.5, M 6.5) instead, or the draw if not contained.
    const auto root = graph.SnapshotNode(NodeKey{100});
    ASSERT_TRUE(root);
    const auto& edge = TriedAt(*root, 0);
    EXPECT_EQ(edge.in_flight, 0u);
    EXPECT_EQ(edge.stats.visits, 1u);
    EXPECT_FLOAT_EQ(edge.stats.Q(), contained ? -0.25f : 0.0f);
    EXPECT_FLOAT_EQ(edge.stats.D(), contained ? 0.5f : 1.0f);
    EXPECT_FLOAT_EQ(edge.stats.M(), contained ? 7.5f : 4.0f);
    EXPECT_EQ(root->stats.visits, 2u);
    EXPECT_FLOAT_EQ(root->stats.Q(), contained ? 0.125f : 0.25f);
  }
}

TEST(Lc5SearchTest, TwoFoldDrawsFromRealCyclesDrain) {
  // Kings and blocked pawns: few moves, so king shuffles repeat early.
  const auto root = Origin("k7/p7/P7/8/8/8/8/7K w - - 0 1");
  for (const auto mode : {TwoFoldDraws::kRoot, TwoFoldDraws::kContained}) {
    for (const int threads : {1, 2}) {
      SCOPED_TRACE(static_cast<int>(mode));
      SCOPED_TRACE(threads);
      auto params = TestParams();
      params.two_fold_draws = mode;
      PolicyEngine<PuctAveraging> engine(params);
      auto& graph = engine.graph();
      ControlledBackend backend;
      RecordingResponder responder;
      constexpr int64_t kNodes = 2000;
      SearchSession session(&engine, &backend, &responder,
                            TestSettings(threads, 16), TestTimeManagement(),
                            root, GoParams{.nodes = kNodes},
                            std::chrono::steady_clock::now());
      session.Start();
      session.Wait();
      ExpectDrained(session, graph);
      ExpectFinalOutput(responder, kNodes);
      EXPECT_GT(session.metrics().two_fold_visits.load(), 0u);
      const auto node = graph.SnapshotNode(root.key);
      ASSERT_TRUE(node);
      EXPECT_EQ(node->stats.visits, static_cast<uint64_t>(kNodes));
    }
  }
}

TEST(Lc5SearchTest, TerminalOwnerAndWaitersNeverEvaluate) {
  struct Case {
    const char* fen;
    TerminalKind kind;
    float q;
    float d;
  };
  for (const auto& test : {Case{"7k/6Q1/5K2/8/8/8/8/8 b - - 0 1",
                                TerminalKind::kCheckmate, -1.0f, 0.0f},
                           Case{"7k/5K2/6Q1/8/8/8/8/8 b - - 0 1",
                                TerminalKind::kStalemate, 0.0f, 1.0f},
                           Case{"8/8/8/8/8/5k2/6q1/7K w - - 0 1",
                                TerminalKind::kCheckmate, -1.0f, 0.0f},
                           Case{"8/8/8/8/8/6q1/5k2/7K w - - 0 1",
                                TerminalKind::kStalemate, 0.0f, 1.0f}}) {
    SCOPED_TRACE(test.fen);
    PolicyEngine<PuctAveraging> engine(TestParams());
    auto& graph = engine.graph();
    ControlledBackend backend;
    RecordingResponder responder;
    const auto root = Origin(test.fen);
    SearchSession session(&engine, &backend, &responder, TestSettings(),
                          TestTimeManagement(), root, GoParams{.nodes = 4},
                          std::chrono::steady_clock::now());
    for (int i = 0; i < 4; ++i)
      ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, root));
    session.Start();
    session.Wait();
    ExpectDrained(session, graph);
    ExpectFinalOutput(responder, 4);
    EXPECT_EQ(responder.bestmoves[0].bestmove, Move{});
    EXPECT_EQ(backend.inputs.load(), 0u);
    EXPECT_EQ(backend.computes.load(), 0u);
    EXPECT_EQ(session.metrics().tickets_created.load(), 1u);
    EXPECT_EQ(session.metrics().ticket_waiters.load(), 3u);
    EXPECT_EQ(session.metrics().terminal_visits.load(), 4u);
    EXPECT_EQ(session.metrics().visits_resumed.load(), 0u);
    auto node = graph.SnapshotNode(root.key);
    ASSERT_TRUE(node);
    EXPECT_EQ(node->terminal, test.kind);
    EXPECT_EQ(node->stats.visits, 4u);
    EXPECT_FLOAT_EQ(node->stats.Q(), test.q);
    EXPECT_FLOAT_EQ(node->stats.D(), test.d);
    const auto& final = responder.thinking.back();
    ASSERT_TRUE(final.score.has_value());
    EXPECT_EQ(*final.score, test.kind == TerminalKind::kCheckmate ? -12780 : 0);
    ASSERT_TRUE(final.wdl.has_value());
    EXPECT_EQ(final.wdl->w, 0);
    EXPECT_EQ(final.wdl->d, test.kind == TerminalKind::kStalemate ? 1000 : 0);
    EXPECT_EQ(final.wdl->l, test.kind == TerminalKind::kCheckmate ? 1000 : 0);
  }
}

TEST(Lc5SearchTest, ImmediateAndMixedResultsKeepJobStorageAlive) {
  for (const auto cache : {ControlledBackend::Cache::kAll,
                           ControlledBackend::Cache::kAlternating}) {
    SCOPED_TRACE(static_cast<int>(cache));
    PolicyEngine<PuctAveraging> engine(TestParams());
    auto& graph = engine.graph();
    ControlledBackend backend(cache);
    RecordingResponder responder;
    const auto root = Origin();
    SearchSession session(&engine, &backend, &responder, TestSettings(),
                          TestTimeManagement(), root, GoParams{.nodes = 4},
                          std::chrono::steady_clock::now());
    std::vector<VisitOrigin> origins;
    const auto moves = root.history.Last().GetBoard().GenerateLegalMoves();
    for (int i = 0; i < 4; ++i) {
      auto origin = root;
      origin.history.Append(moves[i]);
      origin.key = MakeNodeKey(origin.history, 7);
      ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, origin));
      origins.push_back(std::move(origin));
    }
    EXPECT_FALSE(SearchSessionTestPeer::AdmitVisit(session, root));
    session.Start();
    session.Wait();
    ExpectDrained(session, graph);
    ExpectFinalOutput(responder, 4);
    const bool all = cache == ControlledBackend::Cache::kAll;
    EXPECT_EQ(backend.inputs.load(), 4u);
    EXPECT_EQ(backend.computes.load(), all ? 0u : 1u);
    EXPECT_EQ(session.metrics().cache_hits.load(), all ? 4u : 2u);
    EXPECT_EQ(session.metrics().nn_evaluations.load(), all ? 0u : 2u);
    for (const auto& origin : origins) {
      const auto node = graph.SnapshotNode(origin.key);
      ASSERT_TRUE(node);
      EXPECT_EQ(node->stats.visits, 1u);
      EXPECT_FLOAT_EQ(node->stats.Q(), 0.25f);
      EXPECT_FLOAT_EQ(node->stats.D(), 0.5f);
      EXPECT_FLOAT_EQ(node->stats.M(), 3.0f);
      for (const auto& edge : node->edges)
        EXPECT_EQ(edge.prior, PackedPrior(1.0f / node->edges.size()));
    }
  }
}

TEST(Lc5SearchTest, BlockedBackendStopAndAbortDrainReservations) {
  for (const bool abort : {false, true}) {
    SCOPED_TRACE(abort);
    PolicyEngine<PuctAveraging> engine(TestParams());
    auto& graph = engine.graph();
    auto& store = InstallStore<RecordingStore>(engine);
    ControlledBackend backend(ControlledBackend::Cache::kNone, true);
    RecordingResponder responder;
    const auto root = Origin();
    const auto moves = root.history.Last().GetBoard().GenerateLegalMoves();
    // All visits reserve the same root edge and share the child ticket.
    graph.CompleteMaterialization(root.key, 1,
                                  {.moves = {moves[0]}, .priors = {1.0f}});
    auto child = root;
    child.history.Append(moves[0]);
    child.key = MakeNodeKey(child.history, 7);
    SearchSession session(&engine, &backend, &responder, TestSettings(),
                          TestTimeManagement(), root,
                          GoParams{.nodes = 4, .infinite = true},
                          std::chrono::steady_clock::now());
    for (int i = 0; i < 4; ++i)
      ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, root));
    session.Start();
    const bool entered = backend.WaitForCompute();
    // Always unblock before any fatal assertion or SearchSession destruction.
    if (!entered) {
      session.Abort();
      backend.Release();
      session.Wait();
      FAIL() << "Backend computation was not reached";
    }
    EXPECT_EQ(TriedAt(*graph.SnapshotNode(root.key), 0).in_flight, 4u);
    EXPECT_EQ(session.metrics().ticket_waiters.load(), 3u);
    if (abort)
      session.Abort();
    else {
      session.Stop();
      EXPECT_TRUE(responder.WaitForBestmove());
      session.Abort();  // A committed response cannot be retracted.
      session.Stop();
    }
    EXPECT_FALSE(SearchSessionTestPeer::AdmitVisit(session, root));
    EXPECT_FALSE(session.IsFinished());
    EXPECT_EQ(responder.bestmove_count.load(), abort ? 0u : 1u);
    backend.Release();
    session.Wait();
    ExpectDrained(session, graph);
    // The evaluation was authorized before stop, so the owner backs up when
    // it returns. The waiters would select below the child and cancel.
    EXPECT_EQ(session.metrics().visits_admitted.load(), 4u);
    EXPECT_EQ(session.metrics().visits_completed.load(), 1u);
    EXPECT_EQ(session.metrics().visits_cancelled.load(), 3u);
    EXPECT_EQ(session.metrics().node_store_misses.load(), 1u);
    // Evaluations are not stored; only evicted nodes are.
    EXPECT_EQ(store.stored, 0u);
    // The setup install plus the owner's late backup.
    const auto root_node = graph.SnapshotNode(root.key);
    EXPECT_EQ(root_node->stats.visits, 2u);
    EXPECT_EQ(TriedAt(*root_node, 0).stats.visits, 1u);
    EXPECT_EQ(TriedAt(*root_node, 0).in_flight, 0u);
    if (abort) {
      EXPECT_TRUE(responder.bestmoves.empty());
      EXPECT_TRUE(std::none_of(responder.thinking.begin(),
                               responder.thinking.end(),
                               [](const ThinkingInfo& info) {
                                 return info.comment.ends_with(" final");
                               }));
    } else {
      ExpectFinalOutput(responder, 0);
    }
  }
}

TEST(Lc5SearchTest, SavedGraphRestoresWithStatisticsInFreshGraph) {
  ControlledBackend backend;
  const auto root = Origin();
  PolicyEngine<PuctAveraging> saved_engine(TestParams());
  {
    auto& saved_store = InstallStore<RecordingStore>(saved_engine);
    RecordingResponder responder;
    SearchSession session(&saved_engine, &backend, &responder, TestSettings(),
                          TestTimeManagement(), root, GoParams{.nodes = 3},
                          std::chrono::steady_clock::now());
    session.Start();
    session.Wait();
    ExpectDrained(session, saved_engine.graph());
    EXPECT_EQ(saved_store.stored, 0u);
  }
  // A fresh graph restores the root with its statistics instead of
  // evaluating it, and the new visit goes below it.
  PolicyEngine<PuctAveraging> engine(TestParams());
  auto& graph = engine.graph();
  auto& store = InstallStore<RecordingStore>(engine);
  SeedStore(saved_engine.graph(), store);
  EXPECT_EQ(store.entries.size(), saved_engine.graph().Size());
  const auto saved = *saved_engine.graph().SnapshotNode(root.key);
  ASSERT_EQ(saved.stats.visits, 3u);
  const size_t evaluated = backend.inputs.load();

  RecordingResponder responder;
  SearchSession session(&engine, &backend, &responder, TestSettings(),
                        TestTimeManagement(), root, GoParams{.nodes = 1},
                        std::chrono::steady_clock::now());
  session.Start();
  session.Wait();
  ExpectDrained(session, graph);
  ExpectFinalOutput(responder, 1);
  EXPECT_GE(session.metrics().node_store_hits.load(), 1u);
  EXPECT_GE(session.metrics().nodes_restored.load(), 1u);
  EXPECT_EQ(session.metrics().restores_rejected.load(), 0u);
  const auto node = graph.SnapshotNode(root.key);
  ASSERT_TRUE(node);
  EXPECT_EQ(node->stats.visits, saved.stats.visits + 1);
  ASSERT_EQ(node->edges.size(), saved.edges.size());
  for (size_t i = 0; i < node->edges.size(); ++i) {
    EXPECT_EQ(node->edges[i].move, saved.edges[i].move);
    EXPECT_EQ(node->edges[i].prior, saved.edges[i].prior);
  }
  uint64_t edge_visits = 0;
  for (const auto& tried : node->tried) edge_visits += tried.stats.visits;
  EXPECT_EQ(edge_visits, node->stats.visits - 1);
  // Saved nodes are restored too; the visit ends at the first node that was
  // never expanded, which is the only new evaluation.
  EXPECT_EQ(backend.inputs.load(), evaluated + 1);
}

TEST(Lc5SearchTest, RecordFromAnotherPolicyIsEvaluatedInstead) {
  PolicyEngine<PuctDag> engine(TestParams());
  auto& store = InstallStore<RecordingStore>(engine);
  ControlledBackend backend;
  const auto root = Origin();
  {
    PolicyEngine<PuctAveraging> other(TestParams());
    other.graph().CompleteMaterialization(
        root.key, 1, {.leaf = {0.5f, 0.0f, 0.0f}, .moves = {}, .priors = {}});
    SeedStore(other.graph(), store);
  }
  ASSERT_EQ(store.entries.size(), 1u);
  RecordingResponder responder;
  SearchSession session(&engine, &backend, &responder, TestSettings(),
                        TestTimeManagement(), root, GoParams{.nodes = 1},
                        std::chrono::steady_clock::now());
  session.Start();
  session.Wait();
  ExpectDrained(session, engine.graph());
  EXPECT_EQ(session.metrics().node_store_hits.load(), 1u);
  EXPECT_EQ(session.metrics().restores_rejected.load(), 1u);
  EXPECT_EQ(session.metrics().nodes_restored.load(), 0u);
  EXPECT_EQ(backend.inputs.load(), 1u);
  const auto node = engine.graph().SnapshotNode(root.key);
  ASSERT_TRUE(node);
  EXPECT_FALSE(node->edges.empty());
}

TEST(Lc5SearchTest, RootBootstrapFlushesDependencyWithoutWaitingForTimer) {
  // A positive one-minute delay cannot be the reason this bootstrap flushes.
  for (const int delay : {0, 60000}) {
    SCOPED_TRACE(delay);
    PolicyEngine<PuctAveraging> engine(TestParams());
    auto& graph = engine.graph();
    ControlledBackend backend(ControlledBackend::Cache::kNone, true);
    RecordingResponder responder;
    const auto root = Origin();
    auto settings = TestSettings(4);
    settings.max_batch_delay_ms = delay;
    SearchSession session(&engine, &backend, &responder, settings,
                          TestTimeManagement(), root, GoParams{.nodes = 1},
                          std::chrono::steady_clock::now());
    session.Start();
    const bool entered = backend.WaitForCompute();
    if (entered) {
      // Assert before stop: stop/drain must not be what unblocks collection.
      EXPECT_FALSE(session.IsFinished());
      EXPECT_EQ(session.metrics().nn_evaluations.load(), 1u);
      EXPECT_EQ(session.metrics().partial_starvation_flushes.load(), 1u);
      EXPECT_EQ(session.metrics().partial_timeout_flushes.load(), 0u);
      EXPECT_EQ(responder.bestmove_count.load(), 0u);
    } else {
      session.Abort();
    }
    // Unblock before fatal assertions, including the failure path.
    backend.Release();
    session.Wait();
    ASSERT_TRUE(entered) << "Root bootstrap depended on the collection timer";
    ExpectDrained(session, graph);
    ExpectFinalOutput(responder, 1);
    EXPECT_EQ(backend.batch_sizes, (std::vector<size_t>{1}));
  }
}

TEST(Lc5SearchTest, CollectionFlushesWhileAnotherWorkerIsBlockedInStore) {
  for (const int delay : {0, 1}) {
    SCOPED_TRACE(delay);
    PolicyEngine<PuctAveraging> engine(TestParams());
    auto& graph = engine.graph();
    // First load misses; the other worker's load remains executing. With a
    // positive delay it prevents dependency-starvation detection, so the
    // collector must actually time out. Zero must flush without that timer.
    auto& store = InstallStore<BlockingStore>(engine, 2, false);
    ControlledBackend backend(ControlledBackend::Cache::kNone, true);
    RecordingResponder responder;
    const auto root = Origin();
    auto other = root;
    other.history.Append(
        root.history.Last().GetBoard().GenerateLegalMoves()[0]);
    other.key = MakeNodeKey(other.history, 7);
    auto settings = TestSettings(2, 2);
    settings.max_batch_delay_ms = delay;
    SearchSession session(&engine, &backend, &responder, settings,
                          TestTimeManagement(), root, GoParams{.nodes = 2},
                          std::chrono::steady_clock::now());
    ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, root));
    ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, other));
    session.Start();
    const bool loading = store.load.WaitForArrival();
    const bool computing = backend.WaitForCompute();
    if (loading && computing) {
      EXPECT_FALSE(session.IsFinished());
      EXPECT_EQ(session.metrics().nn_evaluations.load(), 1u);
      EXPECT_EQ(session.metrics().partial_timeout_flushes.load(),
                delay ? 1u : 0u);
      EXPECT_EQ(session.metrics().partial_starvation_flushes.load(),
                delay ? 0u : 1u);
    }
    // This also exercises waking collectors and finishing returned store work
    // after stop. Never leave either executor blocked on a failed assertion.
    // The final info precedes release, so the late backup is not counted.
    session.Stop();
    const bool responding = responder.WaitForBestmove();
    store.Release();
    backend.Release();
    session.Wait();
    ASSERT_TRUE(loading);
    ASSERT_TRUE(computing)
        << "Collection depended on the blocked store returning";
    ASSERT_TRUE(responding);
    ExpectDrained(session, graph);
    ExpectFinalOutput(responder, 0);
    // The computed evaluation backs up; the other store miss is cancelled.
    EXPECT_EQ(session.metrics().visits_completed.load(), 1u);
    EXPECT_EQ(session.metrics().visits_cancelled.load(), 1u);
    EXPECT_EQ(session.metrics().node_store_misses.load(), 2u);
    EXPECT_EQ(store.recording.stored, 0u);
    EXPECT_EQ(backend.computes.load(), 1u);
  }
}

TEST(Lc5SearchTest, PositiveDelayCollectsAnotherWorkersStoreMissIntoSameBatch) {
  PolicyEngine<PuctAveraging> engine(TestParams());
  auto& graph = engine.graph();
  auto& store = InstallStore<BlockingStore>(engine, 2, false);
  ControlledBackend backend(ControlledBackend::Cache::kNone, true);
  RecordingResponder responder;
  const auto root = Origin();
  auto other = root;
  other.history.Append(root.history.Last().GetBoard().GenerateLegalMoves()[0]);
  other.key = MakeNodeKey(other.history, 7);
  auto settings = TestSettings(2, 2);
  settings.minibatch_size = 2;
  settings.max_batch_delay_ms = 60000;
  SearchSession session(&engine, &backend, &responder, settings,
                        TestTimeManagement(), root, GoParams{.nodes = 2},
                        std::chrono::steady_clock::now());
  ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, root));
  ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, other));
  session.Start();
  const bool loading = store.load.WaitForArrival();
  const bool first_input = backend.WaitForInputs(1);
  // The first miss has entered AddInput, but the other worker cannot submit
  // its miss until this barrier opens. No helper thread or sleep is needed to
  // prove the collector retains its first job across the store dependency.
  if (!loading || !first_input) session.Abort();
  store.Release();
  const bool computing = backend.WaitForCompute();
  if (!computing) session.Abort();
  backend.Release();
  session.Wait();
  ASSERT_TRUE(loading);
  ASSERT_TRUE(first_input);
  ASSERT_TRUE(computing);
  ExpectDrained(session, graph);
  ExpectFinalOutput(responder, 2);
  EXPECT_EQ(backend.batch_sizes, (std::vector<size_t>{2}));
  EXPECT_EQ(session.metrics().evaluation_batches.load(), 1u);
  EXPECT_EQ(session.metrics().partial_timeout_flushes.load(), 0u);
  EXPECT_EQ(session.metrics().partial_starvation_flushes.load(), 0u);
  EXPECT_EQ(store.recording.stored, 0u);
}

TEST(Lc5SearchTest, MultiworkerMixedCacheResultsRespectBatchAndItemCredits) {
  struct Case {
    int evaluators;
    int batch_size;
    int capacity;
    int replies_per_parent;
  };
  for (const auto& test :
       {Case{1, 4, 40, 2}, Case{2, 16, 64, 4}, Case{4, 32, 128, 8}}) {
    SCOPED_TRACE(test.evaluators);
    SCOPED_TRACE(test.batch_size);
    for (const int delay : {0, 10}) {
      SCOPED_TRACE(delay);
      PolicyEngine<PuctAveraging> engine(TestParams());
      auto& graph = engine.graph();
      ControlledBackend backend(ControlledBackend::Cache::kAlternating, true,
                                test.batch_size, test.evaluators);
      RecordingResponder responder;
      const auto root = Origin();
      auto settings = TestSettings(4, test.capacity);
      settings.eval_threads = test.evaluators;
      settings.minibatch_size = test.batch_size;
      settings.max_batch_delay_ms = delay;
      SearchSession session(
          &engine, &backend, &responder, settings, TestTimeManagement(), root,
          GoParams{.nodes = test.capacity}, std::chrono::steady_clock::now());
      std::vector<VisitOrigin> origins;
      const auto moves = root.history.Last().GetBoard().GenerateLegalMoves();
      ASSERT_EQ(moves.size(), 20u);
      // Pre-admission gives every worker independent leaves before submission.
      // Target 4 stresses credit replenishment with one-item jobs; targets
      // 16/32 exercise mixed results in four/eight-item jobs across evaluators.
      for (const auto move : moves) {
        auto parent = root;
        parent.history.Append(move);
        const auto replies =
            parent.history.Last().GetBoard().GenerateLegalMoves();
        ASSERT_GE(replies.size(), static_cast<size_t>(test.replies_per_parent));
        for (int i = 0; i < test.replies_per_parent; ++i) {
          auto origin = parent;
          origin.history.Append(replies[i]);
          origin.key = MakeNodeKey(origin.history, 7);
          ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, origin));
          origins.push_back(std::move(origin));
        }
        if (origins.size() == static_cast<size_t>(test.capacity)) break;
      }
      ASSERT_EQ(origins.size(), static_cast<size_t>(test.capacity));
      EXPECT_FALSE(SearchSessionTestPeer::AdmitVisit(session, root));
      session.Start();
      const bool computing = backend.WaitForCompute();
      if (!computing) session.Abort();
      // Release also opens the initial evaluator cohort on failure. Never
      // destroy a session with fixture barriers still blocking its executors.
      backend.Release();
      session.Wait();
      ASSERT_TRUE(computing);
      ExpectDrained(session, graph);
      ExpectFinalOutput(responder, test.capacity);
      EXPECT_EQ(backend.evaluator_ids.size(),
                static_cast<size_t>(test.evaluators));
      EXPECT_EQ(backend.inputs.load(), test.capacity);
      EXPECT_EQ(session.metrics().cache_hits.load(), test.capacity / 2);
      EXPECT_EQ(session.metrics().nn_evaluations.load(), test.capacity / 2);
      EXPECT_EQ(session.metrics().visits_admitted.load(), test.capacity);
      EXPECT_EQ(session.metrics().visits_completed.load(), test.capacity);
      EXPECT_EQ(session.metrics().visits_cancelled.load(), 0u);
      EXPECT_LE(session.metrics().active_visits_high_water.load(),
                test.capacity);
      EXPECT_EQ(graph.Size(), test.capacity);
      const size_t job_size = (test.batch_size + 3) / 4;
      const size_t credit_jobs =
          std::min<size_t>(16, 2 * test.batch_size / job_size);
      EXPECT_LE(session.metrics().io_jobs_high_water.load(), 4 * credit_jobs);
      if (delay == 0) {
        EXPECT_EQ(session.metrics().partial_timeout_flushes.load(), 0u);
      }
      size_t evaluated = 0;
      for (const auto size : backend.batch_sizes) {
        EXPECT_GT(size, 0u);
        EXPECT_LE(size, static_cast<size_t>(settings.minibatch_size));
        evaluated += size;
      }
      EXPECT_EQ(evaluated, test.capacity / 2);
      for (const auto& origin : origins) {
        const auto node = graph.SnapshotNode(origin.key);
        ASSERT_TRUE(node);
        EXPECT_EQ(node->stats.visits, 1u);
        EXPECT_FLOAT_EQ(node->stats.Q(), 0.25f);
        EXPECT_FLOAT_EQ(node->stats.D(), 0.5f);
        EXPECT_FLOAT_EQ(node->stats.M(), 3.0f);
        for (const auto& edge : node->edges)
          EXPECT_EQ(edge.prior, PackedPrior(1.0f / node->edges.size()));
      }
    }
  }
}

TEST(Lc5SearchTest, BlockedStoreLoadsDrainAfterStopOrAbort) {
  for (const bool hit : {false, true}) {
    for (const bool abort : {false, true}) {
      SCOPED_TRACE(hit);
      SCOPED_TRACE(abort);
      PolicyEngine<PuctAveraging> engine(TestParams());
      auto& graph = engine.graph();
      auto& store = InstallStore<BlockingStore>(engine);
      ControlledBackend backend(ControlledBackend::Cache::kAll);
      RecordingResponder responder;
      const auto root = Origin();
      const auto move = root.history.Last().GetBoard().GenerateLegalMoves()[0];
      graph.CompleteMaterialization(root.key, 1,
                                    {.moves = {move}, .priors = {1.0f}});
      auto child = root;
      child.history.Append(move);
      child.key = MakeNodeKey(child.history, 7);
      if (hit) {
        // Uniform priors, so edge order is by move encoding.
        auto legal = child.history.Last().GetBoard().GenerateLegalMoves();
        std::ranges::sort(legal, {}, &Move::raw_data);
        NodeState<PuctAveraging> stored{
            .lifecycle = NodeLifecycle::kExpanded,
            .stats = {.visits = 1, .q = 0.75f, .d = 0.25f, .m = 7.0f}};
        for (const Move legal_move : legal) {
          stored.edges.push_back(
              {.move = legal_move, .prior = 1.0f / legal.size()});
        }
        store.recording.entries.emplace(child.key, EncodeNode(stored));
      }
      SearchSession session(&engine, &backend, &responder, TestSettings(),
                            TestTimeManagement(), root,
                            GoParams{.nodes = 4, .infinite = true},
                            std::chrono::steady_clock::now());
      for (int i = 0; i < 4; ++i)
        ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, root));
      session.Start();
      const bool loading = store.load.WaitForArrival();
      if (loading) {
        EXPECT_EQ(TriedAt(*graph.SnapshotNode(root.key), 0).in_flight, 4u);
        EXPECT_EQ(session.metrics().ticket_waiters.load(), 3u);
        EXPECT_EQ(backend.inputs.load(), 0u);
      }
      if (abort || !loading)
        session.Abort();
      else {
        session.Stop();
        EXPECT_TRUE(responder.WaitForBestmove());
      }
      EXPECT_FALSE(SearchSessionTestPeer::AdmitVisit(session, root));
      EXPECT_FALSE(session.IsFinished());
      store.load.Release();
      store.Release();
      session.Wait();
      ASSERT_TRUE(loading);
      ExpectDrained(session, graph);
      // A load already running at stop completes: on a hit, the node is
      // restored, but a restored node has nothing new to back up, so the
      // owner and the waiters all cancel. A miss is never evaluated.
      EXPECT_EQ(session.metrics().visits_admitted.load(), 4u);
      EXPECT_EQ(session.metrics().visits_completed.load(), 0u);
      EXPECT_EQ(session.metrics().visits_cancelled.load(), 4u);
      EXPECT_EQ(session.metrics().node_store_hits.load(), hit ? 1u : 0u);
      EXPECT_EQ(session.metrics().nodes_restored.load(), hit ? 1u : 0u);
      EXPECT_EQ(session.metrics().node_store_misses.load(), hit ? 0u : 1u);
      EXPECT_EQ(store.recording.stored, 0u);
      EXPECT_EQ(backend.inputs.load(), 0u);
      EXPECT_EQ(backend.computes.load(), 0u);
      EXPECT_EQ(store.recording.entries.contains(child.key), hit);
      EXPECT_EQ(graph.SnapshotNode(child.key).has_value(), hit);
      if (hit) {
        const auto restored = graph.SnapshotNode(child.key);
        EXPECT_EQ(restored->stats.visits, 1u);
        EXPECT_FLOAT_EQ(restored->stats.Q(), 0.75f);
      }
      // Only the setup install.
      EXPECT_EQ(graph.SnapshotNode(root.key)->stats.visits, 1u);
      EXPECT_EQ(TriedAt(*graph.SnapshotNode(root.key), 0).stats.visits, 0u);
      if (abort) {
        EXPECT_TRUE(responder.bestmoves.empty());
        EXPECT_TRUE(std::none_of(responder.thinking.begin(),
                                 responder.thinking.end(),
                                 [](const ThinkingInfo& info) {
                                   return info.comment.ends_with(" final");
                                 }));
      } else {
        ExpectFinalOutput(responder, 0);
      }
    }
  }
}

std::vector<VisitOrigin> AdmitChildren(SearchSession& session,
                                       const VisitOrigin& root, size_t count) {
  std::vector<VisitOrigin> origins;
  const auto legal = root.history.Last().GetBoard().GenerateLegalMoves();
  for (size_t i = 0; i < count; ++i) {
    auto origin = root;
    origin.history.Append(legal[i]);
    origin.key = MakeNodeKey(origin.history, 7);
    EXPECT_TRUE(SearchSessionTestPeer::AdmitVisit(session, origin));
    origins.push_back(std::move(origin));
  }
  SearchSessionTestPeer::PrepareAdmissions(session);
  return origins;
}

TEST(Lc5SearchTest, QueuedBacklogCannotComputeAgainAfterStop) {
  for (const int evaluators : {1, 3}) {
    SCOPED_TRACE(evaluators);
    PolicyEngine<PuctAveraging> engine(TestParams());
    auto& graph = engine.graph();
    ControlledBackend backend(ControlledBackend::Cache::kNone, true, 4,
                              evaluators);
    RecordingResponder responder;
    const auto root = Origin();
    auto settings = TestSettings(4, 20);
    settings.eval_threads = evaluators;
    SearchSession session(&engine, &backend, &responder, settings,
                          TestTimeManagement(), root, GoParams{.nodes = 20},
                          std::chrono::steady_clock::now());
    AdmitChildren(session, root, 20);
    session.Start();
    // Each evaluator is executing one authorized batch, with more ready jobs
    // than that cohort can consume. Stop must not authorize a second batch.
    const bool computing = backend.WaitForCompute(evaluators);
    session.Stop();
    const bool responding = responder.WaitForBestmove();
    EXPECT_FALSE(session.IsFinished());
    backend.Release();
    session.Wait();
    ASSERT_TRUE(computing);
    ASSERT_TRUE(responding);
    EXPECT_EQ(backend.computes.load(), static_cast<size_t>(evaluators));
    EXPECT_EQ(backend.calls_by_evaluator.size(),
              static_cast<size_t>(evaluators));
    for (const auto& [id, calls] : backend.calls_by_evaluator)
      EXPECT_EQ(calls, 1u);
    EXPECT_EQ(backend.inputs.load(), 4u * evaluators);
    EXPECT_EQ(graph.Size(), 4u * evaluators);
    // Authorized batches back up their visits; the queued backlog cancels.
    EXPECT_EQ(session.metrics().visits_completed.load(), 4u * evaluators);
    ExpectDrained(session, graph);
    ExpectLegalBestmove(responder, root);
  }
}

TEST(Lc5SearchTest, AuthorizedEvaluationsBackUpToRootAfterStop) {
  PolicyEngine<PuctAveraging> engine(TestParams());
  auto& graph = engine.graph();
  ControlledBackend backend(ControlledBackend::Cache::kNone, true);
  RecordingResponder responder;
  const auto root = Origin();
  // Uniform priors, so edge order is by move encoding.
  auto legal = root.history.Last().GetBoard().GenerateLegalMoves();
  std::ranges::sort(legal, {}, &Move::raw_data);
  graph.CompleteMaterialization(
      root.key, 1,
      {.moves = {legal.begin(), legal.end()},
       .priors = std::vector<float>(legal.size(), 1.0f / legal.size())});
  SearchSession session(&engine, &backend, &responder, TestSettings(4, 20),
                        TestTimeManagement(), root, GoParams{.nodes = 20},
                        std::chrono::steady_clock::now());
  for (int i = 0; i < 20; ++i)
    ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, root));
  session.Start();
  // One authorized batch of four children is computing; the other visits wait
  // on tickets of any worker, or are still queued, and must cancel.
  const bool computing = backend.WaitForCompute();
  session.Stop();
  const bool responding = responder.WaitForBestmove();
  EXPECT_FALSE(session.IsFinished());
  backend.Release();
  session.Wait();
  ASSERT_TRUE(computing);
  ASSERT_TRUE(responding);
  ExpectDrained(session, graph);
  EXPECT_EQ(backend.computes.load(), 1u);
  EXPECT_EQ(backend.inputs.load(), 4u);
  // Each computed ticket's owner backs up through the root after stop.
  EXPECT_EQ(session.metrics().visits_completed.load(), 4u);
  const auto node = graph.SnapshotNode(root.key);
  ASSERT_TRUE(node);
  EXPECT_EQ(node->stats.visits, 5u);
  uint64_t edge_visits = 0;
  for (const auto& tried : node->tried) edge_visits += tried.stats.visits;
  EXPECT_EQ(edge_visits, 4u);
  ExpectFinalOutput(responder, 20);
}

TEST(Lc5SearchTest, StopDuringAddInputKeepsOnlyImmediateResults) {
  PolicyEngine<PuctAveraging> engine(TestParams());
  auto& graph = engine.graph();
  ControlledBackend backend(ControlledBackend::Cache::kAlternating, true, 8);
  backend.blocked_input = 2;
  RecordingResponder responder;
  const auto root = Origin();
  auto settings = TestSettings(1, 4);
  settings.minibatch_size = 8;
  SearchSession session(&engine, &backend, &responder, settings,
                        TestTimeManagement(), root, GoParams{.nodes = 4},
                        std::chrono::steady_clock::now());
  const auto origins = AdmitChildren(session, root, 4);
  session.Start();
  // The first job has an immediate hit and an AddInput that will enqueue only
  // after release. The second job must not even reach AddInput after stop.
  const bool adding = backend.input.WaitForArrival();
  session.Stop();
  const bool responding = responder.WaitForBestmove();
  EXPECT_FALSE(session.IsFinished());
  backend.Release();
  session.Wait();
  ASSERT_TRUE(adding);
  ASSERT_TRUE(responding);
  EXPECT_EQ(backend.inputs.load(), 2u);
  EXPECT_EQ(backend.computes.load(), 0u);
  EXPECT_EQ(graph.Size(), 1u);
  auto hit = graph.SnapshotNode(origins[0].key);
  ASSERT_TRUE(hit);
  // The install counted the leaf; its owner completes with nothing above it.
  EXPECT_EQ(hit->stats.visits, 1u);
  EXPECT_EQ(session.metrics().visits_completed.load(), 1u);
  EXPECT_EQ(session.metrics().visits_cancelled.load(), 3u);
  EXPECT_FLOAT_EQ(hit->stats.Q(), backend.q);
  for (const auto& edge : hit->edges)
    EXPECT_EQ(edge.prior, PackedPrior(1.0f / hit->edges.size()));
  for (size_t i = 1; i < origins.size(); ++i)
    EXPECT_FALSE(graph.SnapshotNode(origins[i].key));
  ExpectDrained(session, graph);
}

TEST(Lc5SearchTest, RetainedNonfittingJobIsCancelledAfterAuthorizedBatch) {
  PolicyEngine<PuctAveraging> engine(TestParams());
  auto& graph = engine.graph();
  ControlledBackend backend(ControlledBackend::Cache::kNone, true, 5);
  RecordingResponder responder;
  const auto root = Origin();
  auto settings = TestSettings(1, 6);
  settings.minibatch_size = 5;
  SearchSession session(&engine, &backend, &responder, settings,
                        TestTimeManagement(), root, GoParams{.nodes = 6},
                        std::chrono::steady_clock::now());
  const auto origins = AdmitChildren(session, root, 6);
  session.Start();
  // Three whole jobs of two inputs: after adding four inputs, the collector
  // has popped and retained the last job because 4 + 2 exceeds the target 5.
  const bool computing = backend.WaitForCompute();
  session.Stop();
  const bool responding = responder.WaitForBestmove();
  backend.Release();
  session.Wait();
  ASSERT_TRUE(computing);
  ASSERT_TRUE(responding);
  EXPECT_EQ(backend.batch_sizes, (std::vector<size_t>{4}));
  EXPECT_EQ(backend.inputs.load(), 4u);
  EXPECT_EQ(backend.computes.load(), 1u);
  EXPECT_EQ(graph.Size(), 4u);
  for (size_t i = 4; i < origins.size(); ++i)
    EXPECT_FALSE(graph.SnapshotNode(origins[i].key));
  ExpectDrained(session, graph);
}

TEST(Lc5SearchTest, AbortOverridesRepeatedStopBeforeCommitment) {
  PolicyEngine<PuctAveraging> engine(TestParams());
  auto& graph = engine.graph();
  ControlledBackend backend;
  RecordingResponder responder;
  const auto root = Origin();
  SearchSession session(&engine, &backend, &responder, TestSettings(),
                        TestTimeManagement(), root, GoParams{.nodes = 4},
                        std::chrono::steady_clock::now());
  AdmitChildren(session, root, 4);
  session.Stop();
  session.Stop();
  session.Abort();
  session.Stop();
  session.Abort();
  session.Start();
  session.Wait();
  ExpectDrained(session, graph);
  EXPECT_TRUE(responder.bestmoves.empty());
  EXPECT_TRUE(responder.thinking.empty());
  EXPECT_EQ(backend.inputs.load(), 0u);
  EXPECT_EQ(backend.computes.load(), 0u);
  EXPECT_EQ(graph.Size(), 0u);
}

TEST(Lc5SearchTest, RespondsBeforeBlockedEvictionDrains) {
  PolicyEngine<PuctAveraging> engine(TestParams());
  auto& graph = engine.graph();
  auto& store = InstallStore<BlockingStore>(engine, 0);
  ControlledBackend backend(ControlledBackend::Cache::kNone, true);
  RecordingResponder responder;
  const auto root = Origin();
  const auto legal = root.history.Last().GetBoard().GenerateLegalMoves();
  // The one visit takes the first edge and blocks in the backend; the
  // sibling below the second edge is the node to evict.
  graph.CompleteMaterialization(
      root.key, 1, {.moves = {legal[0], legal[1]}, .priors = {0.9f, 0.1f}});
  auto sibling = root;
  sibling.history.Append(legal[1]);
  sibling.key = MakeNodeKey(sibling.history, 7);
  graph.CompleteMaterialization(
      sibling.key, 2, {.leaf = {0.5f, 0.0f, 0.0f}, .moves = {}, .priors = {}});
  auto settings = TestSettings(1, 1);
  settings.graph_node_limit = 1;
  // Store threads come one per eval thread. With only one, the visit's store
  // load could queue behind the blocked eviction and never reach the backend.
  settings.eval_threads = 2;
  SearchSession session(&engine, &backend, &responder, settings,
                        TestTimeManagement(), root, GoParams{.nodes = 1},
                        std::chrono::steady_clock::now());
  session.Start();
  const bool persisting = store.persist.WaitForArrival();
  const bool computing = backend.WaitForCompute();
  session.Stop();
  const bool responding = responder.WaitForBestmove();
  EXPECT_FALSE(session.IsFinished());
  store.Release();
  backend.Release();
  session.Wait();
  ASSERT_TRUE(persisting);
  ASSERT_TRUE(computing);
  ASSERT_TRUE(responding);
  ExpectDrained(session, graph);
  ExpectFinalOutput(responder, 0);
  // Stored before it was erased.
  EXPECT_EQ(session.metrics().nodes_evicted.load(), 1u);
  EXPECT_TRUE(store.recording.entries.contains(sibling.key));
  EXPECT_FALSE(graph.SnapshotNode(sibling.key));
  EXPECT_TRUE(graph.SnapshotNode(root.key));
}

TEST(Lc5SearchTest, EvictionKeepsGraphSmallAndLosesNoVisits) {
  for (const int threads : {1, 3}) {
    SCOPED_TRACE(threads);
    PolicyEngine<PuctAveraging> engine(TestParams());
    auto& graph = engine.graph();
    auto& store = InstallStore<MemoryNodeStore>(engine);
    ControlledBackend backend;
    RecordingResponder responder;
    const auto root = Origin();
    auto settings = TestSettings(threads, 8);
    settings.graph_node_limit = 32;
    SearchSession session(&engine, &backend, &responder, settings,
                          TestTimeManagement(), root, GoParams{.nodes = 3000},
                          std::chrono::steady_clock::now());
    session.Start();
    session.Wait();
    ExpectDrained(session, graph);
    ExpectFinalOutput(responder, 3000);
    const auto& m = session.metrics();
    EXPECT_EQ(m.visits_completed.load(), 3000u);
    EXPECT_GT(m.nodes_evicted.load(), 0u);
    EXPECT_GT(m.nodes_restored.load(), 0u);
    EXPECT_EQ(m.restores_rejected.load(), 0u);
    EXPECT_EQ(m.invariant_underflow_prevented.load(), 0u);
    // Every completed visit reached the root, which is never evicted.
    const auto node = graph.SnapshotNode(root.key);
    ASSERT_TRUE(node);
    EXPECT_EQ(node->stats.visits, m.visits_completed.load());
    uint64_t edge_visits = 0;
    for (const auto& tried : node->tried) edge_visits += tried.stats.visits;
    EXPECT_EQ(edge_visits, node->stats.visits - 1);
    // Far fewer nodes than visits stay in RAM; every evicted one is stored.
    EXPECT_LT(graph.Size(), 3000u / 4);
    EXPECT_GE(store.Size(), m.nodes_evicted.load() - m.nodes_restored.load());
  }
}

TEST(Lc5SearchTest, SingleNodeSearchPlaysTopPriorMove) {
  PolicyEngine<PuctAveraging> engine(TestParams());
  auto& graph = engine.graph();
  ControlledBackend backend;
  RecordingResponder responder;
  // Black to move, so the PV move is flipped back for UCI.
  const auto root =
      Origin("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
  SearchSession session(&engine, &backend, &responder, TestSettings(),
                        TestTimeManagement(), root, GoParams{.nodes = 1},
                        std::chrono::steady_clock::now());
  session.Start();
  session.Wait();
  ExpectDrained(session, graph);
  ExpectFinalOutput(responder, 1);
  const auto node = graph.SnapshotNode(root.key);
  ASSERT_TRUE(node);
  ASSERT_FALSE(node->edges.empty());
  for (const auto& tried : node->tried) EXPECT_EQ(tried.stats.visits, 0u);
  Move expected = node->edges[0].move;
  expected.Flip();
  ASSERT_EQ(responder.thinking.back().pv.size(), 1u);
  EXPECT_EQ(responder.thinking.back().pv[0], expected);
  EXPECT_EQ(responder.bestmoves[0].bestmove, expected);
}

TEST(Lc5SearchTest,
     FreshRootStoppedBeforeStartUsesLegalFallbackWithoutEvaluation) {
  PolicyEngine<PuctAveraging> engine(TestParams());
  auto& graph = engine.graph();
  ControlledBackend backend;
  RecordingResponder responder;
  const auto root = Origin();
  SearchSession session(&engine, &backend, &responder, TestSettings(),
                        TestTimeManagement(), root, GoParams{},
                        std::chrono::steady_clock::now());
  ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, root));
  session.Stop();
  session.Start();
  session.Wait();
  ExpectDrained(session, graph);
  ExpectLegalBestmove(responder, root);
  EXPECT_EQ(backend.inputs.load(), 0u);
  EXPECT_EQ(backend.computes.load(), 0u);
  EXPECT_EQ(graph.Size(), 0u);
}

TEST(Lc5SearchTest, EnginePreservesExplicitMovetimeAndInfinitePrecedence) {
  for (const auto& params :
       {GoParams{.wtime = 0, .btime = 0, .nodes = 1, .movetime = 60000},
        GoParams{.wtime = 0, .btime = 0, .nodes = 1, .infinite = true},
        GoParams{.wtime = 0,
                 .btime = 0,
                 .nodes = 1,
                 .movetime = 60000,
                 .infinite = true}}) {
    for (const auto* fen :
         {ChessBoard::kStartposFen,
          "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"}) {
      SCOPED_TRACE(fen);
      SCOPED_TRACE(params.infinite);
      SCOPED_TRACE(params.movetime.value_or(-1));
      OptionsParser parser;
      Settings::Populate(&parser);
      PopulatePolicyOptions(&parser);
      parser.SetUciOption("Threads", "1");
      parser.SetUciOption("EvalThreads", "1");
      parser.SetUciOption("MaxActiveVisits", "1");
      ControlledBackend backend;
      RecordingResponder responder;
      Lc5Engine engine(&responder, &parser.GetOptionsDict());
      engine.SetBackend(&backend);
      engine.SetPosition({.startpos = Position::FromFen(fen), .moves = {}});
      engine.StartClock();
      engine.StartSearch(params);
      engine.WaitSearch();
      ExpectFinalOutput(responder, 1);
      ExpectLegalBestmove(responder, Origin(fen));
      EXPECT_EQ(backend.inputs.load(), 1u);
    }
  }
}

TEST(Lc5SearchTest, EngineRetainsGraphAcrossSearchesAndResetsForNewGame) {
  OptionsParser parser;
  Settings::Populate(&parser);
  PopulatePolicyOptions(&parser);
  parser.SetUciOption("Threads", "1");
  parser.SetUciOption("EvalThreads", "1");
  parser.SetUciOption("MaxActiveVisits", "1");
  ControlledBackend backend;
  RecordingResponder responder;
  Lc5Engine engine(&responder, &parser.GetOptionsDict());
  engine.SetBackend(&backend);
  const GameState state{.startpos = Position::FromFen(ChessBoard::kStartposFen),
                        .moves = {}};
  for (int i = 0; i < 3; ++i) {
    if (i == 2) engine.NewGame();
    // Repeated SetPosition must retain nodes for the same game as well.
    engine.SetPosition(state);
    engine.StartClock();
    engine.StartSearch(GoParams{.nodes = 1});
    engine.WaitSearch();
    EXPECT_EQ(responder.bestmoves.size(), static_cast<size_t>(i + 1));
    EXPECT_EQ(responder.thinking.back().nodes, 1);
    EXPECT_NE(responder.thinking.back().comment.find(" active="),
              std::string::npos);
    EXPECT_TRUE(responder.thinking.back().comment.ends_with(" final"));
    SearchSessionTestPeer::ExpectDrained(engine);
  }
  // First run evaluates the root; retained search evaluates its child instead
  // of evaluating the root again. NewGame resets that retained expansion.
  EXPECT_EQ(backend.history_sizes, (std::vector<size_t>{1, 2, 1}));
}

}  // namespace
}  // namespace lczero::lc5

int main(int argc, char** argv) {
  lczero::InitializeMagicBitboards();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
