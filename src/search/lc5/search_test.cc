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

#include <absl/algorithm/container.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "search/lc5/engine.h"
#include "search/lc5/policies/puct.h"
#include "search/lc5/policies/registry.h"
#include "search/lc5/session/search_session.h"
#include "search/lc5/store/memory_node_store.h"
#include "search/lc5/visit_engine.h"

namespace lczero::lc5 {

class SearchSessionTestPeer {
 public:
  // Admits a visit from an arbitrary origin, sharing the node budget with
  // root visits, and gives it to the workers in turn. Succeeds only before
  // Start(): returns false if admission is closed, the budget is exhausted or
  // the pool is full.
  static bool AdmitVisit(SearchSession& session, const VisitOrigin& origin)
      ABSL_NO_THREAD_SAFETY_ANALYSIS {
    const size_t worker_idx =
        session.visit_pool_.active() % session.visit_workers_.size();
    const auto admitted = session.limiter_.AdmitVisits(worker_idx, origin, 1);
    if (admitted.empty()) return false;
    // Starting the worker threads orders this before they run.
    assert(!session.started_.load());
    auto& worker = *session.visit_workers_[worker_idx];
    ++worker.active_visits_;
    worker.ready_visits_.push_back(admitted[0]);
    return true;
  }
  // Before Start only: make the entire backlog ready before collectors start.
  static void PrepareAdmissions(SearchSession& session)
      ABSL_NO_THREAD_SAFETY_ANALYSIS {
    for (auto& worker : session.visit_workers_) {
      while (!worker->ready_visits_.empty()) {
        const VisitId id = worker->ready_visits_.front();
        worker->ready_visits_.pop_front();
        worker->AdvanceVisit(id);
      }
      worker->SendJobs();
      worker->ReportProducingStatus(false);
    }
  }
  // Inspect owner-local state only after Wait's full join barrier.
  static void ExpectDrained(const SearchSession& session)
      ABSL_NO_THREAD_SAFETY_ANALYSIS {
    EXPECT_EQ(session.visit_pool_.active(), 0u);
    EXPECT_TRUE(session.materializations_.empty());
    EXPECT_EQ(session.jobs_in_flight_.load(), 0u);
    EXPECT_EQ(session.evictions_in_flight_.load(), 0u);
    EXPECT_EQ(session.evaluator_.jobs_in_collection_.load(), 0u);
    EXPECT_EQ(session.evaluator_.queue_size(), 0u);
    EXPECT_EQ(session.store_jobs_.Size(), 0u);
    for (const auto& worker : session.visit_workers_) {
      EXPECT_TRUE(worker->drained_.load());
      EXPECT_EQ(worker->active_visits_, 0u);
      EXPECT_TRUE(worker->ready_visits_.empty());
      EXPECT_TRUE(worker->local_wakeups_.empty());
      EXPECT_TRUE(worker->queued_loads_.empty());
      EXPECT_TRUE(worker->queued_evals_.empty());
      EXPECT_EQ(worker->jobs_in_flight_, 0u);
      EXPECT_EQ(worker->materializations_in_flight_, 0u);
    }
    for (const auto& inbox : session.inboxes_) {
      EXPECT_TRUE(inbox.wakeups.empty());
      EXPECT_TRUE(inbox.completions.empty());
    }
  }
  static uint64_t ComputeRootRemainingVisits(
      const SearchSession& session,
      std::chrono::steady_clock::time_point now) {
    return session.ComputeRootRemainingVisits(now);
  }
  static void ExpectDrained(const Lc5Engine& engine) {
    ASSERT_TRUE(engine.session_);
    ASSERT_TRUE(engine.session_->IsFinished());
    ExpectDrained(*engine.session_);
    ASSERT_TRUE(engine.visit_engine_);
  }
};

namespace {

Settings::Resolved TestSettings(int threads = 1, int capacity = 4) {
  return {.threads = threads,
          .eval_threads = 1,
          .minibatch_size = 4,
          .max_active_visits = capacity,
          .max_batch_delay_ms = 0,
          .active_visits_ratio = 0.0f};
}

std::unique_ptr<VisitEngine> MakeTestEngine(std::string_view policy = "puct",
                                            bool sticky_endgames = false,
                                            int history_key_length = 7) {
  OptionsParser parser;
  Settings::Populate(&parser);
  PopulatePolicyOptions(&parser);
  parser.SetUciOption("CPuct", "1");
  parser.SetUciOption("CPuctBase", "100");
  parser.SetUciOption("CPuctFactor", "0");
  parser.SetUciOption("FpuStrategy", "absolute");
  parser.SetUciOption("FpuValue", "0");
  parser.SetUciOption("StickyEndgames", sticky_endgames ? "true" : "false");
  parser.SetUciOption("HistoryKeyLength", std::to_string(history_key_length));
  auto engine = MakeVisitEngine(policy);
  engine->Configure(parser.GetOptionsDict(), {.has_mlh = true,
                                              .has_wdl = true,
                                              .runs_on_cpu = true,
                                              .suggested_num_search_threads = 1,
                                              .recommended_batch_size = 4,
                                              .maximum_batch_size = 4});
  return engine;
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
    {
      std::lock_guard lock(mutex_);
      thinking.insert(thinking.end(), infos->begin(), infos->end());
    }
    cv_.notify_all();
  }
  template <typename Pred>
  bool WaitForThinking(Pred pred, std::chrono::milliseconds timeout =
                                      std::chrono::milliseconds(500)) {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, timeout, [&] {
      return std::any_of(thinking.begin(), thinking.end(), pred);
    });
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

// Gives `engine` a new store and returns it.
template <typename S, typename... Args>
S& InstallStore(VisitEngine& engine, Args&&... args) {
  auto store = std::make_unique<S>(std::forward<Args>(args)...);
  S& result = *store;
  engine.SetStore(std::move(store));
  return result;
}

void ExpectDrained(const SearchSession& session) {
  ASSERT_TRUE(session.IsFinished());
  SearchSessionTestPeer::ExpectDrained(session);
  const auto& m = session.metrics();
  EXPECT_EQ(m.visits_admitted.load(),
            m.visits_completed.load() + m.visits_cancelled.load());
  EXPECT_EQ(m.publications_rejected.load(), 0u);
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
    auto engine = MakeTestEngine();
    auto& store = InstallStore<RecordingStore>(*engine);
    ControlledBackend backend;
    RecordingResponder responder;
    const auto root = Origin();
    // A synthetic old origin also verifies that preparation time is charged.
    SearchSession session(
        engine.get(), &backend, &responder, TestSettings(),
        TestTimeManagement(), root, params,
        std::chrono::steady_clock::now() - std::chrono::minutes(2));
    ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, root));
    session.Start();
    session.Wait();
    // Startup can race expiration; neither completed visits nor evaluations
    // are required to be zero. All admitted work must still drain.
    ExpectDrained(session);
    ExpectFinalOutput(responder, session.metrics().visits_completed.load());
    ExpectLegalBestmove(responder, root);
    // Nothing is evicted without a graph node limit.
    EXPECT_EQ(store.stored, 0u);
  }
}

TEST(Lc5SearchTest, AbortSuppressesOutputWithExpiredTimeBudget) {
  auto engine = MakeTestEngine();
  InstallStore<RecordingStore>(*engine);
  ControlledBackend backend;
  RecordingResponder responder;
  const auto root = Origin();
  SearchSession session(
      engine.get(), &backend, &responder, TestSettings(), TestTimeManagement(),
      root, GoParams{.movetime = 0},
      std::chrono::steady_clock::now() - std::chrono::minutes(2));
  ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, root));
  // Abort before startup so output commitment cannot race the assertion.
  session.Abort();
  session.Start();
  session.Wait();
  ExpectDrained(session);
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
      auto engine = MakeTestEngine();
      ControlledBackend backend;
      RecordingResponder responder;
      const auto root = Origin(fen);
      SearchSession session(engine.get(), &backend, &responder, TestSettings(),
                            TestTimeManagement(), root, params,
                            std::chrono::steady_clock::now());
      session.Start();
      session.Wait();
      ExpectDrained(session);
      EXPECT_EQ(session.metrics().visits_admitted.load(), 1u);
      EXPECT_EQ(session.metrics().visits_completed.load(), 1u);
      EXPECT_EQ(session.metrics().visits_cancelled.load(), 0u);
      ExpectFinalOutput(responder, 1);
      ExpectLegalBestmove(responder, root);
    }
  }
}

TEST(Lc5SearchTest, PonderIgnoresZeroNodeAndTimeLimits) {
  auto engine = MakeTestEngine();
  ControlledBackend backend(ControlledBackend::Cache::kNone, true);
  RecordingResponder responder;
  const auto root = Origin();
  SearchSession session(
      engine.get(), &backend, &responder, TestSettings(1, 1),
      TestTimeManagement(), root,
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
  ExpectDrained(session);
  EXPECT_GE(session.metrics().visits_admitted.load(), 2u);
  ExpectFinalOutput(responder, session.metrics().visits_completed.load());
  ExpectLegalBestmove(responder, root);
}

TEST(Lc5SearchTest, ExactNodeBudgetWithMoreWorkersThanCapacity) {
  for (const int64_t nodes : {0, 1, 7}) {
    SCOPED_TRACE(nodes);
    auto engine = MakeTestEngine();
    ControlledBackend backend;
    RecordingResponder responder;
    auto root = Origin();
    SearchSession session(engine.get(), &backend, &responder,
                          TestSettings(8, 2), TestTimeManagement(), root,
                          GoParams{.nodes = nodes},
                          std::chrono::steady_clock::now());
    session.Start();
    session.Wait();
    ExpectDrained(session);
    const uint64_t expected = std::max<int64_t>(1, nodes);
    EXPECT_EQ(session.metrics().visits_admitted.load(), expected);
    EXPECT_EQ(session.metrics().visits_completed.load(), expected);
    EXPECT_EQ(session.metrics().visits_cancelled.load(), 0u);
    EXPECT_LE(session.metrics().active_visits_high_water.load(), 2u);
    EXPECT_FALSE(SearchSessionTestPeer::AdmitVisit(session, root));
    ExpectFinalOutput(responder, expected);
    EXPECT_TRUE(responder.thinking.back().score.has_value());
    EXPECT_TRUE(responder.thinking.back().wdl.has_value());
    const auto legal = root.history.Last().GetBoard().GenerateLegalMoves();
    EXPECT_NE(absl::c_find(legal, responder.bestmoves[0].bestmove),
              legal.end());
    // Each completed visit counts once at the root, including the one that
    // materialized it.
    const RootStats stats = engine->GetRootStats(root.key);
    ASSERT_TRUE(stats.materialized);
    EXPECT_EQ(stats.visits, expected);
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
      auto engine = MakeTestEngine();
      ControlledBackend backend;
      backend.q = test.q;
      RecordingResponder responder;
      SearchSession session(engine.get(), &backend, &responder, TestSettings(),
                            TestTimeManagement(), Origin(fen),
                            GoParams{.nodes = 1},
                            std::chrono::steady_clock::now());
      session.Start();
      session.Wait();
      ExpectDrained(session);
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
  auto engine = MakeTestEngine();
  ControlledBackend backend;
  RecordingResponder responder;
  auto root = Origin();
  SearchSession session(engine.get(), &backend, &responder, TestSettings(),
                        TestTimeManagement(), root, GoParams{.nodes = 4},
                        std::chrono::steady_clock::now());
  // One worker consumes all four admissions before sending the root job.
  for (int i = 0; i < 4; ++i)
    ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, root));
  EXPECT_FALSE(SearchSessionTestPeer::AdmitVisit(session, root));
  session.Start();
  session.Wait();
  ExpectDrained(session);
  ExpectFinalOutput(responder, 4);
  EXPECT_EQ(session.metrics().visits_completed.load(), 4u);
  EXPECT_GE(session.metrics().materialization_waiters.load(), 3u);
  EXPECT_GE(session.metrics().visits_resumed.load(), 3u);
  const RootStats stats = engine->GetRootStats(root.key);
  ASSERT_TRUE(stats.materialized);
  EXPECT_EQ(stats.visits, 4u);
  // Only the owner backs up at the root leaf; each waiter traverses an edge to
  // a distinct child.
  EXPECT_EQ(stats.best_edge_visits, 1u);
  EXPECT_EQ(engine->GetGraphSize(), 4u);
}

TEST(Lc5SearchTest, WaitersDrainAndCountOnce) {
  for (const int threads : {1, 2}) {
    SCOPED_TRACE(threads);
    auto engine = MakeTestEngine();
    ControlledBackend backend;
    RecordingResponder responder;
    auto root = Origin();
    constexpr int64_t kNodes = 64;
    SearchSession session(engine.get(), &backend, &responder,
                          TestSettings(threads, 16), TestTimeManagement(), root,
                          GoParams{.nodes = kNodes},
                          std::chrono::steady_clock::now());
    session.Start();
    session.Wait();
    ExpectDrained(session);
    ExpectFinalOutput(responder, kNodes);
    EXPECT_EQ(session.metrics().visits_completed.load(), kNodes);
    const RootStats stats = engine->GetRootStats(root.key);
    ASSERT_TRUE(stats.materialized);
    EXPECT_EQ(stats.visits, static_cast<uint64_t>(kNodes));
  }
}

TEST(Lc5SearchTest, WaiterReselectsAfterOwnerBackup) {
  const auto root = Origin();
  const auto moves = root.history.Last().GetBoard().GenerateLegalMoves();
  for (const int threads : {1, 2}) {
    for (const bool child_bad_for_root : {true, false}) {
      SCOPED_TRACE(threads);
      SCOPED_TRACE(child_bad_for_root);
      auto engine = MakeTestEngine();
      engine->CompleteMaterialization(
          root.key, {.moves = {moves[0], moves[1]}, .priors = {0.8f, 0.2f}});
      ControlledBackend backend;
      // Positive child Q flips to negative on the root's edge.
      backend.q = child_bad_for_root ? 0.9f : -0.9f;
      RecordingResponder responder;
      SearchSession session(engine.get(), &backend, &responder,
                            TestSettings(threads, 4), TestTimeManagement(),
                            root, GoParams{.nodes = 2},
                            std::chrono::steady_clock::now());
      // Both visits pick moves[0] (0.8, then 0.8 / 2 = 0.4 > 0.2): one owner
      // and one waiter (on the other worker when threads == 2).
      ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, root));
      ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, root));
      SearchSessionTestPeer::PrepareAdmissions(session);
      session.Start();
      session.Wait();
      ExpectDrained(session);
      ExpectFinalOutput(responder, 3);
      EXPECT_EQ(session.metrics().visits_completed.load(), 2u);
      EXPECT_EQ(session.metrics().materialization_waiters.load(), 1u);
      EXPECT_EQ(session.metrics().visits_reselected.load(),
                child_bad_for_root ? 1u : 0u);
      const RootStats stats = engine->GetRootStats(root.key);
      ASSERT_TRUE(stats.materialized);
      EXPECT_EQ(stats.best_edge_visits, child_bad_for_root ? 1u : 2u);
      EXPECT_EQ(stats.second_edge_visits, child_bad_for_root ? 1u : 0u);
    }
  }
}

TEST(Lc5SearchTest, PuctDagDrainsOverTranspositions) {
  // Kings and blocked pawns: with HistoryKeyLength 0, 1. Kg1 Kb8 2. Kg2 and
  // 1. Kh2 Kb8 2. Kg2 commute to the same node at the third ply.
  auto settings = TestSettings();
  PositionHistory history;
  history.Reset(Position::FromFen("k7/p7/P7/8/8/8/8/7K w - - 0 1"));
  PositionHistory parent = history;
  parent.Append(Move::White({kFileH, kRank1}, {kFileG, kRank1}));
  parent.Append(parent.Last().GetBoard().GenerateLegalMoves()[0]);
  PositionHistory shared = parent;
  shared.Append(Move::White({kFileG, kRank1}, {kFileG, kRank2}));
  const NodeKey parent_key = MakeNodeKey(parent, 0);
  const NodeKey shared_key = MakeNodeKey(shared, 0);
  const VisitOrigin root{.key = MakeNodeKey(history, 0),
                         .history = std::move(history),
                         .backup_prefix = {}};
  for (const int threads : {1, 2}) {
    SCOPED_TRACE(threads);
    settings.threads = threads;
    settings.max_active_visits = 16;
    auto engine = MakeTestEngine("puct-dag", /*sticky_endgames=*/false,
                                 /*history_key_length=*/0);
    ControlledBackend backend;
    RecordingResponder responder;
    constexpr int64_t kNodes = 2000;
    SearchSession session(engine.get(), &backend, &responder, settings,
                          TestTimeManagement(), root, GoParams{.nodes = kNodes},
                          std::chrono::steady_clock::now());
    session.Start();
    session.Wait();
    ExpectDrained(session);
    ExpectFinalOutput(responder, kNodes);
    ExpectLegalBestmove(responder, root);
    const RootStats stats = engine->GetRootStats(root.key);
    ASSERT_TRUE(stats.materialized);
    EXPECT_EQ(stats.visits, static_cast<uint64_t>(kNodes));
    const RootStats parent_stats = engine->GetRootStats(parent_key);
    const RootStats shared_stats = engine->GetRootStats(shared_key);
    ASSERT_TRUE(parent_stats.materialized);
    ASSERT_TRUE(shared_stats.materialized);
    // In a tree a child has at most 1 + parent.best_edge_visits visits; here
    // `shared` also receives visits via 1. Kh2 Kb8 2. Kg2.
    EXPECT_GT(shared_stats.visits, 1 + parent_stats.best_edge_visits);
  }
}

TEST(Lc5SearchTest, StickyEndgamesPlaysMateAndStopsEarly) {
  // Mate in one: d1d8.
  const auto root = Origin("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1");
  const Move mate = Move::White(Square(kFileD, kRank1), Square(kFileD, kRank8));
  for (const bool sticky : {false, true}) {
    SCOPED_TRACE(sticky);
    auto engine = MakeTestEngine("puct", sticky);
    ControlledBackend backend;
    RecordingResponder responder;
    // Without the option the search runs to the limit, so keep it small.
    const int nodes = sticky ? 1'000'000 : 500;
    SearchSession session(engine.get(), &backend, &responder, TestSettings(),
                          TestTimeManagement(), root, GoParams{.nodes = nodes},
                          std::chrono::steady_clock::now());
    session.Start();
    session.Wait();
    ExpectDrained(session);
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

TEST(Lc5SearchTest, TwoFoldDrawBacksUpToRoot) {
  // Path: 4 materialized nodes, then a skipped leaf with update_node = false.
  constexpr size_t kNodes = 4;
  const VisitValue draw{.q = 0.0f, .d = 1.0f, .m = 0.0f};
  auto engine = MakeTestEngine();
  std::vector<VisitOrigin> origins;
  VisitOrigin cur = Origin();
  for (size_t i = 0; i < kNodes; ++i) {
    const Move move = cur.history.Last().GetBoard().GenerateLegalMoves()[0];
    engine->CompleteMaterialization(cur.key,
                                    {.leaf = {.q = 0.5f, .d = 0.0f, .m = 10.0f},
                                     .moves = {move},
                                     .priors = {1.0f}});
    origins.push_back(cur);
    cur.history.Append(move);
    cur.key = MakeNodeKey(cur.history, 7);
  }
  Metrics metrics;
  Visit visit;
  visit.origin = origins.front();
  visit.history = origins.front().history;
  visit.current_key = origins.front().key;
  ASSERT_EQ(engine->DescendVisit(visit, metrics),
            VisitEngine::DescendResult::kUnmaterializedLeaf);
  ASSERT_EQ(visit.path.size(), kNodes + 1);
  EXPECT_FALSE(visit.path.back().update_node);
  visit.result = draw;
  engine->BackupVisit(visit, metrics);

  for (size_t i = 0; i < kNodes; ++i) {
    SCOPED_TRACE(i);
    const RootStats stats = engine->GetRootStats(origins[i].key);
    ASSERT_TRUE(stats.materialized);
    EXPECT_EQ(stats.visits, 2u);
    EXPECT_EQ(stats.best_edge_visits, 1u);
    ThinkingInfo info;
    engine->FillThinkingInfo(origins[i], info);
    ASSERT_TRUE(info.score.has_value());
    EXPECT_EQ(*info.score, CentipawnsFromQ(0.25f));
    ASSERT_TRUE(info.wdl.has_value());
    EXPECT_EQ(info.wdl->d, 500);
  }
  InstallStore<RecordingStore>(*engine);
  EXPECT_EQ(engine->EvictToStore(kNodes, NodeKey{~0ull}, metrics), kNodes);
}

TEST(Lc5SearchTest, TwoFoldDrawsFromRealCyclesDrain) {
  // Kings and blocked pawns: few moves, so king shuffles repeat early.
  const auto root = Origin("k7/p7/P7/8/8/8/8/7K w - - 0 1");
  for (const int threads : {1, 2}) {
    SCOPED_TRACE(threads);
    auto engine = MakeTestEngine();
    ControlledBackend backend;
    RecordingResponder responder;
    constexpr int64_t kNodes = 2000;
    SearchSession session(engine.get(), &backend, &responder,
                          TestSettings(threads, 16), TestTimeManagement(), root,
                          GoParams{.nodes = kNodes},
                          std::chrono::steady_clock::now());
    session.Start();
    session.Wait();
    ExpectDrained(session);
    ExpectFinalOutput(responder, kNodes);
    EXPECT_GT(session.metrics().two_fold_visits.load(), 0u);
    const RootStats stats = engine->GetRootStats(root.key);
    ASSERT_TRUE(stats.materialized);
    EXPECT_EQ(stats.visits, static_cast<uint64_t>(kNodes));
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
    // Without a store the owner finds the terminal right away, so later visits
    // see it materialized. A store lookup comes first, so they wait on the
    // materialization.
    for (const bool with_store : {false, true}) {
      SCOPED_TRACE(test.fen);
      SCOPED_TRACE(with_store);
      auto engine = MakeTestEngine();
      if (with_store) InstallStore<MemoryNodeStore>(*engine);
      ControlledBackend backend;
      RecordingResponder responder;
      const auto root = Origin(test.fen);
      SearchSession session(engine.get(), &backend, &responder, TestSettings(),
                            TestTimeManagement(), root, GoParams{.nodes = 4},
                            std::chrono::steady_clock::now());
      for (int i = 0; i < 4; ++i)
        ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, root));
      session.Start();
      session.Wait();
      ExpectDrained(session);
      ExpectFinalOutput(responder, 4);
      EXPECT_EQ(responder.bestmoves[0].bestmove,
                Move::WhitePromotion(kSquareH1, kSquareH1, kQueen));
      EXPECT_EQ(backend.inputs.load(), 0u);
      EXPECT_EQ(backend.computes.load(), 0u);
      EXPECT_EQ(session.metrics().materializations_started.load(), 1u);
      EXPECT_EQ(session.metrics().materialization_waiters.load(),
                with_store ? 3u : 0u);
      EXPECT_EQ(session.metrics().terminal_visits.load(), 4u);
      EXPECT_EQ(session.metrics().visits_resumed.load(), 0u);
      const RootStats stats = engine->GetRootStats(root.key);
      ASSERT_TRUE(stats.materialized);
      EXPECT_EQ(stats.edges, 0u);
      EXPECT_EQ(stats.visits, 4u);
      const auto& final = responder.thinking.back();
      ASSERT_TRUE(final.score.has_value());
      EXPECT_EQ(*final.score,
                test.kind == TerminalKind::kCheckmate ? -12780 : 0);
      ASSERT_TRUE(final.wdl.has_value());
      EXPECT_EQ(final.wdl->w, 0);
      EXPECT_EQ(final.wdl->d, test.kind == TerminalKind::kStalemate ? 1000 : 0);
      EXPECT_EQ(final.wdl->l, test.kind == TerminalKind::kCheckmate ? 1000 : 0);
    }
  }
}

TEST(Lc5SearchTest, ImmediateAndMixedResultsKeepJobStorageAlive) {
  for (const auto cache : {ControlledBackend::Cache::kAll,
                           ControlledBackend::Cache::kAlternating}) {
    SCOPED_TRACE(static_cast<int>(cache));
    auto engine = MakeTestEngine();
    ControlledBackend backend(cache);
    RecordingResponder responder;
    const auto root = Origin();
    const auto moves = root.history.Last().GetBoard().GenerateLegalMoves();
    engine->CompleteMaterialization(
        root.key, {.moves = {moves[0], moves[1]}, .priors = {0.5f, 0.5f}});
    SearchSession session(engine.get(), &backend, &responder, TestSettings(),
                          TestTimeManagement(), root, GoParams{.nodes = 4},
                          std::chrono::steady_clock::now());
    std::vector<VisitOrigin> origins;
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
    ExpectDrained(session);
    ExpectFinalOutput(responder, 5);
    const bool all = cache == ControlledBackend::Cache::kAll;
    EXPECT_EQ(backend.inputs.load(), 4u);
    EXPECT_EQ(backend.computes.load(), all ? 0u : 1u);
    EXPECT_EQ(session.metrics().cache_hits.load(), all ? 4u : 2u);
    EXPECT_EQ(session.metrics().nn_evaluations.load(), all ? 0u : 2u);
    for (const auto& origin : origins) {
      const RootStats stats = engine->GetRootStats(origin.key);
      ASSERT_TRUE(stats.materialized);
      EXPECT_EQ(stats.visits, 1u);
      EXPECT_EQ(stats.edges,
                origin.history.Last().GetBoard().GenerateLegalMoves().size());
      ThinkingInfo info;
      engine->FillThinkingInfo(origin, info);
      ASSERT_TRUE(info.score.has_value());
      EXPECT_EQ(*info.score, CentipawnsFromQ(0.25f));
      ASSERT_TRUE(info.wdl.has_value());
      EXPECT_EQ(info.wdl->d, 500);
    }
  }
}

TEST(Lc5SearchTest, BlockedBackendStopAndAbortDrainReservations) {
  for (const bool abort : {false, true}) {
    SCOPED_TRACE(abort);
    auto engine = MakeTestEngine();
    auto& store = InstallStore<RecordingStore>(*engine);
    ControlledBackend backend(ControlledBackend::Cache::kNone, true);
    RecordingResponder responder;
    const auto root = Origin();
    const auto moves = root.history.Last().GetBoard().GenerateLegalMoves();
    // All visits reserve the same root edge and share the child
    // materialization.
    engine->CompleteMaterialization(root.key,
                                    {.moves = {moves[0]}, .priors = {1.0f}});
    auto child = root;
    child.history.Append(moves[0]);
    child.key = MakeNodeKey(child.history, 7);
    SearchSession session(engine.get(), &backend, &responder, TestSettings(),
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
    EXPECT_EQ(session.metrics().materialization_waiters.load(), 3u);
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
    ExpectDrained(session);
    // The evaluation was authorized before stop, so the owner backs up when
    // it returns. The waiters would select below the child and cancel.
    EXPECT_EQ(session.metrics().visits_admitted.load(), 4u);
    EXPECT_EQ(session.metrics().visits_completed.load(), 1u);
    EXPECT_EQ(session.metrics().visits_cancelled.load(), 3u);
    EXPECT_EQ(session.metrics().node_store_misses.load(), 1u);
    // Evaluations are not stored; only evicted nodes are.
    EXPECT_EQ(store.stored, 0u);
    // The setup install plus the owner's late backup.
    const RootStats root_stats = engine->GetRootStats(root.key);
    EXPECT_EQ(root_stats.visits, 2u);
    EXPECT_EQ(root_stats.best_edge_visits, 1u);
    if (abort) {
      EXPECT_TRUE(responder.bestmoves.empty());
      EXPECT_TRUE(
          absl::c_none_of(responder.thinking, [](const ThinkingInfo& info) {
            return info.comment.ends_with(" final");
          }));
    } else {
      ExpectFinalOutput(responder, 1);
    }
    // Every reservation was released, so both nodes can be evicted.
    Metrics evict_metrics;
    EXPECT_EQ(engine->EvictToStore(engine->GetGraphSize(), NodeKey{~0ull},
                                   evict_metrics),
              2u);
  }
}

TEST(Lc5SearchTest, SavedGraphRestoresWithStatisticsInFreshGraph) {
  ControlledBackend backend;
  const auto root = Origin();
  auto saved_engine = MakeTestEngine();
  auto& saved_store = InstallStore<RecordingStore>(*saved_engine);
  {
    RecordingResponder responder;
    SearchSession session(saved_engine.get(), &backend, &responder,
                          TestSettings(), TestTimeManagement(), root,
                          GoParams{.nodes = 3},
                          std::chrono::steady_clock::now());
    session.Start();
    session.Wait();
    ExpectDrained(session);
    EXPECT_EQ(saved_store.stored, 0u);
  }
  const RootStats saved = saved_engine->GetRootStats(root.key);
  ASSERT_EQ(saved.visits, 3u);
  const size_t saved_size = saved_engine->GetGraphSize();
  Metrics evict_metrics;
  ASSERT_EQ(
      saved_engine->EvictToStore(saved_size, NodeKey{~0ull}, evict_metrics),
      saved_size);

  // A fresh graph restores the root with its statistics instead of
  // evaluating it, and the new visit goes below it.
  auto engine = MakeTestEngine();
  auto& store = InstallStore<RecordingStore>(*engine);
  store.entries = saved_store.entries;
  EXPECT_EQ(store.entries.size(), saved_size);
  const size_t evaluated = backend.inputs.load();

  RecordingResponder responder;
  SearchSession session(engine.get(), &backend, &responder, TestSettings(),
                        TestTimeManagement(), root, GoParams{.nodes = 1},
                        std::chrono::steady_clock::now());
  session.Start();
  session.Wait();
  ExpectDrained(session);
  ExpectFinalOutput(responder, 1);
  EXPECT_GE(session.metrics().node_store_hits.load(), 1u);
  EXPECT_GE(session.metrics().nodes_restored.load(), 1u);
  EXPECT_EQ(session.metrics().restores_rejected.load(), 0u);
  const RootStats stats = engine->GetRootStats(root.key);
  ASSERT_TRUE(stats.materialized);
  EXPECT_EQ(stats.visits, saved.visits + 1);
  EXPECT_EQ(stats.edges, saved.edges);
  // Saved nodes are restored too; the visit ends at the first node that was
  // never materialized, which is the only new evaluation.
  EXPECT_EQ(backend.inputs.load(), evaluated + 1);
}

TEST(Lc5SearchTest, RootBootstrapFlushesDependencyWithoutWaitingForTimer) {
  // A positive one-minute delay cannot be the reason this bootstrap flushes.
  for (const int delay : {0, 60000}) {
    SCOPED_TRACE(delay);
    auto engine = MakeTestEngine();
    ControlledBackend backend(ControlledBackend::Cache::kNone, true);
    RecordingResponder responder;
    const auto root = Origin();
    auto settings = TestSettings(4);
    settings.max_batch_delay_ms = delay;
    SearchSession session(engine.get(), &backend, &responder, settings,
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
    ExpectDrained(session);
    ExpectFinalOutput(responder, 1);
    EXPECT_EQ(backend.batch_sizes, (std::vector<size_t>{1}));
  }
}

TEST(Lc5SearchTest, CollectionFlushesWhileAnotherWorkerIsBlockedInStore) {
  for (const int delay : {0, 1}) {
    SCOPED_TRACE(delay);
    auto engine = MakeTestEngine();
    // First load misses; the other worker's load remains executing. With a
    // positive delay it prevents dependency-starvation detection, so the
    // collector must actually time out. Zero must flush without that timer.
    auto& store = InstallStore<BlockingStore>(*engine, 2, false);
    ControlledBackend backend(ControlledBackend::Cache::kNone, true);
    RecordingResponder responder;
    const auto root = Origin();
    const auto moves = root.history.Last().GetBoard().GenerateLegalMoves();
    engine->CompleteMaterialization(
        root.key, {.moves = {moves[0], moves[1]}, .priors = {0.5f, 0.5f}});
    auto first = root;
    first.history.Append(moves[0]);
    first.key = MakeNodeKey(first.history, 7);
    auto second = root;
    second.history.Append(moves[1]);
    second.key = MakeNodeKey(second.history, 7);
    auto settings = TestSettings(2, 2);
    settings.max_batch_delay_ms = delay;
    SearchSession session(engine.get(), &backend, &responder, settings,
                          TestTimeManagement(), root, GoParams{.nodes = 2},
                          std::chrono::steady_clock::now());
    ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, first));
    ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, second));
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
    ExpectDrained(session);
    ExpectFinalOutput(responder, 1);
    // The computed evaluation backs up; the other store miss is cancelled.
    EXPECT_EQ(session.metrics().visits_completed.load(), 1u);
    EXPECT_EQ(session.metrics().visits_cancelled.load(), 1u);
    EXPECT_EQ(session.metrics().node_store_misses.load(), 2u);
    EXPECT_EQ(store.recording.stored, 0u);
    EXPECT_EQ(backend.computes.load(), 1u);
  }
}

TEST(Lc5SearchTest, PositiveDelayCollectsAnotherWorkersStoreMissIntoSameBatch) {
  auto engine = MakeTestEngine();
  auto& store = InstallStore<BlockingStore>(*engine, 2, false);
  ControlledBackend backend(ControlledBackend::Cache::kNone, true);
  RecordingResponder responder;
  const auto root = Origin();
  auto other = root;
  other.history.Append(root.history.Last().GetBoard().GenerateLegalMoves()[0]);
  other.key = MakeNodeKey(other.history, 7);
  auto settings = TestSettings(2, 2);
  settings.minibatch_size = 2;
  settings.max_batch_delay_ms = 60000;
  SearchSession session(engine.get(), &backend, &responder, settings,
                        TestTimeManagement(), root, GoParams{.nodes = 2},
                        std::chrono::steady_clock::now());
  ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, root));
  ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, other));
  session.Start();
  const bool loading = store.load.WaitForArrival();
  const bool first_input = backend.WaitForInputs(1);
  // The first miss has entered AddInput, but the other worker cannot send
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
  ExpectDrained(session);
  ExpectFinalOutput(responder, 2);
  EXPECT_EQ(backend.batch_sizes, (std::vector<size_t>{2}));
  EXPECT_EQ(session.metrics().evaluation_batches.load(), 1u);
  EXPECT_EQ(session.metrics().partial_timeout_flushes.load(), 0u);
  EXPECT_EQ(session.metrics().partial_starvation_flushes.load(), 0u);
  EXPECT_EQ(store.recording.stored, 0u);
}

TEST(Lc5SearchTest, MultiworkerMixedCacheResultsRespectBatchAndItemLimits) {
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
      auto engine = MakeTestEngine();
      ControlledBackend backend(ControlledBackend::Cache::kAlternating, true,
                                test.batch_size, test.evaluators);
      RecordingResponder responder;
      const auto root = Origin();
      const auto moves = root.history.Last().GetBoard().GenerateLegalMoves();
      ASSERT_EQ(moves.size(), 20u);
      engine->CompleteMaterialization(
          root.key, {.moves = {moves[0], moves[1]}, .priors = {0.5f, 0.5f}});
      auto settings = TestSettings(4, test.capacity);
      settings.eval_threads = test.evaluators;
      settings.minibatch_size = test.batch_size;
      settings.max_batch_delay_ms = delay;
      SearchSession session(engine.get(), &backend, &responder, settings,
                            TestTimeManagement(), root,
                            GoParams{.nodes = test.capacity},
                            std::chrono::steady_clock::now());
      std::vector<VisitOrigin> origins;
      // Pre-admission gives every worker independent leaves before submission.
      // Target 4 stresses the in-flight limits with one-item jobs; targets
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
      ExpectDrained(session);
      ExpectFinalOutput(responder, 1 + test.capacity);
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
      EXPECT_EQ(engine->GetGraphSize(), 1u + test.capacity);
      const size_t job_size = (test.batch_size + 3) / 4;
      const size_t max_jobs =
          std::min<size_t>(16, 2 * test.batch_size / job_size);
      EXPECT_LE(session.metrics().io_jobs_high_water.load(), 4 * max_jobs);
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
        const RootStats stats = engine->GetRootStats(origin.key);
        ASSERT_TRUE(stats.materialized);
        EXPECT_EQ(stats.visits, 1u);
        EXPECT_EQ(stats.edges,
                  origin.history.Last().GetBoard().GenerateLegalMoves().size());
        ThinkingInfo info;
        engine->FillThinkingInfo(origin, info);
        ASSERT_TRUE(info.score.has_value());
        EXPECT_EQ(*info.score, CentipawnsFromQ(0.25f));
        ASSERT_TRUE(info.wdl.has_value());
        EXPECT_EQ(info.wdl->d, 500);
      }
    }
  }
}

TEST(Lc5SearchTest, BlockedStoreLoadsDrainAfterStopOrAbort) {
  for (const bool hit : {false, true}) {
    for (const bool abort : {false, true}) {
      SCOPED_TRACE(hit);
      SCOPED_TRACE(abort);
      auto engine = MakeTestEngine();
      auto& store = InstallStore<BlockingStore>(*engine);
      ControlledBackend backend(ControlledBackend::Cache::kAll);
      RecordingResponder responder;
      const auto root = Origin();
      const auto move = root.history.Last().GetBoard().GenerateLegalMoves()[0];
      engine->CompleteMaterialization(root.key,
                                      {.moves = {move}, .priors = {1.0f}});
      auto child = root;
      child.history.Append(move);
      child.key = MakeNodeKey(child.history, 7);
      if (hit) {
        auto donor = MakeTestEngine();
        auto& donor_store = InstallStore<RecordingStore>(*donor);
        auto legal = child.history.Last().GetBoard().GenerateLegalMoves();
        absl::c_sort(
            legal, [](Move a, Move b) { return a.raw_data() < b.raw_data(); });
        donor->CompleteMaterialization(
            child.key,
            {.leaf = {.q = 0.75f, .d = 0.25f, .m = 7.0f},
             .moves = {legal.begin(), legal.end()},
             .priors = std::vector<float>(legal.size(), 1.0f / legal.size())});
        Metrics evict_metrics;
        ASSERT_EQ(donor->EvictToStore(1, root.key, evict_metrics), 1u);
        store.recording.entries = donor_store.entries;
      }
      SearchSession session(engine.get(), &backend, &responder, TestSettings(),
                            TestTimeManagement(), root,
                            GoParams{.nodes = 4, .infinite = true},
                            std::chrono::steady_clock::now());
      for (int i = 0; i < 4; ++i)
        ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, root));
      session.Start();
      const bool loading = store.load.WaitForArrival();
      if (loading) {
        EXPECT_EQ(session.metrics().materialization_waiters.load(), 3u);
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
      ExpectDrained(session);
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
      EXPECT_EQ(engine->ContainsNode(child.key), hit);
      if (hit) {
        EXPECT_EQ(engine->GetRootStats(child.key).visits, 1u);
        ThinkingInfo info;
        engine->FillThinkingInfo(child, info);
        ASSERT_TRUE(info.score.has_value());
        EXPECT_EQ(*info.score, CentipawnsFromQ(0.75f));
      }
      // Only the setup install.
      const RootStats root_stats = engine->GetRootStats(root.key);
      EXPECT_EQ(root_stats.visits, 1u);
      EXPECT_EQ(root_stats.best_edge_visits, 0u);
      if (abort) {
        EXPECT_TRUE(responder.bestmoves.empty());
        EXPECT_TRUE(
            absl::c_none_of(responder.thinking, [](const ThinkingInfo& info) {
              return info.comment.ends_with(" final");
            }));
      } else {
        ExpectFinalOutput(responder, 1);
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
    auto engine = MakeTestEngine();
    ControlledBackend backend(ControlledBackend::Cache::kNone, true, 4,
                              evaluators);
    RecordingResponder responder;
    const auto root = Origin();
    const auto moves = root.history.Last().GetBoard().GenerateLegalMoves();
    engine->CompleteMaterialization(root.key,
                                    {.moves = {moves[0]}, .priors = {1.0f}});
    auto settings = TestSettings(4, 20);
    settings.eval_threads = evaluators;
    SearchSession session(engine.get(), &backend, &responder, settings,
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
    EXPECT_EQ(engine->GetGraphSize(), 1u + 4u * evaluators);
    // Authorized batches back up their visits; the queued backlog cancels.
    EXPECT_EQ(session.metrics().visits_completed.load(), 4u * evaluators);
    ExpectDrained(session);
    ExpectLegalBestmove(responder, root);
  }
}

TEST(Lc5SearchTest, AuthorizedEvaluationsBackUpToRootAfterStop) {
  auto engine = MakeTestEngine();
  ControlledBackend backend(ControlledBackend::Cache::kNone, true);
  RecordingResponder responder;
  const auto root = Origin();
  // Uniform priors, so edge order is by move encoding.
  auto legal = root.history.Last().GetBoard().GenerateLegalMoves();
  absl::c_sort(legal,
               [](Move a, Move b) { return a.raw_data() < b.raw_data(); });
  engine->CompleteMaterialization(
      root.key,
      {.moves = {legal.begin(), legal.end()},
       .priors = std::vector<float>(legal.size(), 1.0f / legal.size())});
  SearchSession session(engine.get(), &backend, &responder, TestSettings(4, 20),
                        TestTimeManagement(), root, GoParams{.nodes = 20},
                        std::chrono::steady_clock::now());
  for (int i = 0; i < 20; ++i)
    ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, root));
  session.Start();
  // One authorized batch of four children is computing; the other visits wait
  // on materializations of any worker, or are still queued, and must cancel.
  const bool computing = backend.WaitForCompute();
  session.Stop();
  const bool responding = responder.WaitForBestmove();
  EXPECT_FALSE(session.IsFinished());
  backend.Release();
  session.Wait();
  ASSERT_TRUE(computing);
  ASSERT_TRUE(responding);
  ExpectDrained(session);
  EXPECT_EQ(backend.computes.load(), 1u);
  EXPECT_EQ(backend.inputs.load(), 4u);
  // Each computed materialization's owner backs up through the root after stop.
  EXPECT_EQ(session.metrics().visits_completed.load(), 4u);
  const RootStats stats = engine->GetRootStats(root.key);
  ASSERT_TRUE(stats.materialized);
  EXPECT_EQ(stats.visits, 5u);
  EXPECT_GE(stats.best_edge_visits, 1u);
  ExpectFinalOutput(responder, 21);
}

TEST(Lc5SearchTest, StopDuringAddInputKeepsOnlyImmediateResults) {
  auto engine = MakeTestEngine();
  ControlledBackend backend(ControlledBackend::Cache::kAlternating, true, 8);
  backend.blocked_input = 2;
  RecordingResponder responder;
  const auto root = Origin();
  const auto moves = root.history.Last().GetBoard().GenerateLegalMoves();
  engine->CompleteMaterialization(root.key,
                                  {.moves = {moves[0]}, .priors = {1.0f}});
  auto settings = TestSettings(1, 4);
  settings.minibatch_size = 8;
  SearchSession session(engine.get(), &backend, &responder, settings,
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
  EXPECT_EQ(engine->GetGraphSize(), 2u);
  const RootStats hit = engine->GetRootStats(origins[0].key);
  ASSERT_TRUE(hit.materialized);
  // The install counted the leaf; its owner completes with nothing above it.
  EXPECT_EQ(hit.visits, 1u);
  EXPECT_EQ(session.metrics().visits_completed.load(), 1u);
  EXPECT_EQ(session.metrics().visits_cancelled.load(), 3u);
  ThinkingInfo info;
  engine->FillThinkingInfo(origins[0], info);
  ASSERT_TRUE(info.score.has_value());
  EXPECT_EQ(*info.score, CentipawnsFromQ(backend.q));
  for (size_t i = 1; i < origins.size(); ++i)
    EXPECT_FALSE(engine->ContainsNode(origins[i].key));
  ExpectDrained(session);
}

TEST(Lc5SearchTest, NonfittingJobStaysQueuedAndIsCancelledAfterStop) {
  auto engine = MakeTestEngine();
  ControlledBackend backend(ControlledBackend::Cache::kNone, true, 5);
  RecordingResponder responder;
  const auto root = Origin();
  const auto moves = root.history.Last().GetBoard().GenerateLegalMoves();
  engine->CompleteMaterialization(root.key,
                                  {.moves = {moves[0]}, .priors = {1.0f}});
  auto settings = TestSettings(1, 6);
  settings.minibatch_size = 5;
  SearchSession session(engine.get(), &backend, &responder, settings,
                        TestTimeManagement(), root, GoParams{.nodes = 6},
                        std::chrono::steady_clock::now());
  const auto origins = AdmitChildren(session, root, 6);
  session.Start();
  // Three whole jobs of two inputs: after adding four inputs, the collector
  // leaves the last job queued because 4 + 2 exceeds the target 5.
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
  EXPECT_EQ(engine->GetGraphSize(), 5u);
  for (size_t i = 4; i < origins.size(); ++i)
    EXPECT_FALSE(engine->ContainsNode(origins[i].key));
  ExpectDrained(session);
}

TEST(Lc5SearchTest, AbortOverridesRepeatedStopBeforeCommitment) {
  auto engine = MakeTestEngine();
  ControlledBackend backend;
  RecordingResponder responder;
  const auto root = Origin();
  SearchSession session(engine.get(), &backend, &responder, TestSettings(),
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
  ExpectDrained(session);
  EXPECT_TRUE(responder.bestmoves.empty());
  EXPECT_TRUE(responder.thinking.empty());
  EXPECT_EQ(backend.inputs.load(), 0u);
  EXPECT_EQ(backend.computes.load(), 0u);
  EXPECT_EQ(engine->GetGraphSize(), 0u);
}

TEST(Lc5SearchTest, BlockedEvictionDoesNotDelayPartialBatch) {
  auto engine = MakeTestEngine();
  auto& store = InstallStore<BlockingStore>(*engine);
  ControlledBackend backend(ControlledBackend::Cache::kNone, true);
  RecordingResponder responder;
  const auto root = Origin();
  const auto legal = root.history.Last().GetBoard().GenerateLegalMoves();
  // As in RespondsBeforeBlockedEvictionDrains: the sibling is evicted while
  // the one visit evaluates the first edge's child.
  engine->CompleteMaterialization(
      root.key, {.moves = {legal[0], legal[1]}, .priors = {0.9f, 0.1f}});
  auto sibling = root;
  sibling.history.Append(legal[1]);
  sibling.key = MakeNodeKey(sibling.history, 7);
  engine->CompleteMaterialization(
      sibling.key,
      {.leaf = {.q = 0.5f, .d = 0.0f, .m = 0.0f}, .moves = {}, .priors = {}});
  auto settings = TestSettings(1, 1);
  settings.graph_node_limit = 1;
  settings.eval_threads = 2;
  // The one-item batch can only flush on starvation, not at the deadline.
  settings.max_batch_delay_ms = 60'000;
  SearchSession session(engine.get(), &backend, &responder, settings,
                        TestTimeManagement(), root, GoParams{.nodes = 1},
                        std::chrono::steady_clock::now());
  session.Start();
  const bool persisting = store.persist.WaitForArrival();
  store.load.Release();
  const bool computing = backend.WaitForCompute();
  // Stop flushes the batch even if starvation was missed.
  session.Stop();
  store.Release();
  backend.Release();
  session.Wait();
  ASSERT_TRUE(persisting);
  EXPECT_TRUE(computing);
  ExpectDrained(session);
  EXPECT_EQ(session.metrics().partial_starvation_flushes.load(), 1u);
  EXPECT_EQ(session.metrics().partial_timeout_flushes.load(), 0u);
}

TEST(Lc5SearchTest, RespondsBeforeBlockedEvictionDrains) {
  auto engine = MakeTestEngine();
  auto& store = InstallStore<BlockingStore>(*engine, 0);
  ControlledBackend backend(ControlledBackend::Cache::kNone, true);
  RecordingResponder responder;
  const auto root = Origin();
  const auto legal = root.history.Last().GetBoard().GenerateLegalMoves();
  // The one visit takes the first edge and blocks in the backend; the
  // sibling below the second edge is the node to evict.
  engine->CompleteMaterialization(
      root.key, {.moves = {legal[0], legal[1]}, .priors = {0.9f, 0.1f}});
  auto sibling = root;
  sibling.history.Append(legal[1]);
  sibling.key = MakeNodeKey(sibling.history, 7);
  engine->CompleteMaterialization(
      sibling.key,
      {.leaf = {.q = 0.5f, .d = 0.0f, .m = 0.0f}, .moves = {}, .priors = {}});
  auto settings = TestSettings(1, 1);
  settings.graph_node_limit = 1;
  // Store threads come one per eval thread. With only one, the visit's store
  // load could queue behind the blocked eviction and never reach the backend.
  settings.eval_threads = 2;
  SearchSession session(engine.get(), &backend, &responder, settings,
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
  ExpectDrained(session);
  ExpectFinalOutput(responder, 1);
  // Stored before it was erased.
  EXPECT_EQ(session.metrics().nodes_evicted.load(), 1u);
  EXPECT_TRUE(store.recording.entries.contains(sibling.key));
  EXPECT_FALSE(engine->ContainsNode(sibling.key));
  EXPECT_TRUE(engine->ContainsNode(root.key));
}

TEST(Lc5SearchTest, EvictionKeepsGraphSmallAndLosesNoVisits) {
  for (const int threads : {1, 3}) {
    SCOPED_TRACE(threads);
    auto engine = MakeTestEngine();
    auto& store = InstallStore<MemoryNodeStore>(*engine);
    ControlledBackend backend;
    RecordingResponder responder;
    const auto root = Origin();
    auto settings = TestSettings(threads, 8);
    settings.graph_node_limit = 32;
    SearchSession session(engine.get(), &backend, &responder, settings,
                          TestTimeManagement(), root, GoParams{.nodes = 3000},
                          std::chrono::steady_clock::now());
    session.Start();
    session.Wait();
    ExpectDrained(session);
    ExpectFinalOutput(responder, 3000);
    const auto& m = session.metrics();
    EXPECT_EQ(m.visits_completed.load(), 3000u);
    EXPECT_GT(m.nodes_evicted.load(), 0u);
    EXPECT_GT(m.nodes_restored.load(), 0u);
    EXPECT_EQ(m.restores_rejected.load(), 0u);
    // Every completed visit reached the root, which is never evicted.
    const RootStats stats = engine->GetRootStats(root.key);
    ASSERT_TRUE(stats.materialized);
    EXPECT_EQ(stats.visits, m.visits_completed.load());
    EXPECT_GT(stats.best_edge_visits, 0u);
    // Far fewer nodes than visits stay in RAM; every evicted one is stored.
    EXPECT_LT(engine->GetGraphSize(), 3000u / 4);
    EXPECT_GE(store.Size(), m.nodes_evicted.load() - m.nodes_restored.load());
  }
}

TEST(Lc5SearchTest, SingleNodeSearchPlaysTopPriorMove) {
  auto engine = MakeTestEngine();
  ControlledBackend backend;
  RecordingResponder responder;
  // Black to move, so the PV move is flipped back for UCI.
  const auto root =
      Origin("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
  SearchSession session(engine.get(), &backend, &responder, TestSettings(),
                        TestTimeManagement(), root, GoParams{.nodes = 1},
                        std::chrono::steady_clock::now());
  session.Start();
  session.Wait();
  ExpectDrained(session);
  ExpectFinalOutput(responder, 1);
  const RootStats stats = engine->GetRootStats(root.key);
  ASSERT_TRUE(stats.materialized);
  ASSERT_GT(stats.edges, 0u);
  EXPECT_EQ(stats.best_edge_visits, 0u);
  Move expected = stats.best_move;
  expected.Flip();
  ASSERT_EQ(responder.thinking.back().pv.size(), 1u);
  EXPECT_EQ(responder.thinking.back().pv[0], expected);
  EXPECT_EQ(responder.bestmoves[0].bestmove, expected);
}

TEST(Lc5SearchTest, FreshRootStoppedBeforeStartStillEvaluatesRootOnce) {
  auto engine = MakeTestEngine();
  ControlledBackend backend;
  RecordingResponder responder;
  const auto root = Origin();
  SearchSession session(engine.get(), &backend, &responder, TestSettings(),
                        TestTimeManagement(), root, GoParams{},
                        std::chrono::steady_clock::now());
  ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, root));
  session.Stop();
  session.Start();
  session.Wait();
  ExpectDrained(session);
  ExpectFinalOutput(responder, 1);
  ExpectLegalBestmove(responder, root);
  EXPECT_EQ(backend.inputs.load(), 1u);
  EXPECT_EQ(backend.computes.load(), 1u);
  EXPECT_EQ(engine->GetGraphSize(), 1u);
  const RootStats stats = engine->GetRootStats(root.key);
  ASSERT_TRUE(stats.materialized);
  ASSERT_GT(stats.edges, 0u);
  EXPECT_EQ(responder.bestmoves[0].bestmove, stats.best_move);
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
    EXPECT_EQ(responder.thinking.back().nodes, i == 1 ? 2 : 1);
    EXPECT_NE(responder.thinking.back().comment.find(" active="),
              std::string::npos);
    EXPECT_TRUE(responder.thinking.back().comment.ends_with(" final"));
    SearchSessionTestPeer::ExpectDrained(engine);
  }
  // First run evaluates the root; retained search evaluates its child instead
  // of evaluating the root again. NewGame resets that retained graph.
  EXPECT_EQ(backend.history_sizes, (std::vector<size_t>{1, 2, 1}));
}

TEST(Lc5SearchTest, ThinkingInfoEmittedOnDepthSeldepthAndBestMoveChanges) {
  auto engine = MakeTestEngine();
  ControlledBackend backend(ControlledBackend::Cache::kAll);
  RecordingResponder responder;
  const auto root = Origin();
  const auto legal = root.history.Last().GetBoard().GenerateLegalMoves();
  const auto [m0, m1] = std::minmax(legal[0], legal[1], [](Move a, Move b) {
    return a.raw_data() < b.raw_data();
  });
  // Equal priors break ties to m0 first, and keep m1 ahead once m0 gets one bad
  // evaluation instead of m0's higher prior reclaiming the lead before the
  // 10ms controller poll.
  engine->CompleteMaterialization(root.key,
                                  {.moves = {m0, m1}, .priors = {0.5f, 0.5f}});

  // Pause at input 2 before Fill() so visit 1's backup is reported first, then
  // switch q so visit 2 flips the root's best move well before the 1s timer.
  backend.q = 0.8f;
  backend.blocked_input = 2;
  SearchSession session(engine.get(), &backend, &responder, TestSettings(1, 1),
                        TestTimeManagement(), root, GoParams{.infinite = true},
                        std::chrono::steady_clock::now());
  session.Start();

  // Visit 1 evaluates m0's child with +0.8 for opponent (-0.8 for root) and
  // backs up: depth=2, seldepth=2, bestmove=m0. Visit 2 blocks at input 2.
  ASSERT_TRUE(backend.input.WaitForArrival());
  ASSERT_TRUE(responder.WaitForThinking([&](const ThinkingInfo& info) {
    return !info.pv.empty() && info.pv.front() == m0 && info.depth == 2 &&
           info.seldepth == 2;
  }));

  // Visit 2 and later visits evaluate with 0.0 (better than m0's -0.8 at every
  // ply): bestmove switches from m0 to m1, and depth/seldepth increase.
  backend.q = 0.0f;
  backend.input.Release();
  ASSERT_TRUE(responder.WaitForThinking([&](const ThinkingInfo& info) {
    return !info.pv.empty() && info.pv.front() == m1 && info.depth > 2 &&
           info.seldepth > 2;
  }));

  session.Stop();
  session.Wait();
  ExpectDrained(session);
  ASSERT_GE(responder.thinking.size(), 3u);
  EXPECT_GE(responder.thinking[0].depth, 0);
  EXPECT_GE(responder.thinking[0].seldepth, 1);
  EXPECT_EQ(responder.thinking[0].pv.front(), m0);
  EXPECT_TRUE(absl::c_any_of(absl::MakeConstSpan(responder.thinking)
                                 .subspan(0, responder.thinking.size() - 1),
                             [&](const ThinkingInfo& info) {
                               return !info.pv.empty() && info.pv.front() == m1;
                             }));
  EXPECT_EQ(responder.bestmoves[0].bestmove, m1);
}

TEST(Lc5SearchTest, PrepareMaterializationUsesSynchronousBackendCache) {
  class SyncCachedBackend final : public Backend {
   public:
    BackendAttributes GetAttributes() const override {
      return {.has_mlh = true,
              .has_wdl = true,
              .runs_on_cpu = true,
              .suggested_num_search_threads = 1,
              .recommended_batch_size = 4,
              .maximum_batch_size = 4};
    }
    std::unique_ptr<BackendComputation> CreateComputation() override {
      ADD_FAILURE() << "Unexpected computation for cached positions";
      return nullptr;
    }
    std::optional<EvalResult> GetCachedEvaluation(
        const EvalPosition& pos) override {
      lookups.fetch_add(1);
      EvalResult result{.q = 0.25f, .d = 0.5f, .m = 3.0f, .p = {}};
      result.p.assign(pos.legal_moves.size(), 1.0f / pos.legal_moves.size());
      return result;
    }
    std::atomic<size_t> lookups{0};
  };

  auto engine = MakeTestEngine();
  SyncCachedBackend backend;
  RecordingResponder responder;
  const auto root = Origin();
  SearchSession session(engine.get(), &backend, &responder, TestSettings(2, 8),
                        TestTimeManagement(), root, GoParams{.nodes = 10},
                        std::chrono::steady_clock::now());
  session.Start();
  session.Wait();
  ExpectDrained(session);
  ExpectFinalOutput(responder, 10);
  EXPECT_GE(backend.lookups.load(), 10u);
  EXPECT_EQ(session.metrics().evaluation_batches.load(), 0u);
  EXPECT_EQ(session.metrics().nn_evaluations.load(), 0u);
  EXPECT_GE(session.metrics().cache_hits.load(), 10u);
}

TEST(Lc5SearchTest, EvaluatorReturnsPureCachedJobsWithoutWaitingForBatch) {
  class ScriptedCacheBackend final : public Backend {
   public:
    BackendAttributes GetAttributes() const override {
      return {.has_mlh = true,
              .has_wdl = true,
              .runs_on_cpu = true,
              .suggested_num_search_threads = 1,
              .recommended_batch_size = 4,
              .maximum_batch_size = 4};
    }
    std::unique_ptr<BackendComputation> CreateComputation() override {
      return std::make_unique<Computation>(*this);
    }
    bool WaitForMissInput() {
      std::unique_lock lock(mutex_);
      return cv_.wait_for(lock, std::chrono::seconds(5),
                          [&] { return miss_inputs_ >= 1; });
    }
    void ReleaseCompute() {
      std::lock_guard lock(mutex_);
      compute_released_ = true;
      cv_.notify_all();
    }
    std::atomic<size_t> computes{0};

   private:
    class Computation final : public BackendComputation {
     public:
      explicit Computation(ScriptedCacheBackend& backend) : backend_(backend) {}
      size_t UsedBatchSize() const override { return pending_.size(); }
      AddInputResult AddInput(const EvalPosition& pos,
                              EvalResultPtr result) override {
        // Only the very first position (root's first child) is a cache miss.
        if (backend_.inputs_.fetch_add(1) == 0) {
          {
            std::lock_guard lock(backend_.mutex_);
            ++backend_.miss_inputs_;
          }
          backend_.cv_.notify_all();
          pending_.push_back(result);
          return ENQUEUED_FOR_EVAL;
        }
        if (result.q) *result.q = 0.0f;
        if (result.d) *result.d = 0.5f;
        if (result.m) *result.m = 2.0f;
        std::fill(result.p.begin(), result.p.end(), 1.0f / pos.legal_moves.size());
        return FETCHED_IMMEDIATELY;
      }
      void ComputeBlocking() override {
        {
          std::unique_lock lock(backend_.mutex_);
          backend_.cv_.wait(lock, [&] { return backend_.compute_released_; });
        }
        backend_.computes.fetch_add(1);
        for (auto& result : pending_) {
          if (result.q) *result.q = 0.1f;
          if (result.d) *result.d = 0.5f;
          if (result.m) *result.m = 2.0f;
          std::fill(result.p.begin(), result.p.end(), 1.0f / result.p.size());
        }
      }

     private:
      ScriptedCacheBackend& backend_;
      std::vector<EvalResultPtr> pending_;
    };

    std::atomic<size_t> inputs_{0};
    std::mutex mutex_;
    std::condition_variable cv_;
    size_t miss_inputs_ = 0;
    bool compute_released_ = false;
  };

  auto engine = MakeTestEngine();
  ScriptedCacheBackend backend;
  RecordingResponder responder;
  const auto root = Origin();
  const auto moves = root.history.Last().GetBoard().GenerateLegalMoves();
  ASSERT_EQ(moves.size(), 20u);
  engine->CompleteMaterialization(
      root.key, {.moves = {moves[0], moves[1]}, .priors = {0.5f, 0.5f}});

  auto settings = TestSettings(2, 4);
  settings.minibatch_size = 4;
  settings.max_batch_delay_ms = 5000;
  SearchSession session(engine.get(), &backend, &responder, settings,
                        TestTimeManagement(), root, GoParams{.infinite = true},
                        std::chrono::steady_clock::now());
  session.Start();
  ASSERT_TRUE(backend.WaitForMissInput());
  // Wait for the second move's cached job to return and complete visits while
  // the miss batch is still blocked waiting for minibatch_size = 4.
  ASSERT_TRUE(responder.WaitForThinking([&](const ThinkingInfo&) {
    return session.metrics().visits_completed.load() >= 2 &&
           session.metrics().cache_hits.load() >= 1;
  }));
  EXPECT_EQ(backend.computes.load(), 0u);
  backend.ReleaseCompute();
  session.Stop();
  session.Wait();
  ExpectDrained(session);
}

TEST(Lc5SearchTest, IdleBackendFlushesHalfBatchForParallelEvaluators) {
  class StagedBackend final : public Backend {
   public:
    BackendAttributes GetAttributes() const override {
      return {.has_mlh = true,
              .has_wdl = true,
              .runs_on_cpu = false,
              .suggested_num_search_threads = 1,
              .recommended_batch_size = 4,
              .maximum_batch_size = 4};
    }
    std::unique_ptr<BackendComputation> CreateComputation() override {
      return std::make_unique<Computation>(*this);
    }
    bool WaitForInputs(size_t count) {
      std::unique_lock lock(mutex_);
      return cv_.wait_for(lock, std::chrono::seconds(5),
                          [&] { return inputs_ >= count || released_; }) &&
             inputs_ >= count;
    }
    bool WaitForComputeStarted(size_t count) {
      std::unique_lock lock(mutex_);
      return cv_.wait_for(
                 lock, std::chrono::seconds(5),
                 [&] { return computes_started_ >= count || released_; }) &&
             computes_started_ >= count;
    }
    size_t computes_started() {
      std::lock_guard lock(mutex_);
      return computes_started_;
    }
    void ReleaseBatch1() {
      {
        std::lock_guard lock(mutex_);
        release_batch1_ = true;
      }
      cv_.notify_all();
    }
    void ReleaseAll() {
      {
        std::lock_guard lock(mutex_);
        release_batch1_ = true;
        released_ = true;
      }
      cv_.notify_all();
    }
    std::vector<size_t> batch_sizes;

   private:
    class Computation final : public BackendComputation {
     public:
      explicit Computation(StagedBackend& backend) : backend_(backend) {}
      size_t UsedBatchSize() const override { return pending_.size(); }
      AddInputResult AddInput(const EvalPosition&,
                              EvalResultPtr result) override {
        pending_.push_back(result);
        {
          std::lock_guard lock(backend_.mutex_);
          ++backend_.inputs_;
        }
        backend_.cv_.notify_all();
        return ENQUEUED_FOR_EVAL;
      }
      void ComputeBlocking() override {
        size_t batch_index = 0;
        {
          std::unique_lock lock(backend_.mutex_);
          batch_index = ++backend_.computes_started_;
          backend_.batch_sizes.push_back(pending_.size());
          backend_.cv_.notify_all();
          if (batch_index == 1) {
            backend_.cv_.wait(lock, [&] { return backend_.release_batch1_; });
          } else {
            backend_.cv_.wait(lock, [&] { return backend_.released_; });
          }
        }
        for (const auto& result : pending_) {
          if (result.q) *result.q = 0.25f;
          if (result.d) *result.d = 0.5f;
          if (result.m) *result.m = 3.0f;
          std::fill(result.p.begin(), result.p.end(), 1.0f / result.p.size());
        }
      }

     private:
      StagedBackend& backend_;
      std::vector<EvalResultPtr> pending_;
    };

    std::mutex mutex_;
    std::condition_variable cv_;
    size_t inputs_ = 0;
    size_t computes_started_ = 0;
    bool release_batch1_ = false;
    bool released_ = false;
  };

  class StagedStore final : public NodeStore {
   public:
    explicit StagedStore(StagedBackend& backend) : backend_(backend) {}
    std::vector<std::optional<std::vector<uint8_t>>> LoadBatch(
        std::span<const NodeKey> keys) override {
      const size_t call = calls_.fetch_add(1);
      if (call == 1) {
        backend_.WaitForInputs(1);
      } else if (call == 2) {
        backend_.WaitForComputeStarted(1);
      } else if (call == 3) {
        backend_.WaitForInputs(3);
      } else if (call == 4) {
        load5.ArriveAndWait();
      }
      return std::vector<std::optional<std::vector<uint8_t>>>(keys.size());
    }
    void StoreBatch(std::span<const StoredNode>) override {}
    void EraseBatch(std::span<const NodeKey>) override {}
    void Clear() override {}
    OperationBarrier load5;

   private:
    StagedBackend& backend_;
    std::atomic<size_t> calls_{0};
  };

  auto engine = MakeTestEngine();
  StagedBackend backend;
  auto& store = InstallStore<StagedStore>(*engine, backend);
  RecordingResponder responder;
  const auto root = Origin();
  const auto moves = root.history.Last().GetBoard().GenerateLegalMoves();
  ASSERT_GE(moves.size(), 5u);
  engine->CompleteMaterialization(
      root.key, {.moves = {moves[0], moves[1], moves[2], moves[3], moves[4]},
                 .priors = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}});

  auto settings = TestSettings(1, 5);
  settings.eval_threads = 2;
  settings.minibatch_size = 4;
  settings.max_batch_delay_ms = 60'000;
  SearchSession session(engine.get(), &backend, &responder, settings,
                        TestTimeManagement(), root, GoParams{.nodes = 5},
                        std::chrono::steady_clock::now());
  for (size_t i = 0; i < 5; ++i) {
    auto child = root;
    child.history.Append(moves[i]);
    child.key = MakeNodeKey(child.history, 7);
    ASSERT_TRUE(SearchSessionTestPeer::AdmitVisit(session, child));
  }

  session.Start();
  // Batch 1 reaches half-batch (2) with idle backend and flushes immediately.
  const bool batch1 = backend.WaitForComputeStarted(1);
  // Batch 2 accumulates 2 inputs while Batch 1 is computing and load 5 blocks
  // starvation, so Batch 2 stays queued until Batch 1 completes.
  const bool inputs4 = backend.WaitForInputs(4);
  const bool load5_blocked = store.load5.WaitForArrival();
  const size_t computes_before_release = backend.computes_started();
  backend.ReleaseBatch1();
  const bool batch2 = backend.WaitForComputeStarted(2);

  session.Stop();
  store.load5.Release();
  backend.ReleaseAll();
  session.Wait();

  ASSERT_TRUE(batch1);
  ASSERT_TRUE(inputs4);
  ASSERT_TRUE(load5_blocked);
  EXPECT_EQ(computes_before_release, 1u);
  ASSERT_TRUE(batch2);
  ExpectDrained(session);
  EXPECT_EQ(backend.batch_sizes, (std::vector<size_t>{2, 2}));
  EXPECT_EQ(session.metrics().partial_timeout_flushes.load(), 0u);
}

TEST(Lc5SearchTest, ControllerUpdatesAndResetsRootRemainingVisits) {
  constexpr uint64_t kUnlimited = std::numeric_limits<uint64_t>::max();
  auto base_engine = MakeTestEngine();
  RecordingResponder responder;
  const auto root = Origin();
  const auto t0 = std::chrono::steady_clock::now();

  // Disabled for infinite, ponder, and SmartPruningFactor=0.
  {
    ControlledBackend backend;
    TimeManager::Config tm = TestTimeManagement();
    tm.alphazero_time_pct = 10.0f;
    tm.smart_pruning_factor = 1.5f;
    for (const GoParams& params : {GoParams{.wtime = 10200, .ponder = true},
                                   GoParams{.wtime = 10200, .infinite = true}}) {
      SearchSession session(base_engine.get(), &backend, &responder,
                            TestSettings(), tm, root, params, t0);
      EXPECT_EQ(SearchSessionTestPeer::ComputeRootRemainingVisits(
                    session, t0 + std::chrono::milliseconds(100)),
                kUnlimited);
    }

    tm.smart_pruning_factor = 0.0f;
    SearchSession no_prune(base_engine.get(), &backend, &responder,
                           TestSettings(), tm, root, GoParams{.nodes = 10}, t0);
    EXPECT_EQ(SearchSessionTestPeer::ComputeRootRemainingVisits(
                  no_prune, t0 + std::chrono::milliseconds(100)),
              kUnlimited);
  }

  // Forwarding engine captures every SetRootRemainingVisits update during a
  // live search and checks cleanup to kUnlimited on Wait().
  class RecordingEngine final : public VisitEngine {
   public:
    explicit RecordingEngine(std::unique_ptr<VisitEngine> inner)
        : inner_(std::move(inner)) {}
    std::string_view GetName() const override { return inner_->GetName(); }
    void Configure(const OptionsDict& options,
                   const BackendAttributes& backend) override {
      inner_->Configure(options, backend);
    }
    void Clear() override { inner_->Clear(); }
    DescendResult DescendVisit(Visit& visit, Metrics& metrics) override {
      return inner_->DescendVisit(visit, metrics);
    }
    void BackupVisit(const Visit& visit, Metrics& metrics) override {
      inner_->BackupVisit(visit, metrics);
    }
    void CancelVisit(const Visit& visit) override {
      inner_->CancelVisit(visit);
    }
    ResumeAction ResumeVisit(Visit& visit, Metrics& metrics) override {
      return inner_->ResumeVisit(visit, metrics);
    }
    void FillThinkingInfo(const VisitOrigin& origin,
                          ThinkingInfo& info) const override {
      inner_->FillThinkingInfo(origin, info);
    }
    RootStats GetRootStats(NodeKey root_key) const override {
      return inner_->GetRootStats(root_key);
    }
    void SetRootRemainingVisits(uint64_t remaining_visits) override {
      {
        std::lock_guard lock(mutex_);
        updates_.push_back(remaining_visits);
      }
      cv_.notify_all();
      inner_->SetRootRemainingVisits(remaining_visits);
    }
    bool ContainsNode(NodeKey key) const override {
      return inner_->ContainsNode(key);
    }
    bool CompleteMaterialization(NodeKey key, const NodeEval& eval) override {
      return inner_->CompleteMaterialization(key, eval);
    }
    size_t GetGraphSize() const override { return inner_->GetGraphSize(); }
    void SetStore(std::unique_ptr<NodeStore> store) override {
      inner_->SetStore(std::move(store));
    }
    bool HasStore() const override { return inner_->HasStore(); }
    std::vector<std::optional<std::vector<uint8_t>>> ReadFromStore(
        std::span<const NodeKey> keys) override {
      return inner_->ReadFromStore(keys);
    }
    bool RestoreFromRecord(NodeKey key,
                           std::span<const uint8_t> record) override {
      return inner_->RestoreFromRecord(key, record);
    }
    size_t EvictToStore(size_t max, NodeKey keep, Metrics& metrics) override {
      return inner_->EvictToStore(max, keep, metrics);
    }

    bool WaitForUpdateInRange(uint64_t min_inclusive, uint64_t max_exclusive) {
      std::unique_lock lock(mutex_);
      return cv_.wait_for(lock, std::chrono::seconds(5), [&] {
        return absl::c_any_of(updates_, [&](uint64_t v) {
          return v >= min_inclusive && v < max_exclusive;
        });
      });
    }
    std::vector<uint64_t> SnapshotUpdates() const {
      std::lock_guard lock(mutex_);
      return updates_;
    }

   private:
    std::unique_ptr<VisitEngine> inner_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<uint64_t> updates_;
  };

  RecordingEngine recording_engine(MakeTestEngine());
  {
    ControlledBackend backend;
    backend.blocked_input = 2;
    TimeManager::Config tm = TestTimeManagement();
    tm.move_overhead_ms = 200;
    tm.alphazero_time_pct = 10.0f;  // 5000ms budget for wtime=50200.
    tm.smart_pruning_factor = 1.0f;

    SearchSession session(&recording_engine, &backend, &responder,
                          TestSettings(1, 1), tm, root,
                          GoParams{.wtime = 50200, .nodes = 1000000},
                          std::chrono::steady_clock::now());
    session.Start();
    ASSERT_TRUE(backend.input.WaitForArrival());

    // Live controller thread pushes `node_remaining` during the 80ms warmup,
    // then transitions to the finite `EstimatedRemainingVisits` time bound.
    ASSERT_TRUE(recording_engine.WaitForUpdateInRange(999000u, kUnlimited));
    ASSERT_TRUE(recording_engine.WaitForUpdateInRange(0u, 10000u));

    session.Stop();
    backend.Release();
    session.Wait();
    ExpectDrained(session);
  }

  const auto updates = recording_engine.SnapshotUpdates();
  ASSERT_FALSE(updates.empty());
  EXPECT_EQ(updates.back(), kUnlimited);
}

}  // namespace
}  // namespace lczero::lc5

int main(int argc, char** argv) {
  lczero::InitializeMagicBitboards();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
