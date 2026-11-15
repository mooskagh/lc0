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
#include <absl/types/span.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "search/lc5/engine.h"
#include "search/lc5/policies/puct.h"
#include "search/lc5/policies/registry.h"
#include "search/lc5/session/search_session.h"
#include "search/lc5/store/memory_node_store.h"
#include "search/lc5/visit_engine.h"

namespace lczero::lc5 {
namespace {

Settings::Resolved TestSettings(int threads = 1, int capacity = 4) {
  return {.threads = threads,
          .eval_threads = 1,
          .minibatch_size = 4,
          .max_active_visits = capacity,
          .max_batch_delay_ms = 0,
          .active_visits_ratio = 0.0f,
          .graph_node_limit = 0};
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
  return {.move_overhead_ms = 200,
          .alphazero_time_pct = 3.0f,
          .smart_pruning_factor = 1.33f};
}

VisitOrigin Origin(std::string_view fen = ChessBoard::kStartposFen) {
  PositionHistory history;
  history.Reset(Position::FromFen(fen));
  return {.key = MakeNodeKey(history, 7), .history = std::move(history)};
}

// Pre-inserts `root` with `count` non-terminal legal moves at uniform prior.
// With FPU = 0 and positive backend leaf Q (negative at root), root visits fan
// out 1-to-1 across the returned child origins in exact edge order.
std::vector<VisitOrigin> InsertUniformRootChildren(VisitEngine& engine,
                                                   const VisitOrigin& root,
                                                   size_t count) {
  auto legal = root.history.Last().GetBoard().GenerateLegalMoves();
  absl::c_sort(legal,
               [](Move a, Move b) { return a.raw_data() < b.raw_data(); });
  std::vector<Move> moves;
  std::vector<VisitOrigin> children;
  moves.reserve(count);
  children.reserve(count);
  for (const Move move : legal) {
    if (moves.size() == count) break;
    VisitOrigin child = root;
    child.history.Append(move);
    if (child.history.Last().GetBoard().GenerateLegalMoves().empty()) continue;
    child.key = MakeNodeKey(child.history, 7);
    moves.push_back(move);
    children.push_back(std::move(child));
  }
  EXPECT_EQ(moves.size(), count);
  EXPECT_TRUE(engine.InsertNode(
      root.key,
      {.moves = moves,
       .priors = std::vector<float>(moves.size(), 1.0f / moves.size())}));
  return children;
}

uint64_t CommentMetric(const ThinkingInfo& info, std::string_view key) {
  const std::string needle = " " + std::string(key) + "=";
  const size_t pos = info.comment.find(needle);
  if (pos == std::string::npos) {
    ADD_FAILURE() << "Missing metric '" << key << "' in: " << info.comment;
    return 0;
  }
  const char* begin = info.comment.data() + pos + needle.size();
  const char* end = info.comment.data() + info.comment.size();
  uint64_t value = 0;
  const auto [ptr, ec] = std::from_chars(begin, end, value);
  EXPECT_EQ(ec, std::errc{}) << info.comment;
  return value;
}

// Runs session.Wait() on a helper thread so tests can verify that Wait() has
// not yet returned while backend or store operations remain in flight.
class SessionWaiter {
 public:
  explicit SessionWaiter(SearchSession& session)
      : thread_([this, &session] {
          session.Wait();
          done_.store(true);
        }) {}
  ~SessionWaiter() { Join(); }
  bool IsDone() const { return done_.load(); }
  void Join() {
    if (thread_.joinable()) thread_.join();
  }

 private:
  std::atomic<bool> done_{false};
  std::thread thread_;
};

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
    return cv_.wait_for(lock, timeout,
                        [&] { return absl::c_any_of(thinking, pred); });
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
    absl::c_fill(result.p, 1.0f / result.p.size());
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
      absl::Span<const NodeKey> keys) override {
    std::lock_guard lock(mutex_);
    loads += keys.size();
    std::vector<std::optional<std::vector<uint8_t>>> result;
    for (auto key : keys) {
      auto it = entries.find(key);
      result.push_back(it == entries.end() ? std::nullopt
                                           : std::optional(it->second));
    }
    return result;
  }
  void StoreBatch(absl::Span<const StoredNode> batch) override {
    std::lock_guard lock(mutex_);
    for (const auto& node : batch) {
      entries.insert_or_assign(
          node.key,
          std::vector<uint8_t>(node.record.begin(), node.record.end()));
      ++stored;
    }
  }
  void Clear() override {
    std::lock_guard lock(mutex_);
    entries.clear();
  }
  // Seed before Start(), inspect after Wait().
  absl::flat_hash_map<NodeKey, std::vector<uint8_t>> entries;
  size_t loads = 0;
  size_t stored = 0;

 private:
  std::mutex mutex_;
};

class BlockingStore final : public NodeStore {
 public:
  explicit BlockingStore(size_t blocked_load = 1, bool block_persistence = true)
      : blocked_load_(blocked_load), block_persistence_(block_persistence) {}
  std::vector<std::optional<std::vector<uint8_t>>> LoadBatch(
      absl::Span<const NodeKey> keys) override {
    if (load_calls_.fetch_add(1) + 1 == blocked_load_) load.ArriveAndWait();
    return recording.LoadBatch(keys);
  }
  void StoreBatch(absl::Span<const StoredNode> nodes) override {
    if (block_persistence_) persist.ArriveAndWait();
    recording.StoreBatch(nodes);
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

// Evicts every node from `engine` (call last: clears the graph). Because
// EvictToStore skips any node with `in_flight > 0`, evicting the full graph
// proves every visit either backed up or cancelled its reservations cleanly.
void ExpectNoLeakedReservations(VisitEngine& engine) {
  const size_t size = engine.GetGraphSize();
  if (size == 0) return;
  RecordingStore sink;
  Metrics ignored;
  size_t evicted = 0;
  while (const size_t n =
             engine.EvictToStore(sink, size, NodeKey{~0ull}, ignored)) {
    evicted += n;
  }
  EXPECT_EQ(evicted, size);
  EXPECT_EQ(engine.GetGraphSize(), 0u);
}

void ExpectFinalOutput(const RecordingResponder& responder,
                       uint64_t max_nodes) {
  ASSERT_EQ(responder.bestmoves.size(), 1u);
  ASSERT_FALSE(responder.thinking.empty());
  const auto& final = responder.thinking.back();
  EXPECT_LE(final.nodes, max_nodes);
  EXPECT_NE(final.comment.find(" active="), std::string::npos);
  EXPECT_NE(final.comment.find(" ready_eval="), std::string::npos);
  EXPECT_EQ(CommentMetric(final, "rejected"), 0u);
  EXPECT_TRUE(final.comment.ends_with(" final"));
}

// For searches that finished naturally rather than via an external Stop() while
// jobs were still in flight: final info is emitted once active and queued eval
// counts are already zero.
void ExpectIdleFinalOutput(const RecordingResponder& responder,
                           uint64_t expected_nodes) {
  ExpectFinalOutput(responder, expected_nodes);
  const auto& final = responder.thinking.back();
  EXPECT_EQ(final.nodes, expected_nodes);
  EXPECT_EQ(CommentMetric(final, "active"), 0u);
  EXPECT_EQ(CommentMetric(final, "ready_eval"), 0u);
}

void ExpectLegalBestmove(const RecordingResponder& responder,
                         const VisitOrigin& root) {
  ASSERT_EQ(responder.bestmoves.size(), 1u);
  auto move = responder.bestmoves[0].bestmove;
  if (root.history.Last().IsBlackToMove()) move.Flip();
  const auto legal = root.history.Last().GetBoard().GenerateLegalMoves();
  EXPECT_NE(absl::c_find(legal, move), legal.end());
}

TEST(Lc5SearchTest, ExpiredTimeBudgetStopsAndDrains) {
  for (const auto& params :
       {GoParams{.movetime = 0}, GoParams{.movetime = -1},
        GoParams{.wtime = 0, .btime = 0}, GoParams{.movetime = 60000}}) {
    auto engine = MakeTestEngine();
    RecordingStore store;
    ControlledBackend backend;
    RecordingResponder responder;
    const auto root = Origin();
    // A synthetic old origin also verifies that preparation time is charged.
    SearchSession session(
        engine.get(), &backend, &responder, TestSettings(),
        TestTimeManagement(), root, params,
        std::chrono::steady_clock::now() - std::chrono::minutes(2), 0, &store);
    session.Start();
    session.Wait();
    // Fresh roots still evaluate the root at least once before stopping, and
    // all admitted work must drain.
    ExpectFinalOutput(responder, 4);
    ExpectLegalBestmove(responder, root);
    // Nothing is evicted without a graph node limit.
    EXPECT_EQ(store.stored, 0u);
    ExpectNoLeakedReservations(*engine);
  }
}

TEST(Lc5SearchTest, AbortSuppressesOutputWithExpiredTimeBudget) {
  for (const bool abort_before_start : {true, false}) {
    SCOPED_TRACE(abort_before_start);
    auto engine = MakeTestEngine();
    RecordingStore store;
    ControlledBackend backend(ControlledBackend::Cache::kNone,
                              !abort_before_start);
    RecordingResponder responder;
    const auto root = Origin();
    SearchSession session(
        engine.get(), &backend, &responder, TestSettings(),
        TestTimeManagement(), root, GoParams{.movetime = 0},
        std::chrono::steady_clock::now() - std::chrono::minutes(2), 0, &store);
    if (abort_before_start) {
      session.Abort();
      session.Start();
    } else {
      session.Start();
      const bool entered = backend.WaitForCompute();
      session.Abort();
      backend.Release();
      ASSERT_TRUE(entered);
    }
    session.Wait();
    EXPECT_TRUE(responder.bestmoves.empty());
    EXPECT_TRUE(responder.thinking.empty());
    ExpectNoLeakedReservations(*engine);
  }
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
                            std::chrono::steady_clock::now(), 0, nullptr);
      session.Start();
      session.Wait();
      ExpectIdleFinalOutput(responder, 1);
      ExpectLegalBestmove(responder, root);
      EXPECT_EQ(engine->GetRootStats(root.key).visits, 1u);
      ExpectNoLeakedReservations(*engine);
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
      std::chrono::steady_clock::now() - std::chrono::minutes(2), 0, nullptr);
  session.Start();
  SessionWaiter waiter(session);
  const bool computing = backend.WaitForCompute();
  if (computing) {
    EXPECT_FALSE(waiter.IsDone());
    EXPECT_EQ(responder.bestmove_count.load(), 0u);
  } else {
    session.Abort();
  }
  // Release before fatal assertions or destruction, including failure paths.
  backend.Release();
  const bool continued = computing && backend.WaitForInputs(2);
  session.Stop();
  waiter.Join();
  ASSERT_TRUE(computing) << "Ponder must admit work despite zero limits";
  ASSERT_TRUE(continued) << "Ponder must continue after the first evaluation";
  EXPECT_GE(backend.inputs.load(), 2u);
  ExpectFinalOutput(responder, engine->GetRootStats(root.key).visits);
  ExpectLegalBestmove(responder, root);
  ExpectNoLeakedReservations(*engine);
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
                          std::chrono::steady_clock::now(), 0, nullptr);
    session.Start();
    session.Wait();
    const uint64_t expected = std::max<int64_t>(1, nodes);
    ExpectIdleFinalOutput(responder, expected);
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
    ExpectNoLeakedReservations(*engine);
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
                            std::chrono::steady_clock::now(), 0, nullptr);
      session.Start();
      session.Wait();
      ExpectIdleFinalOutput(responder, 1);
      const auto& final = responder.thinking.back();
      ASSERT_TRUE(final.score.has_value());
      EXPECT_EQ(*final.score, test.score);
      ASSERT_TRUE(final.wdl.has_value());
      EXPECT_EQ(final.wdl->w, test.win);
      EXPECT_EQ(final.wdl->d, 500);
      EXPECT_EQ(final.wdl->l, test.loss);
      ExpectNoLeakedReservations(*engine);
    }
  }
}

TEST(Lc5SearchTest, OwnerBacksUpLeafWhileWaitersResumeSelection) {
  auto engine = MakeTestEngine();
  ControlledBackend backend;
  RecordingResponder responder;
  auto root = Origin();
  // With 1 worker and capacity 4, pass 1 admits all four root visits and
  // pass 2 suspends one owner and three waiters on the root before sending the
  // root evaluation job.
  SearchSession session(engine.get(), &backend, &responder, TestSettings(1, 4),
                        TestTimeManagement(), root, GoParams{.nodes = 4},
                        std::chrono::steady_clock::now(), 0, nullptr);
  session.Start();
  session.Wait();
  ExpectIdleFinalOutput(responder, 4);
  // 4 suspensions at the root (1 owner + 3 waiters), then 3 at distinct
  // children after the waiters resume.
  EXPECT_EQ(CommentMetric(responder.thinking.back(), "suspended"), 7u);
  const RootStats stats = engine->GetRootStats(root.key);
  ASSERT_TRUE(stats.materialized);
  EXPECT_EQ(stats.visits, 4u);
  // Only the owner backs up at the root leaf; each waiter traverses an edge to
  // a distinct child.
  EXPECT_EQ(stats.best_edge_visits, 1u);
  EXPECT_EQ(engine->GetGraphSize(), 4u);
  ExpectNoLeakedReservations(*engine);
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
                          std::chrono::steady_clock::now(), 0, nullptr);
    session.Start();
    session.Wait();
    ExpectIdleFinalOutput(responder, kNodes);
    const RootStats stats = engine->GetRootStats(root.key);
    ASSERT_TRUE(stats.materialized);
    EXPECT_EQ(stats.visits, static_cast<uint64_t>(kNodes));
    ExpectNoLeakedReservations(*engine);
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
      engine->InsertNode(
          root.key, {.moves = {moves[0], moves[1]}, .priors = {0.8f, 0.2f}});
      ControlledBackend backend;
      // Positive child Q flips to negative on the root's edge.
      backend.q = child_bad_for_root ? 0.9f : -0.9f;
      RecordingResponder responder;
      auto settings = TestSettings(threads, 4);
      // Hold the 1-item batch until every worker blocks on admission/waiter
      // suspension, so both visits pick moves[0] (0.8, then 0.8 / 2 = 0.4 >
      // 0.2) before the owner's evaluation flushes on starvation.
      settings.max_batch_delay_ms = 60'000;
      SearchSession session(engine.get(), &backend, &responder, settings,
                            TestTimeManagement(), root, GoParams{.nodes = 2},
                            std::chrono::steady_clock::now(), 0, nullptr);
      session.Start();
      session.Wait();
      ExpectIdleFinalOutput(responder, 3);
      const auto& final = responder.thinking.back();
      // Owner + waiter on moves[0], then waiter suspends once more on moves[1]
      // (reselected) or on moves[0]'s child.
      EXPECT_EQ(CommentMetric(final, "suspended"), 3u);
      EXPECT_EQ(CommentMetric(final, "reselected"),
                child_bad_for_root ? 1u : 0u);
      const RootStats stats = engine->GetRootStats(root.key);
      ASSERT_TRUE(stats.materialized);
      EXPECT_EQ(stats.visits, 3u);
      EXPECT_EQ(stats.best_edge_visits, child_bad_for_root ? 1u : 2u);
      EXPECT_EQ(stats.second_edge_visits, child_bad_for_root ? 1u : 0u);
      ExpectNoLeakedReservations(*engine);
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
                         .history = std::move(history)};
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
                          std::chrono::steady_clock::now(), 0, nullptr);
    session.Start();
    session.Wait();
    ExpectIdleFinalOutput(responder, kNodes);
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
    ExpectNoLeakedReservations(*engine);
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
                          std::chrono::steady_clock::now(), 0, nullptr);
    session.Start();
    session.Wait();
    ExpectFinalOutput(responder, nodes);
    const auto& final = responder.thinking.back();
    if (sticky) {
      EXPECT_EQ(responder.bestmoves[0].bestmove, mate);
      EXPECT_EQ(final.mate, 1);
      EXPECT_FALSE(final.score);
      EXPECT_LT(final.nodes, static_cast<uint64_t>(nodes));
    } else {
      EXPECT_FALSE(final.mate);
      EXPECT_TRUE(final.score);
      EXPECT_EQ(final.nodes, static_cast<uint64_t>(nodes));
    }
    ExpectNoLeakedReservations(*engine);
  }
}

TEST(Lc5SearchTest, TwoFoldDrawBacksUpToRoot) {
  // Path: 4 materialized nodes, then an unmaterialized leaf skipped on backup.
  constexpr size_t kNodes = 4;
  const VisitValue draw{.q = 0.0f, .d = 1.0f, .m = 0.0f};
  auto engine = MakeTestEngine();
  std::vector<VisitOrigin> origins;
  VisitOrigin cur = Origin();
  for (size_t i = 0; i < kNodes; ++i) {
    const Move move = cur.history.Last().GetBoard().GenerateLegalMoves()[0];
    engine->InsertNode(cur.key, {.leaf = {.q = 0.5f, .d = 0.0f, .m = 10.0f},
                                 .moves = {move},
                                 .priors = {1.0f}});
    origins.push_back(cur);
    cur.history.Append(move);
    cur.key = MakeNodeKey(cur.history, 7);
  }
  Metrics metrics;
  Visit visit{.id = {},
              .state = VisitState::kReadyDescend,
              .history = origins.front().history,
              .current_key = origins.front().key,
              .path = {}};
  ASSERT_EQ(engine->DescendVisit(visit, metrics, false),
            VisitEngine::DescendResult::kUnmaterializedLeaf);
  ASSERT_EQ(visit.path.size(), kNodes);
  EXPECT_EQ(visit.current_key, cur.key);
  engine->BackupLeaf(visit, draw, metrics, TerminalKind::kNonTerminal, false);

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
  RecordingStore store;
  EXPECT_EQ(engine->EvictToStore(store, kNodes, NodeKey{~0ull}, metrics),
            kNodes);
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
                          std::chrono::steady_clock::now(), 0, nullptr);
    session.Start();
    session.Wait();
    ExpectIdleFinalOutput(responder, kNodes);
    EXPECT_GT(CommentMetric(responder.thinking.back(), "twofold"), 0u);
    const RootStats stats = engine->GetRootStats(root.key);
    ASSERT_TRUE(stats.materialized);
    EXPECT_EQ(stats.visits, static_cast<uint64_t>(kNodes));
    ExpectNoLeakedReservations(*engine);
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
      MemoryNodeStore store;
      ControlledBackend backend;
      RecordingResponder responder;
      const auto root = Origin(test.fen);
      SearchSession session(engine.get(), &backend, &responder,
                            TestSettings(1, 4), TestTimeManagement(), root,
                            GoParams{.nodes = 4},
                            std::chrono::steady_clock::now(), 0,
                            with_store ? &store : nullptr);
      session.Start();
      session.Wait();
      ExpectIdleFinalOutput(responder, 4);
      EXPECT_EQ(responder.bestmoves[0].bestmove,
                Move::WhitePromotion(kSquareH1, kSquareH1, kQueen));
      EXPECT_EQ(backend.inputs.load(), 0u);
      EXPECT_EQ(backend.computes.load(), 0u);
      const auto& final = responder.thinking.back();
      EXPECT_EQ(CommentMetric(final, "suspended"), with_store ? 4u : 1u);
      const RootStats stats = engine->GetRootStats(root.key);
      ASSERT_TRUE(stats.materialized);
      EXPECT_EQ(stats.edges, 0u);
      EXPECT_EQ(stats.visits, 4u);
      ASSERT_TRUE(final.score.has_value());
      EXPECT_EQ(*final.score,
                test.kind == TerminalKind::kCheckmate ? -12780 : 0);
      ASSERT_TRUE(final.wdl.has_value());
      EXPECT_EQ(final.wdl->w, 0);
      EXPECT_EQ(final.wdl->d, test.kind == TerminalKind::kStalemate ? 1000 : 0);
      EXPECT_EQ(final.wdl->l, test.kind == TerminalKind::kCheckmate ? 1000 : 0);
      ExpectNoLeakedReservations(*engine);
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
    const auto origins = InsertUniformRootChildren(*engine, root, 4);
    SearchSession session(engine.get(), &backend, &responder,
                          TestSettings(1, 4), TestTimeManagement(), root,
                          GoParams{.nodes = 4},
                          std::chrono::steady_clock::now(), 0, nullptr);
    session.Start();
    session.Wait();
    ExpectIdleFinalOutput(responder, 5);
    const bool all = cache == ControlledBackend::Cache::kAll;
    EXPECT_EQ(backend.inputs.load(), 4u);
    EXPECT_EQ(backend.computes.load(), all ? 0u : 1u);
    const auto& final = responder.thinking.back();
    EXPECT_EQ(CommentMetric(final, "cache"), all ? 4u : 2u);
    EXPECT_EQ(CommentMetric(final, "eps_evals"), all ? 0u : 2u);
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
    ExpectNoLeakedReservations(*engine);
  }
}

TEST(Lc5SearchTest, BlockedBackendStopAndAbortDrainReservations) {
  for (const bool abort : {false, true}) {
    SCOPED_TRACE(abort);
    auto engine = MakeTestEngine();
    RecordingStore store;
    ControlledBackend backend(ControlledBackend::Cache::kNone, true);
    RecordingResponder responder;
    const auto root = Origin();
    const auto moves = root.history.Last().GetBoard().GenerateLegalMoves();
    // All four visits on the single worker reserve the same root edge and share
    // the child materialization before the load and eval jobs are sent.
    engine->InsertNode(root.key, {.moves = {moves[0]}, .priors = {1.0f}});
    SearchSession session(engine.get(), &backend, &responder,
                          TestSettings(1, 4), TestTimeManagement(), root,
                          GoParams{.nodes = 4, .infinite = true},
                          std::chrono::steady_clock::now(), 0, &store);
    session.Start();
    SessionWaiter waiter(session);
    const bool entered = backend.WaitForCompute();
    // Always unblock before any fatal assertion or SearchSession destruction.
    if (!entered) {
      session.Abort();
      backend.Release();
      waiter.Join();
      FAIL() << "Backend computation was not reached";
    }
    if (abort) {
      session.Abort();
    } else {
      session.Stop();
      EXPECT_TRUE(responder.WaitForBestmove());
      session.Abort();  // A committed response cannot be retracted.
      session.Stop();
    }
    EXPECT_FALSE(waiter.IsDone());
    EXPECT_EQ(responder.bestmove_count.load(), abort ? 0u : 1u);
    backend.Release();
    waiter.Join();
    // The evaluation was authorized before stop, so the owner backs up when
    // it returns. The 3 waiters would select below the child and cancel.
    EXPECT_EQ(store.loads, 1u);
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
      // 1 owner + 3 waiters were suspended on the child when bestmove emitted.
      EXPECT_EQ(CommentMetric(responder.thinking.back(), "suspended"), 4u);
    }
    // Every reservation was released, so both nodes can be evicted.
    EXPECT_EQ(engine->GetGraphSize(), 2u);
    ExpectNoLeakedReservations(*engine);
  }
}

TEST(Lc5SearchTest, SavedGraphRestoresWithStatisticsInFreshGraph) {
  ControlledBackend backend;
  const auto root = Origin();
  auto saved_engine = MakeTestEngine();
  RecordingStore saved_store;
  {
    RecordingResponder responder;
    SearchSession session(saved_engine.get(), &backend, &responder,
                          TestSettings(), TestTimeManagement(), root,
                          GoParams{.nodes = 3},
                          std::chrono::steady_clock::now(), 0, &saved_store);
    session.Start();
    session.Wait();
    ExpectIdleFinalOutput(responder, 3);
    EXPECT_EQ(saved_store.stored, 0u);
  }
  const RootStats saved = saved_engine->GetRootStats(root.key);
  ASSERT_EQ(saved.visits, 3u);
  const size_t saved_size = saved_engine->GetGraphSize();
  Metrics evict_metrics;
  ASSERT_EQ(saved_engine->EvictToStore(saved_store, saved_size, NodeKey{~0ull},
                                       evict_metrics),
            saved_size);

  // A fresh graph restores the root with its statistics instead of
  // evaluating it, and the new visit goes below it.
  auto engine = MakeTestEngine();
  RecordingStore store;
  store.entries = saved_store.entries;
  EXPECT_EQ(store.entries.size(), saved_size);
  const size_t evaluated = backend.inputs.load();

  RecordingResponder responder;
  SearchSession session(engine.get(), &backend, &responder, TestSettings(),
                        TestTimeManagement(), root, GoParams{.nodes = 1},
                        std::chrono::steady_clock::now(), 0, &store);
  session.Start();
  session.Wait();
  ExpectIdleFinalOutput(responder, 1);
  const auto& final = responder.thinking.back();
  EXPECT_GE(CommentMetric(final, "store_hit"), 1u);
  EXPECT_EQ(CommentMetric(final, "restored"), CommentMetric(final, "store_hit"));
  const RootStats stats = engine->GetRootStats(root.key);
  ASSERT_TRUE(stats.materialized);
  EXPECT_EQ(stats.visits, saved.visits + 1);
  EXPECT_EQ(stats.edges, saved.edges);
  // Saved nodes are restored too; the visit ends at the first node that was
  // never materialized, which is the only new evaluation.
  EXPECT_EQ(backend.inputs.load(), evaluated + 1);
  ExpectNoLeakedReservations(*engine);
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
                          std::chrono::steady_clock::now(), 0, nullptr);
    session.Start();
    SessionWaiter waiter(session);
    const bool entered = backend.WaitForCompute();
    if (entered) {
      // Assert before stop: stop/drain must not be what unblocks collection.
      EXPECT_FALSE(waiter.IsDone());
      EXPECT_EQ(responder.bestmove_count.load(), 0u);
    } else {
      session.Abort();
    }
    // Unblock before fatal assertions, including the failure path.
    backend.Release();
    waiter.Join();
    ASSERT_TRUE(entered) << "Root bootstrap depended on the collection timer";
    ExpectIdleFinalOutput(responder, 1);
    EXPECT_EQ(CommentMetric(responder.thinking.back(), "eps_evals"), 1u);
    EXPECT_EQ(backend.batch_sizes, (std::vector<size_t>{1}));
    ExpectNoLeakedReservations(*engine);
  }
}

TEST(Lc5SearchTest, CollectionFlushesWhileAnotherWorkerIsBlockedInStore) {
  for (const int delay : {0, 1}) {
    SCOPED_TRACE(delay);
    auto engine = MakeTestEngine();
    // First load misses; the other worker's load remains executing. With a
    // positive delay it prevents dependency-starvation detection, so the
    // collector must actually time out. Zero must flush without that timer.
    BlockingStore store(2, false);
    ControlledBackend backend(ControlledBackend::Cache::kNone, true);
    RecordingResponder responder;
    const auto root = Origin();
    InsertUniformRootChildren(*engine, root, 2);
    auto settings = TestSettings(2, 2);
    settings.max_batch_delay_ms = delay;
    SearchSession session(engine.get(), &backend, &responder, settings,
                          TestTimeManagement(), root, GoParams{.nodes = 2},
                          std::chrono::steady_clock::now(), 0, &store);
    session.Start();
    SessionWaiter waiter(session);
    const bool loading = store.load.WaitForArrival();
    const bool computing = backend.WaitForCompute();
    if (loading && computing) {
      EXPECT_FALSE(waiter.IsDone());
    }
    // This also exercises waking collectors and finishing returned store work
    // after stop. Never leave either executor blocked on a failed assertion.
    // The final info precedes release, so the late backup is not counted.
    session.Stop();
    const bool responding = responder.WaitForBestmove();
    EXPECT_FALSE(waiter.IsDone());
    store.Release();
    backend.Release();
    waiter.Join();
    ASSERT_TRUE(loading);
    ASSERT_TRUE(computing)
        << "Collection depended on the blocked store returning";
    ASSERT_TRUE(responding);
    ExpectFinalOutput(responder, 1);
    EXPECT_EQ(CommentMetric(responder.thinking.back(), "eps_evals"), 1u);
    // The computed evaluation backs up (+1 at root); the other store miss is
    // cancelled.
    EXPECT_EQ(engine->GetRootStats(root.key).visits, 2u);
    EXPECT_EQ(store.recording.loads, 2u);
    EXPECT_EQ(store.recording.stored, 0u);
    EXPECT_EQ(backend.computes.load(), 1u);
    ExpectNoLeakedReservations(*engine);
  }
}

TEST(Lc5SearchTest, PositiveDelayCollectsAnotherWorkersStoreMissIntoSameBatch) {
  auto engine = MakeTestEngine();
  BlockingStore store(2, false);
  ControlledBackend backend(ControlledBackend::Cache::kNone, true);
  RecordingResponder responder;
  const auto root = Origin();
  InsertUniformRootChildren(*engine, root, 2);
  auto settings = TestSettings(2, 2);
  settings.minibatch_size = 2;
  settings.max_batch_delay_ms = 60000;
  SearchSession session(engine.get(), &backend, &responder, settings,
                        TestTimeManagement(), root, GoParams{.nodes = 2},
                        std::chrono::steady_clock::now(), 0, &store);
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
  ExpectIdleFinalOutput(responder, 3);
  EXPECT_EQ(backend.batch_sizes, (std::vector<size_t>{2}));
  EXPECT_EQ(CommentMetric(responder.thinking.back(), "batches"), 1u);
  EXPECT_EQ(store.recording.stored, 0u);
  ExpectNoLeakedReservations(*engine);
}

TEST(Lc5SearchTest, MultiworkerMixedCacheResultsRespectBatchAndItemLimits) {
  // High-mobility position with > 130 non-terminal legal moves so up to 128
  // root visits fan out 1-to-1 across independent unmaterialized children.
  constexpr std::string_view kWideFen =
      "kr6/pp6/3Q4/1Q2Q2Q/8/2Q2Q2/1Q4Q1/3RK3 w - - 0 1";
  struct Case {
    int evaluators;
    int batch_size;
    int capacity;
  };
  for (const auto& test :
       {Case{1, 4, 40}, Case{2, 16, 64}, Case{4, 32, 128}}) {
    SCOPED_TRACE(test.evaluators);
    SCOPED_TRACE(test.batch_size);
    for (const int delay : {0, 10}) {
      SCOPED_TRACE(delay);
      auto engine = MakeTestEngine();
      ControlledBackend backend(ControlledBackend::Cache::kAlternating, true,
                                test.batch_size, test.evaluators);
      RecordingResponder responder;
      const auto root = Origin(kWideFen);
      const auto origins =
          InsertUniformRootChildren(*engine, root, test.capacity);
      auto settings = TestSettings(4, test.capacity);
      settings.eval_threads = test.evaluators;
      settings.minibatch_size = test.batch_size;
      settings.max_batch_delay_ms = delay;
      // Target 4 stresses the in-flight limits with one-item jobs; targets
      // 16/32 exercise mixed results in four/eight-item jobs across evaluators.
      SearchSession session(engine.get(), &backend, &responder, settings,
                            TestTimeManagement(), root,
                            GoParams{.nodes = test.capacity},
                            std::chrono::steady_clock::now(), 0, nullptr);
      session.Start();
      const bool computing = backend.WaitForCompute();
      if (!computing) session.Abort();
      // Release also opens the initial evaluator cohort on failure. Never
      // destroy a session with fixture barriers still blocking its executors.
      backend.Release();
      session.Wait();
      ASSERT_TRUE(computing);
      ExpectIdleFinalOutput(responder, 1 + test.capacity);
      const auto& final = responder.thinking.back();
      EXPECT_EQ(backend.evaluator_ids.size(),
                static_cast<size_t>(test.evaluators));
      EXPECT_EQ(backend.inputs.load(), static_cast<size_t>(test.capacity));
      EXPECT_EQ(CommentMetric(final, "cache"),
                static_cast<uint64_t>(test.capacity / 2));
      EXPECT_EQ(CommentMetric(final, "eps_evals"),
                static_cast<uint64_t>(test.capacity / 2));
      EXPECT_EQ(engine->GetRootStats(root.key).visits,
                1u + static_cast<uint64_t>(test.capacity));
      EXPECT_EQ(engine->GetGraphSize(), 1u + test.capacity);
      const size_t job_size = (test.batch_size + 3) / 4;
      const size_t max_jobs =
          std::min<size_t>(16, 2 * test.batch_size / job_size);
      EXPECT_LE(CommentMetric(final, "io_high"), 4 * max_jobs);
      size_t evaluated = 0;
      for (const auto size : backend.batch_sizes) {
        EXPECT_GT(size, 0u);
        EXPECT_LE(size, static_cast<size_t>(settings.minibatch_size));
        evaluated += size;
      }
      EXPECT_EQ(evaluated, static_cast<size_t>(test.capacity / 2));
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
      ExpectNoLeakedReservations(*engine);
    }
  }
}

TEST(Lc5SearchTest, BlockedStoreLoadsDrainAfterStopOrAbort) {
  for (const bool hit : {false, true}) {
    for (const bool abort : {false, true}) {
      SCOPED_TRACE(hit);
      SCOPED_TRACE(abort);
      auto engine = MakeTestEngine();
      BlockingStore store;
      ControlledBackend backend(ControlledBackend::Cache::kAll);
      RecordingResponder responder;
      const auto root = Origin();
      const auto move = root.history.Last().GetBoard().GenerateLegalMoves()[0];
      engine->InsertNode(root.key, {.moves = {move}, .priors = {1.0f}});
      auto child = root;
      child.history.Append(move);
      child.key = MakeNodeKey(child.history, 7);
      if (hit) {
        auto donor = MakeTestEngine();
        RecordingStore donor_store;
        auto legal = child.history.Last().GetBoard().GenerateLegalMoves();
        absl::c_sort(
            legal, [](Move a, Move b) { return a.raw_data() < b.raw_data(); });
        donor->InsertNode(
            child.key,
            {.leaf = {.q = 0.75f, .d = 0.25f, .m = 7.0f},
             .moves = {legal.begin(), legal.end()},
             .priors = std::vector<float>(legal.size(), 1.0f / legal.size())});
        Metrics evict_metrics;
        ASSERT_EQ(donor->EvictToStore(donor_store, 1, root.key, evict_metrics),
                  1u);
        store.recording.entries = donor_store.entries;
      }
      SearchSession session(engine.get(), &backend, &responder,
                            TestSettings(1, 4), TestTimeManagement(), root,
                            GoParams{.nodes = 4, .infinite = true},
                            std::chrono::steady_clock::now(), 0, &store);
      session.Start();
      SessionWaiter waiter(session);
      const bool loading = store.load.WaitForArrival();
      if (loading) {
        EXPECT_EQ(backend.inputs.load(), 0u);
      }
      if (abort || !loading) {
        session.Abort();
      } else {
        session.Stop();
        EXPECT_TRUE(responder.WaitForBestmove());
      }
      EXPECT_FALSE(waiter.IsDone());
      store.load.Release();
      store.Release();
      waiter.Join();
      ASSERT_TRUE(loading);
      // A load already running at stop completes: on a hit, the node is
      // restored, but a restored node has nothing new to back up, so the
      // owner and the waiters all cancel. A miss is never evaluated.
      EXPECT_EQ(store.recording.loads, 1u);
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
        EXPECT_EQ(CommentMetric(responder.thinking.back(), "suspended"), 4u);
      }
      ExpectNoLeakedReservations(*engine);
    }
  }
}

TEST(Lc5SearchTest, QueuedBacklogCannotComputeAgainAfterStop) {
  // Position with 27 legal moves so up to 24 visits fan out 1-to-1 from root.
  constexpr std::string_view kOpenFen =
      "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3";
  for (const int evaluators : {1, 3}) {
    SCOPED_TRACE(evaluators);
    auto engine = MakeTestEngine();
    ControlledBackend backend(ControlledBackend::Cache::kNone, true, 5,
                              evaluators);
    RecordingResponder responder;
    const auto root = Origin(kOpenFen);
    const int capacity = 8 * evaluators;
    InsertUniformRootChildren(*engine, root, capacity);
    auto settings = TestSettings(evaluators, capacity);
    settings.eval_threads = evaluators;
    // Jobs hold CeilDiv(5, 4) = 2 items. Each evaluator collects 2 jobs (4
    // items) and waits in PopUntil until a non-fitting 2-item backlog job is
    // already queued (4 + 2 > 5) before entering ComputeBlocking.
    settings.minibatch_size = 5;
    settings.max_batch_delay_ms = 60'000;
    SearchSession session(engine.get(), &backend, &responder, settings,
                          TestTimeManagement(), root,
                          GoParams{.nodes = capacity},
                          std::chrono::steady_clock::now(), 0, nullptr);
    session.Start();
    SessionWaiter waiter(session);
    // Each evaluator is executing one authorized batch, with more ready jobs
    // queued in the evaluator. Stop must not authorize a second batch.
    const bool computing = backend.WaitForCompute(evaluators);
    session.Stop();
    const bool responding = responder.WaitForBestmove();
    EXPECT_FALSE(waiter.IsDone());
    backend.Release();
    waiter.Join();
    ASSERT_TRUE(computing);
    ASSERT_TRUE(responding);
    EXPECT_GE(CommentMetric(responder.thinking.back(), "ready_eval"), 1u);
    EXPECT_EQ(backend.computes.load(), static_cast<size_t>(evaluators));
    EXPECT_EQ(backend.calls_by_evaluator.size(),
              static_cast<size_t>(evaluators));
    for (const auto& [id, calls] : backend.calls_by_evaluator) {
      EXPECT_EQ(calls, 1u);
    }
    EXPECT_EQ(backend.inputs.load(), 4u * evaluators);
    EXPECT_EQ(engine->GetGraphSize(), 1u + 4u * evaluators);
    // Authorized batches back up their visits; the queued backlog cancels.
    EXPECT_EQ(engine->GetRootStats(root.key).visits, 1u + 4u * evaluators);
    ExpectLegalBestmove(responder, root);
    ExpectNoLeakedReservations(*engine);
  }
}

TEST(Lc5SearchTest, AuthorizedEvaluationsBackUpToRootAfterStop) {
  auto engine = MakeTestEngine();
  ControlledBackend backend(ControlledBackend::Cache::kNone, true);
  RecordingResponder responder;
  const auto root = Origin();
  InsertUniformRootChildren(*engine, root, 20);
  SearchSession session(engine.get(), &backend, &responder, TestSettings(4, 20),
                        TestTimeManagement(), root, GoParams{.nodes = 20},
                        std::chrono::steady_clock::now(), 0, nullptr);
  session.Start();
  SessionWaiter waiter(session);
  // One authorized batch of four children is computing; the other visits are
  // queued or still running, and must cancel.
  const bool computing = backend.WaitForCompute();
  session.Stop();
  const bool responding = responder.WaitForBestmove();
  EXPECT_FALSE(waiter.IsDone());
  backend.Release();
  waiter.Join();
  ASSERT_TRUE(computing);
  ASSERT_TRUE(responding);
  EXPECT_EQ(backend.computes.load(), 1u);
  EXPECT_EQ(backend.inputs.load(), 4u);
  // Each computed materialization's owner backs up through the root after stop.
  const RootStats stats = engine->GetRootStats(root.key);
  ASSERT_TRUE(stats.materialized);
  EXPECT_EQ(stats.visits, 5u);
  EXPECT_GE(stats.best_edge_visits, 1u);
  ExpectFinalOutput(responder, 21);
  ExpectNoLeakedReservations(*engine);
}

TEST(Lc5SearchTest, StopDuringAddInputKeepsOnlyImmediateResults) {
  auto engine = MakeTestEngine();
  ControlledBackend backend(ControlledBackend::Cache::kAlternating, true, 8);
  backend.blocked_input = 2;
  RecordingResponder responder;
  const auto root = Origin();
  const auto origins = InsertUniformRootChildren(*engine, root, 4);
  auto settings = TestSettings(1, 4);
  settings.minibatch_size = 8;
  SearchSession session(engine.get(), &backend, &responder, settings,
                        TestTimeManagement(), root, GoParams{.nodes = 4},
                        std::chrono::steady_clock::now(), 0, nullptr);
  session.Start();
  SessionWaiter waiter(session);
  // The first job has an immediate hit and an AddInput that will enqueue only
  // after release. The second job must not even reach AddInput after stop.
  const bool adding = backend.input.WaitForArrival();
  session.Stop();
  const bool responding = responder.WaitForBestmove();
  EXPECT_FALSE(waiter.IsDone());
  backend.Release();
  waiter.Join();
  ASSERT_TRUE(adding);
  ASSERT_TRUE(responding);
  EXPECT_EQ(backend.inputs.load(), 2u);
  EXPECT_EQ(backend.computes.load(), 0u);
  EXPECT_EQ(engine->GetGraphSize(), 2u);
  const RootStats hit = engine->GetRootStats(origins[0].key);
  ASSERT_TRUE(hit.materialized);
  EXPECT_EQ(hit.visits, 1u);
  // Setup install plus the 1 immediate cache hit backed up; the other 3 cancel.
  EXPECT_EQ(engine->GetRootStats(root.key).visits, 2u);
  ThinkingInfo info;
  engine->FillThinkingInfo(origins[0], info);
  ASSERT_TRUE(info.score.has_value());
  EXPECT_EQ(*info.score, CentipawnsFromQ(backend.q));
  for (size_t i = 1; i < origins.size(); ++i) {
    EXPECT_FALSE(engine->ContainsNode(origins[i].key));
  }
  ExpectNoLeakedReservations(*engine);
}

TEST(Lc5SearchTest, NonfittingJobStaysQueuedAndIsCancelledAfterStop) {
  auto engine = MakeTestEngine();
  ControlledBackend backend(ControlledBackend::Cache::kNone, true, 5);
  RecordingResponder responder;
  const auto root = Origin();
  const auto origins = InsertUniformRootChildren(*engine, root, 6);
  auto settings = TestSettings(1, 6);
  settings.minibatch_size = 5;
  settings.max_batch_delay_ms = 60'000;
  SearchSession session(engine.get(), &backend, &responder, settings,
                        TestTimeManagement(), root, GoParams{.nodes = 6},
                        std::chrono::steady_clock::now(), 0, nullptr);
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
  EXPECT_EQ(CommentMetric(responder.thinking.back(), "ready_eval"), 1u);
  EXPECT_EQ(backend.batch_sizes, (std::vector<size_t>{4}));
  EXPECT_EQ(backend.inputs.load(), 4u);
  EXPECT_EQ(backend.computes.load(), 1u);
  EXPECT_EQ(engine->GetGraphSize(), 5u);
  for (size_t i = 4; i < origins.size(); ++i) {
    EXPECT_FALSE(engine->ContainsNode(origins[i].key));
  }
  ExpectNoLeakedReservations(*engine);
}

TEST(Lc5SearchTest, AbortOverridesRepeatedStopBeforeCommitment) {
  auto engine = MakeTestEngine();
  ControlledBackend backend;
  RecordingResponder responder;
  const auto root = Origin();
  SearchSession session(engine.get(), &backend, &responder, TestSettings(),
                        TestTimeManagement(), root, GoParams{.nodes = 4},
                        std::chrono::steady_clock::now(), 0, nullptr);
  session.Stop();
  session.Stop();
  session.Abort();
  session.Stop();
  session.Abort();
  session.Start();
  session.Wait();
  EXPECT_TRUE(responder.bestmoves.empty());
  EXPECT_TRUE(responder.thinking.empty());
  EXPECT_EQ(backend.inputs.load(), 0u);
  EXPECT_EQ(backend.computes.load(), 0u);
  EXPECT_EQ(engine->GetGraphSize(), 0u);
}

TEST(Lc5SearchTest, BlockedEvictionDoesNotDelayPartialBatch) {
  auto engine = MakeTestEngine();
  BlockingStore store;
  ControlledBackend backend(ControlledBackend::Cache::kNone, true);
  RecordingResponder responder;
  const auto root = Origin();
  const auto legal = root.history.Last().GetBoard().GenerateLegalMoves();
  // As in RespondsBeforeBlockedEvictionDrains: the sibling is evicted while
  // the one visit evaluates the first edge's child.
  engine->InsertNode(root.key,
                     {.moves = {legal[0], legal[1]}, .priors = {0.9f, 0.1f}});
  auto sibling = root;
  sibling.history.Append(legal[1]);
  sibling.key = MakeNodeKey(sibling.history, 7);
  engine->InsertNode(
      sibling.key,
      {.leaf = {.q = 0.5f, .d = 0.0f, .m = 0.0f}, .moves = {}, .priors = {}});
  auto settings = TestSettings(1, 1);
  settings.graph_node_limit = 1;
  settings.eval_threads = 2;
  // The one-item batch can only flush on starvation, not at the 1-minute
  // deadline (WaitForCompute times out in 5s if starvation is missed).
  settings.max_batch_delay_ms = 60'000;
  SearchSession session(engine.get(), &backend, &responder, settings,
                        TestTimeManagement(), root, GoParams{.nodes = 1},
                        std::chrono::steady_clock::now(), 0, &store);
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
  EXPECT_EQ(backend.batch_sizes, (std::vector<size_t>{1}));
  ExpectNoLeakedReservations(*engine);
}

TEST(Lc5SearchTest, RespondsBeforeBlockedEvictionDrains) {
  auto engine = MakeTestEngine();
  BlockingStore store(0);
  ControlledBackend backend(ControlledBackend::Cache::kNone, true);
  RecordingResponder responder;
  const auto root = Origin();
  const auto legal = root.history.Last().GetBoard().GenerateLegalMoves();
  // The one visit takes the first edge and blocks in the backend; the
  // sibling below the second edge is the node to evict.
  engine->InsertNode(root.key,
                     {.moves = {legal[0], legal[1]}, .priors = {0.9f, 0.1f}});
  auto sibling = root;
  sibling.history.Append(legal[1]);
  sibling.key = MakeNodeKey(sibling.history, 7);
  engine->InsertNode(
      sibling.key,
      {.leaf = {.q = 0.5f, .d = 0.0f, .m = 0.0f}, .moves = {}, .priors = {}});
  auto settings = TestSettings(1, 1);
  settings.graph_node_limit = 1;
  // Store threads come one per eval thread. With only one, the visit's store
  // load could queue behind the blocked eviction and never reach the backend.
  settings.eval_threads = 2;
  SearchSession session(engine.get(), &backend, &responder, settings,
                        TestTimeManagement(), root, GoParams{.nodes = 1},
                        std::chrono::steady_clock::now(), 0, &store);
  session.Start();
  SessionWaiter waiter(session);
  const bool persisting = store.persist.WaitForArrival();
  const bool computing = backend.WaitForCompute();
  session.Stop();
  const bool responding = responder.WaitForBestmove();
  EXPECT_FALSE(waiter.IsDone());
  store.Release();
  backend.Release();
  waiter.Join();
  ASSERT_TRUE(persisting);
  ASSERT_TRUE(computing);
  ASSERT_TRUE(responding);
  ExpectFinalOutput(responder, 1);
  // Stored before it was erased.
  EXPECT_TRUE(store.recording.entries.contains(sibling.key));
  EXPECT_FALSE(engine->ContainsNode(sibling.key));
  EXPECT_TRUE(engine->ContainsNode(root.key));
  ExpectNoLeakedReservations(*engine);
}

TEST(Lc5SearchTest, EvictionKeepsGraphSmallAndLosesNoVisits) {
  for (const int threads : {1, 3}) {
    SCOPED_TRACE(threads);
    auto engine = MakeTestEngine();
    MemoryNodeStore store;
    ControlledBackend backend;
    RecordingResponder responder;
    const auto root = Origin();
    auto settings = TestSettings(threads, 8);
    settings.graph_node_limit = 32;
    SearchSession session(engine.get(), &backend, &responder, settings,
                          TestTimeManagement(), root, GoParams{.nodes = 3000},
                          std::chrono::steady_clock::now(), 0, &store);
    session.Start();
    session.Wait();
    ExpectIdleFinalOutput(responder, 3000);
    const auto& final = responder.thinking.back();
    EXPECT_GT(CommentMetric(final, "evicted"), 0u);
    EXPECT_GT(CommentMetric(final, "restored"), 0u);
    EXPECT_EQ(CommentMetric(final, "restored"),
              CommentMetric(final, "store_hit"));
    // Every completed visit reached the root, which is never evicted.
    const RootStats stats = engine->GetRootStats(root.key);
    ASSERT_TRUE(stats.materialized);
    EXPECT_EQ(stats.visits, 3000u);
    EXPECT_GT(stats.best_edge_visits, 0u);
    // Far fewer nodes than visits stay in RAM.
    EXPECT_LT(engine->GetGraphSize(), 3000u / 4);
    ExpectNoLeakedReservations(*engine);
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
                        std::chrono::steady_clock::now(), 0, nullptr);
  session.Start();
  session.Wait();
  ExpectIdleFinalOutput(responder, 1);
  const RootStats stats = engine->GetRootStats(root.key);
  ASSERT_TRUE(stats.materialized);
  ASSERT_GT(stats.edges, 0u);
  EXPECT_EQ(stats.best_edge_visits, 0u);
  Move expected = stats.best_move;
  expected.Flip();
  ASSERT_EQ(responder.thinking.back().pv.size(), 1u);
  EXPECT_EQ(responder.thinking.back().pv[0], expected);
  EXPECT_EQ(responder.bestmoves[0].bestmove, expected);
  ExpectNoLeakedReservations(*engine);
}

TEST(Lc5SearchTest, FreshRootStoppedBeforeStartStillEvaluatesRootOnce) {
  auto engine = MakeTestEngine();
  ControlledBackend backend;
  RecordingResponder responder;
  const auto root = Origin();
  SearchSession session(engine.get(), &backend, &responder, TestSettings(),
                        TestTimeManagement(), root, GoParams{},
                        std::chrono::steady_clock::now(), 0, nullptr);
  session.Stop();
  session.Start();
  session.Wait();
  ExpectIdleFinalOutput(responder, 1);
  ExpectLegalBestmove(responder, root);
  EXPECT_EQ(backend.inputs.load(), 1u);
  EXPECT_EQ(backend.computes.load(), 1u);
  EXPECT_EQ(engine->GetGraphSize(), 1u);
  const RootStats stats = engine->GetRootStats(root.key);
  ASSERT_TRUE(stats.materialized);
  ASSERT_GT(stats.edges, 0u);
  EXPECT_EQ(responder.bestmoves[0].bestmove, stats.best_move);
  ExpectNoLeakedReservations(*engine);
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
      ExpectIdleFinalOutput(responder, 1);
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
    EXPECT_EQ(responder.thinking.back().nodes, i == 1 ? 2u : 1u);
    EXPECT_EQ(CommentMetric(responder.thinking.back(), "active"), 0u);
    EXPECT_EQ(CommentMetric(responder.thinking.back(), "ready_eval"), 0u);
    EXPECT_EQ(CommentMetric(responder.thinking.back(), "rejected"), 0u);
    EXPECT_TRUE(responder.thinking.back().comment.ends_with(" final"));
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
  engine->InsertNode(root.key, {.moves = {m0, m1}, .priors = {0.5f, 0.5f}});

  // Pause at input 2 before Fill() so visit 1's backup is reported first, then
  // switch q so visit 2 flips the root's best move well before the 1s timer.
  backend.q = 0.8f;
  backend.blocked_input = 2;
  SearchSession session(engine.get(), &backend, &responder, TestSettings(1, 1),
                        TestTimeManagement(), root, GoParams{.infinite = true},
                        std::chrono::steady_clock::now(), 0, nullptr);
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
  ExpectNoLeakedReservations(*engine);
}

}  // namespace
}  // namespace lczero::lc5

int main(int argc, char** argv) {
  lczero::InitializeMagicBitboards();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
