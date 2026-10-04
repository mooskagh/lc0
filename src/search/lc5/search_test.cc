#include "search/lc5/search.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "search/lc5/engine.h"

namespace lczero::lc5 {
namespace {

Settings::Resolved TestSettings(int threads = 1, int capacity = 4) {
  return {.threads = threads,
          .eval_threads = 1,
          .minibatch_size = 4,
          .max_active_visits = capacity,
          .history_key_length = 7,
          .max_batch_delay_ms = 0,
          .cpuct = 1.0f,
          .cpuct_base = 100.0f,
          .cpuct_factor = 0.0f,
          .fpu_strategy = FpuStrategy::kAbsolute,
          .fpu_value = 0.0f};
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
    bestmoves.push_back(*info);
    bestmove_count.fetch_add(1);
  }
  void OutputThinkingInfo(std::vector<ThinkingInfo>* infos) override {
    thinking.insert(thinking.end(), infos->begin(), infos->end());
  }
  std::atomic<size_t> bestmove_count{0};
  std::vector<BestMoveInfo> bestmoves;
  std::vector<ThinkingInfo> thinking;
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
  bool WaitForCompute() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, std::chrono::seconds(5),
                        [&] { return entered_; });
  }
  void Release() {
    {
      std::lock_guard lock(mutex_);
      released_ = true;
    }
    cv_.notify_all();
  }

  std::atomic<size_t> inputs{0};
  std::atomic<size_t> computes{0};
  // Protected by mutex_ while writing; tests read after the run joins.
  std::vector<size_t> history_sizes;
  std::vector<size_t> batch_sizes;
  std::unordered_set<std::thread::id> evaluator_ids;

 private:
  static void Fill(EvalResultPtr result) {
    *result.q = 0.25f;
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
      if (backend_.cache_ == Cache::kAll ||
          (backend_.cache_ == Cache::kAlternating && index % 2 == 0)) {
        Fill(result);
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
        backend_.entered_ = true;
        backend_.cv_.notify_all();
        backend_.cv_.wait(lock, [&] { return backend_.released_; });
      }
      backend_.computes.fetch_add(1);
      for (size_t i = 0; i < pending_.size(); ++i) {
        EXPECT_FALSE(positions_[i].pos.empty());
        EXPECT_EQ(positions_[i].legal_moves.size(), pending_[i].p.size());
        Fill(pending_[i]);
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
  bool entered_ = false;
  bool released_;
};

class RecordingStore final : public NodeStore {
 public:
  std::vector<std::optional<ExpansionPayload>> LoadBatch(
      std::span<const NodeKey> keys) override {
    std::lock_guard lock(mutex_);
    std::vector<std::optional<ExpansionPayload>> result;
    for (auto key : keys) {
      auto it = entries.find(key);
      result.push_back(it == entries.end() ? std::nullopt
                                           : std::optional(it->second));
    }
    return result;
  }
  void StoreBatch(std::span<const StoredExpansion> batch) override {
    std::lock_guard lock(mutex_);
    for (const auto& entry : batch) {
      entries.insert_or_assign(entry.key, entry.payload);
      ++stored;
    }
  }
  void Clear() override {
    std::lock_guard lock(mutex_);
    entries.clear();
  }
  // Seed before Start(), inspect after Wait().
  std::unordered_map<NodeKey, ExpansionPayload, NodeKeyHash> entries;
  size_t stored = 0;

 private:
  std::mutex mutex_;
};

// One-shot barrier between the test thread and an existing scheduler thread.
// Arrival is observable before the operation returns; release is never timed.
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

class BlockingStore final : public NodeStore {
 public:
  explicit BlockingStore(size_t blocked_load = 1, bool block_persistence = true)
      : blocked_load_(blocked_load), block_persistence_(block_persistence) {}
  std::vector<std::optional<ExpansionPayload>> LoadBatch(
      std::span<const NodeKey> keys) override {
    if (load_calls_.fetch_add(1) + 1 == blocked_load_) load.ArriveAndWait();
    return recording.LoadBatch(keys);
  }
  void StoreBatch(std::span<const StoredExpansion> entries) override {
    if (block_persistence_) persist.ArriveAndWait();
    recording.StoreBatch(entries);
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

void ExpectDrained(const SearchRun& run, const GameGraph& graph) {
  ASSERT_TRUE(run.Finished());
  const auto& m = run.metrics();
  EXPECT_EQ(m.visits_admitted.load(),
            m.visits_completed.load() + m.visits_cancelled.load());
  EXPECT_EQ(m.invariant_underflow_prevented.load(), 0u);
  EXPECT_EQ(m.publications_rejected.load(), 0u);
  for (const auto& [key, node] : graph.SnapshotAllForTesting()) {
    SCOPED_TRACE(key.hash);
    EXPECT_EQ(node.lifecycle, NodeLifecycle::kExpanded);
    EXPECT_EQ(node.ticket, 0u);
    for (const auto& edge : node.edges) EXPECT_EQ(edge.in_flight, 0u);
  }
}

void ExpectFinalOutput(const RecordingResponder& responder, uint64_t nodes) {
  ASSERT_EQ(responder.bestmoves.size(), 1u);
  ASSERT_FALSE(responder.thinking.empty());
  const auto& final = responder.thinking.back();
  EXPECT_EQ(final.nodes, nodes);
  EXPECT_NE(final.comment.find(" active=0 "), std::string::npos);
  EXPECT_NE(final.comment.find(" ready_eval=0 "), std::string::npos);
  EXPECT_TRUE(final.comment.ends_with(" final"));
}

TEST(Lc5SearchTest, ExactNodeBudgetWithMoreWorkersThanCapacity) {
  for (const int64_t nodes : {0, 1, 7}) {
    SCOPED_TRACE(nodes);
    GameGraph graph;
    ControlledBackend backend;
    RecordingResponder responder;
    auto root = Origin();
    SearchRun run(&graph, nullptr, &backend, &responder, TestSettings(8, 2),
                  root, GoParams{.nodes = nodes},
                  std::chrono::steady_clock::now());
    run.Start();
    run.Wait();
    ExpectDrained(run, graph);
    EXPECT_EQ(run.metrics().visits_admitted.load(), nodes);
    EXPECT_EQ(run.metrics().visits_completed.load(), nodes);
    EXPECT_EQ(run.metrics().visits_cancelled.load(), 0u);
    EXPECT_LE(run.metrics().active_visits_high_water.load(), 2u);
    EXPECT_FALSE(run.Admit(root));
    ExpectFinalOutput(responder, nodes);
    const auto legal = root.history.Last().GetBoard().GenerateLegalMoves();
    EXPECT_NE(
        std::find(legal.begin(), legal.end(), responder.bestmoves[0].bestmove),
        legal.end());
  }
}

TEST(Lc5SearchTest, OwnerBacksUpLeafWhileWaitersResumeSelection) {
  GameGraph graph;
  ControlledBackend backend;
  RecordingResponder responder;
  auto root = Origin();
  SearchRun run(&graph, nullptr, &backend, &responder, TestSettings(), root,
                GoParams{.nodes = 4}, std::chrono::steady_clock::now());
  // One worker consumes all four admissions before submitting the root job.
  for (int i = 0; i < 4; ++i) ASSERT_TRUE(run.Admit(root));
  EXPECT_FALSE(run.Admit(root));
  run.Start();
  run.Wait();
  ExpectDrained(run, graph);
  ExpectFinalOutput(responder, 4);
  EXPECT_EQ(run.metrics().visits_completed.load(), 4u);
  EXPECT_GE(run.metrics().ticket_waiters.load(), 3u);
  EXPECT_GE(run.metrics().visits_resumed.load(), 3u);
  const auto node = graph.SnapshotNode(root.key);
  ASSERT_TRUE(node);
  EXPECT_EQ(node->value.visits, 4u);
  uint64_t edge_visits = 0;
  for (const auto& edge : node->edges) edge_visits += edge.visits;
  // Only the owner backs up at the root leaf. Each waiter must traverse an
  // edge.
  EXPECT_EQ(edge_visits, 3u);
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
                                TerminalKind::kStalemate, 0.0f, 1.0f}}) {
    SCOPED_TRACE(test.fen);
    GameGraph graph;
    ControlledBackend backend;
    RecordingResponder responder;
    const auto root = Origin(test.fen);
    SearchRun run(&graph, nullptr, &backend, &responder, TestSettings(), root,
                  GoParams{.nodes = 4}, std::chrono::steady_clock::now());
    for (int i = 0; i < 4; ++i) ASSERT_TRUE(run.Admit(root));
    run.Start();
    run.Wait();
    ExpectDrained(run, graph);
    ExpectFinalOutput(responder, 4);
    EXPECT_EQ(responder.bestmoves[0].bestmove, Move{});
    EXPECT_EQ(backend.inputs.load(), 0u);
    EXPECT_EQ(backend.computes.load(), 0u);
    EXPECT_EQ(run.metrics().tickets_created.load(), 1u);
    EXPECT_EQ(run.metrics().ticket_waiters.load(), 3u);
    EXPECT_EQ(run.metrics().terminal_visits.load(), 4u);
    EXPECT_EQ(run.metrics().visits_resumed.load(), 0u);
    auto node = graph.SnapshotNode(root.key);
    ASSERT_TRUE(node);
    EXPECT_EQ(node->terminal, test.kind);
    EXPECT_EQ(node->value.visits, 4u);
    EXPECT_FLOAT_EQ(node->value.Q(), test.q);
    EXPECT_FLOAT_EQ(node->value.D(), test.d);
  }
}

TEST(Lc5SearchTest, ImmediateAndMixedResultsKeepJobStorageAlive) {
  for (const auto cache : {ControlledBackend::Cache::kAll,
                           ControlledBackend::Cache::kAlternating}) {
    SCOPED_TRACE(static_cast<int>(cache));
    GameGraph graph;
    ControlledBackend backend(cache);
    RecordingResponder responder;
    const auto root = Origin();
    SearchRun run(&graph, nullptr, &backend, &responder, TestSettings(), root,
                  GoParams{.nodes = 4}, std::chrono::steady_clock::now());
    std::vector<VisitOrigin> origins;
    const auto moves = root.history.Last().GetBoard().GenerateLegalMoves();
    for (int i = 0; i < 4; ++i) {
      auto origin = root;
      origin.history.Append(moves[i]);
      origin.key = MakeNodeKey(origin.history, 7);
      ASSERT_TRUE(run.Admit(origin));
      origins.push_back(std::move(origin));
    }
    EXPECT_FALSE(run.Admit(root));
    run.Start();
    run.Wait();
    ExpectDrained(run, graph);
    ExpectFinalOutput(responder, 4);
    const bool all = cache == ControlledBackend::Cache::kAll;
    EXPECT_EQ(backend.inputs.load(), 4u);
    EXPECT_EQ(backend.computes.load(), all ? 0u : 1u);
    EXPECT_EQ(run.metrics().cache_hits.load(), all ? 4u : 2u);
    EXPECT_EQ(run.metrics().nn_evaluations.load(), all ? 0u : 2u);
    for (const auto& origin : origins) {
      const auto node = graph.SnapshotNode(origin.key);
      ASSERT_TRUE(node);
      EXPECT_EQ(node->value.visits, 1u);
      EXPECT_FLOAT_EQ(node->value.Q(), 0.25f);
      EXPECT_FLOAT_EQ(node->value.D(), 0.5f);
      EXPECT_FLOAT_EQ(node->value.M(), 3.0f);
      for (const auto& edge : node->edges)
        EXPECT_FLOAT_EQ(edge.prior, 1.0f / node->edges.size());
    }
  }
}

TEST(Lc5SearchTest, BlockedBackendStopAndAbortDrainReservationsAndPersistence) {
  for (const bool abort : {false, true}) {
    SCOPED_TRACE(abort);
    GameGraph graph;
    RecordingStore store;
    ControlledBackend backend(ControlledBackend::Cache::kNone, true);
    RecordingResponder responder;
    const auto root = Origin();
    const auto moves = root.history.Last().GetBoard().GenerateLegalMoves();
    // All visits reserve the same root edge and share the child ticket.
    graph.InstallPayload(root.key, 1, {.moves = {moves[0]}, .priors = {1.0f}});
    auto child = root;
    child.history.Append(moves[0]);
    child.key = MakeNodeKey(child.history, 7);
    SearchRun run(&graph, &store, &backend, &responder, TestSettings(), root,
                  GoParams{.nodes = 4}, std::chrono::steady_clock::now());
    for (int i = 0; i < 4; ++i) ASSERT_TRUE(run.Admit(root));
    run.Start();
    const bool entered = backend.WaitForCompute();
    // Always unblock before any fatal assertion or SearchRun destruction.
    if (!entered) {
      run.Abort();
      backend.Release();
      run.Wait();
      FAIL() << "Backend computation was not reached";
    }
    EXPECT_EQ(graph.SnapshotNode(root.key)->edges[0].in_flight, 4u);
    EXPECT_EQ(run.metrics().ticket_waiters.load(), 3u);
    run.Stop();
    if (abort)
      run.Abort();  // Abort must override an uncommitted stop response.
    EXPECT_FALSE(run.Admit(root));
    EXPECT_FALSE(run.Finished());
    EXPECT_EQ(responder.bestmove_count.load(), 0u);
    backend.Release();
    run.Wait();
    ExpectDrained(run, graph);
    EXPECT_EQ(run.metrics().visits_admitted.load(), 4u);
    EXPECT_EQ(run.metrics().visits_completed.load(), 0u);
    EXPECT_EQ(run.metrics().visits_cancelled.load(), 4u);
    EXPECT_EQ(run.metrics().node_store_misses.load(), 1u);
    EXPECT_EQ(store.stored, 1u);
    ASSERT_TRUE(store.entries.contains(child.key));
    EXPECT_FLOAT_EQ(store.entries.at(child.key).leaf_q, 0.25f);
    EXPECT_EQ(graph.SnapshotNode(root.key)->value.visits, 0u);
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

TEST(Lc5SearchTest, StoreMissPersistsAndFreshGraphRehydratesWithoutEvaluation) {
  RecordingStore store;
  ControlledBackend backend;
  const auto root = Origin();
  for (const bool hit : {false, true}) {
    SCOPED_TRACE(hit);
    GameGraph graph;
    RecordingResponder responder;
    SearchRun run(&graph, &store, &backend, &responder, TestSettings(), root,
                  GoParams{.nodes = 1}, std::chrono::steady_clock::now());
    run.Start();
    run.Wait();
    ExpectDrained(run, graph);
    ExpectFinalOutput(responder, 1);
    EXPECT_EQ(run.metrics().node_store_hits.load(), hit ? 1u : 0u);
    EXPECT_EQ(run.metrics().node_store_misses.load(), hit ? 0u : 1u);
    EXPECT_EQ(run.metrics().graph_nodes_rehydrated.load(), hit ? 1u : 0u);
    EXPECT_EQ(run.metrics().node_store_store_batches.load(), hit ? 0u : 1u);
    EXPECT_EQ(backend.inputs.load(), 1u);
    EXPECT_EQ(store.stored, 1u);
    ASSERT_TRUE(store.entries.contains(root.key));
    auto node = graph.SnapshotNode(root.key);
    ASSERT_TRUE(node);
    EXPECT_EQ(node->value.visits, 1u);
    EXPECT_FLOAT_EQ(node->value.Q(), store.entries.at(root.key).leaf_q);
    ASSERT_EQ(node->edges.size(), store.entries.at(root.key).moves.size());
    for (size_t i = 0; i < node->edges.size(); ++i) {
      EXPECT_EQ(node->edges[i].move, store.entries.at(root.key).moves[i]);
      EXPECT_FLOAT_EQ(node->edges[i].prior,
                      store.entries.at(root.key).priors[i]);
    }
  }
}

TEST(Lc5SearchTest, RootBootstrapFlushesDependencyWithoutWaitingForTimer) {
  // A positive one-minute delay cannot be the reason this bootstrap flushes.
  for (const int delay : {0, 60000}) {
    SCOPED_TRACE(delay);
    GameGraph graph;
    ControlledBackend backend(ControlledBackend::Cache::kNone, true);
    RecordingResponder responder;
    const auto root = Origin();
    auto settings = TestSettings(4);
    settings.max_batch_delay_ms = delay;
    SearchRun run(&graph, nullptr, &backend, &responder, settings, root,
                  GoParams{.nodes = 1}, std::chrono::steady_clock::now());
    run.Start();
    const bool entered = backend.WaitForCompute();
    if (entered) {
      // Assert before stop: stop/drain must not be what unblocks collection.
      EXPECT_FALSE(run.Finished());
      EXPECT_EQ(run.metrics().nn_evaluations.load(), 1u);
      EXPECT_EQ(run.metrics().partial_starvation_flushes.load(), 1u);
      EXPECT_EQ(run.metrics().partial_timeout_flushes.load(), 0u);
      EXPECT_EQ(run.metrics().partial_drain_flushes.load(), 0u);
      EXPECT_EQ(responder.bestmove_count.load(), 0u);
    } else {
      run.Abort();
    }
    // Unblock before fatal assertions, including the failure path.
    backend.Release();
    run.Wait();
    ASSERT_TRUE(entered) << "Root bootstrap depended on the collection timer";
    ExpectDrained(run, graph);
    ExpectFinalOutput(responder, 1);
    EXPECT_EQ(backend.batch_sizes, (std::vector<size_t>{1}));
  }
}

TEST(Lc5SearchTest, CollectionFlushesWhileAnotherWorkerIsBlockedInStore) {
  for (const int delay : {0, 1}) {
    SCOPED_TRACE(delay);
    GameGraph graph;
    // First load misses; the other worker's load remains executing. With a
    // positive delay it prevents dependency-starvation detection, so the
    // collector must actually time out. Zero must flush without that timer.
    BlockingStore store(2, false);
    ControlledBackend backend(ControlledBackend::Cache::kNone, true);
    RecordingResponder responder;
    const auto root = Origin();
    auto other = root;
    other.history.Append(
        root.history.Last().GetBoard().GenerateLegalMoves()[0]);
    other.key = MakeNodeKey(other.history, 7);
    auto settings = TestSettings(2, 2);
    settings.max_batch_delay_ms = delay;
    SearchRun run(&graph, &store, &backend, &responder, settings, root,
                  GoParams{.nodes = 2}, std::chrono::steady_clock::now());
    ASSERT_TRUE(run.Admit(root));
    ASSERT_TRUE(run.Admit(other));
    run.Start();
    const bool loading = store.load.WaitForArrival();
    const bool computing = backend.WaitForCompute();
    if (loading && computing) {
      EXPECT_FALSE(run.Finished());
      EXPECT_EQ(run.metrics().nn_evaluations.load(), 1u);
      EXPECT_EQ(run.metrics().partial_timeout_flushes.load(), delay ? 1u : 0u);
      EXPECT_EQ(run.metrics().partial_starvation_flushes.load(),
                delay ? 0u : 1u);
      EXPECT_EQ(run.metrics().partial_drain_flushes.load(), 0u);
    }
    // This also exercises waking collectors and finishing returned store work
    // after stop. Never leave either executor blocked on a failed assertion.
    run.Stop();
    store.Release();
    backend.Release();
    run.Wait();
    ASSERT_TRUE(loading);
    ASSERT_TRUE(computing)
        << "Collection depended on the blocked store returning";
    ExpectDrained(run, graph);
    ExpectFinalOutput(responder, 0);
    EXPECT_EQ(run.metrics().visits_cancelled.load(), 2u);
    EXPECT_EQ(run.metrics().node_store_misses.load(), 2u);
    EXPECT_EQ(store.recording.stored, 2u);
  }
}

TEST(Lc5SearchTest, PositiveDelayCollectsAnotherWorkersStoreMissIntoSameBatch) {
  GameGraph graph;
  BlockingStore store(2, false);
  ControlledBackend backend(ControlledBackend::Cache::kNone, true);
  RecordingResponder responder;
  const auto root = Origin();
  auto other = root;
  other.history.Append(root.history.Last().GetBoard().GenerateLegalMoves()[0]);
  other.key = MakeNodeKey(other.history, 7);
  auto settings = TestSettings(2, 2);
  settings.minibatch_size = 2;
  settings.max_batch_delay_ms = 60000;
  SearchRun run(&graph, &store, &backend, &responder, settings, root,
                GoParams{.nodes = 2}, std::chrono::steady_clock::now());
  ASSERT_TRUE(run.Admit(root));
  ASSERT_TRUE(run.Admit(other));
  run.Start();
  const bool loading = store.load.WaitForArrival();
  const bool first_input = backend.WaitForInputs(1);
  // The first miss has entered AddInput, but the other worker cannot submit
  // its miss until this barrier opens. No helper thread or sleep is needed to
  // prove the collector retains its first job across the store dependency.
  if (!loading || !first_input) run.Abort();
  store.Release();
  const bool computing = backend.WaitForCompute();
  if (!computing) run.Abort();
  backend.Release();
  run.Wait();
  ASSERT_TRUE(loading);
  ASSERT_TRUE(first_input);
  ASSERT_TRUE(computing);
  ExpectDrained(run, graph);
  ExpectFinalOutput(responder, 2);
  EXPECT_EQ(backend.batch_sizes, (std::vector<size_t>{2}));
  EXPECT_EQ(run.metrics().evaluation_batches.load(), 1u);
  EXPECT_EQ(run.metrics().partial_timeout_flushes.load(), 0u);
  EXPECT_EQ(run.metrics().partial_starvation_flushes.load(), 0u);
  EXPECT_EQ(run.metrics().partial_drain_flushes.load(), 0u);
  EXPECT_EQ(store.recording.stored, 2u);
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
      GameGraph graph;
      ControlledBackend backend(ControlledBackend::Cache::kAlternating, true,
                                test.batch_size, test.evaluators);
      RecordingResponder responder;
      const auto root = Origin();
      auto settings = TestSettings(4, test.capacity);
      settings.eval_threads = test.evaluators;
      settings.minibatch_size = test.batch_size;
      settings.max_batch_delay_ms = delay;
      SearchRun run(&graph, nullptr, &backend, &responder, settings, root,
                    GoParams{.nodes = test.capacity},
                    std::chrono::steady_clock::now());
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
          ASSERT_TRUE(run.Admit(origin));
          origins.push_back(std::move(origin));
        }
        if (origins.size() == static_cast<size_t>(test.capacity)) break;
      }
      ASSERT_EQ(origins.size(), static_cast<size_t>(test.capacity));
      EXPECT_FALSE(run.Admit(root));
      run.Start();
      const bool computing = backend.WaitForCompute();
      if (!computing) run.Abort();
      // Release also opens the initial evaluator cohort on failure. Never
      // destroy a run with fixture barriers still blocking its executors.
      backend.Release();
      run.Wait();
      ASSERT_TRUE(computing);
      ExpectDrained(run, graph);
      ExpectFinalOutput(responder, test.capacity);
      EXPECT_EQ(backend.evaluator_ids.size(),
                static_cast<size_t>(test.evaluators));
      EXPECT_EQ(backend.inputs.load(), test.capacity);
      EXPECT_EQ(run.metrics().cache_hits.load(), test.capacity / 2);
      EXPECT_EQ(run.metrics().nn_evaluations.load(), test.capacity / 2);
      EXPECT_EQ(run.metrics().visits_admitted.load(), test.capacity);
      EXPECT_EQ(run.metrics().visits_completed.load(), test.capacity);
      EXPECT_EQ(run.metrics().visits_cancelled.load(), 0u);
      EXPECT_LE(run.metrics().active_visits_high_water.load(), test.capacity);
      EXPECT_EQ(graph.Size(), test.capacity);
      const size_t job_size = (test.batch_size + 3) / 4;
      const size_t credit_jobs =
          std::min<size_t>(16, 2 * test.batch_size / job_size);
      EXPECT_LE(run.metrics().io_jobs_high_water.load(), 4 * credit_jobs);
      if (delay == 0) {
        EXPECT_EQ(run.metrics().partial_timeout_flushes.load(), 0u);
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
        EXPECT_EQ(node->value.visits, 1u);
        EXPECT_FLOAT_EQ(node->value.Q(), 0.25f);
        EXPECT_FLOAT_EQ(node->value.D(), 0.5f);
        EXPECT_FLOAT_EQ(node->value.M(), 3.0f);
        for (const auto& edge : node->edges)
          EXPECT_FLOAT_EQ(edge.prior, 1.0f / node->edges.size());
      }
    }
  }
}

TEST(Lc5SearchTest, BlockedStoreLoadsAndPersistenceDrainAfterStopOrAbort) {
  for (const bool hit : {false, true}) {
    for (const bool abort : {false, true}) {
      SCOPED_TRACE(hit);
      SCOPED_TRACE(abort);
      GameGraph graph;
      BlockingStore store;
      ControlledBackend backend(ControlledBackend::Cache::kAll);
      RecordingResponder responder;
      const auto root = Origin();
      const auto move = root.history.Last().GetBoard().GenerateLegalMoves()[0];
      graph.InstallPayload(root.key, 1, {.moves = {move}, .priors = {1.0f}});
      auto child = root;
      child.history.Append(move);
      child.key = MakeNodeKey(child.history, 7);
      if (hit) {
        const auto legal = child.history.Last().GetBoard().GenerateLegalMoves();
        store.recording.entries.emplace(
            child.key,
            ExpansionPayload{.leaf_q = 0.75f,
                             .leaf_d = 0.25f,
                             .leaf_m = 7.0f,
                             .moves = {legal.begin(), legal.end()},
                             .priors = std::vector<float>(
                                 legal.size(), 1.0f / legal.size())});
      }
      SearchRun run(&graph, &store, &backend, &responder, TestSettings(), root,
                    GoParams{.nodes = 4}, std::chrono::steady_clock::now());
      for (int i = 0; i < 4; ++i) ASSERT_TRUE(run.Admit(root));
      run.Start();
      const bool loading = store.load.WaitForArrival();
      if (loading) {
        EXPECT_EQ(graph.SnapshotNode(root.key)->edges[0].in_flight, 4u);
        EXPECT_EQ(run.metrics().ticket_waiters.load(), 3u);
        EXPECT_EQ(backend.inputs.load(), 0u);
      }
      run.Stop();
      if (abort || !loading) run.Abort();
      EXPECT_FALSE(run.Admit(root));
      EXPECT_FALSE(run.Finished());
      EXPECT_EQ(responder.bestmove_count.load(), 0u);
      store.load.Release();
      // A miss must publish and persist even though all visits were cancelled;
      // a hit must not write back the payload it just loaded.
      const bool persisting = loading && !hit && store.persist.WaitForArrival();
      if (persisting) {
        EXPECT_FALSE(run.Finished());
        EXPECT_EQ(responder.bestmove_count.load(), 0u);
        const auto node = graph.SnapshotNode(child.key);
        EXPECT_TRUE(node);
        if (node) {
          EXPECT_EQ(node->lifecycle, NodeLifecycle::kExpanded);
          EXPECT_EQ(node->ticket, 0u);
          EXPECT_EQ(node->value.visits, 0u);
        }
      }
      store.Release();
      run.Wait();
      ASSERT_TRUE(loading);
      if (!hit) {
        ASSERT_TRUE(persisting);
      }
      ExpectDrained(run, graph);
      EXPECT_EQ(run.metrics().visits_admitted.load(), 4u);
      EXPECT_EQ(run.metrics().visits_completed.load(), 0u);
      EXPECT_EQ(run.metrics().visits_cancelled.load(), 4u);
      EXPECT_EQ(run.metrics().node_store_hits.load(), hit ? 1u : 0u);
      EXPECT_EQ(run.metrics().node_store_misses.load(), hit ? 0u : 1u);
      EXPECT_EQ(store.recording.stored, hit ? 0u : 1u);
      EXPECT_EQ(backend.inputs.load(), hit ? 0u : 1u);
      EXPECT_EQ(backend.computes.load(), 0u);
      ASSERT_TRUE(store.recording.entries.contains(child.key));
      EXPECT_FLOAT_EQ(store.recording.entries.at(child.key).leaf_q,
                      hit ? 0.75f : 0.25f);
      EXPECT_EQ(graph.SnapshotNode(root.key)->value.visits, 0u);
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

TEST(Lc5SearchTest, EngineRetainsGraphAcrossSearchesAndResetsForNewGame) {
  OptionsParser parser;
  Settings::Populate(&parser);
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
    EXPECT_NE(responder.thinking.back().comment.find(" active=0 "),
              std::string::npos);
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
