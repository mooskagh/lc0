#include "search/lc5/graph.h"

#include <gtest/gtest.h>

namespace lczero::lc5 {
namespace {

Move M(File from, File to) {
  return Move::White(Square(from, kRank2), Square(to, kRank3));
}

Settings::Resolved TestSettings() {
  return {.threads = 1,
          .eval_threads = 1,
          .minibatch_size = 1,
          .max_active_visits = 8,
          .history_key_length = 7,
          .max_batch_delay_ms = 2,
          .cpuct = 1.0f,
          .cpuct_base = 100.0f,
          .cpuct_factor = 0.0f,
          .fpu_strategy = FpuStrategy::kAbsolute,
          .fpu_value = 0.0f};
}

TEST(Lc5ValueTest, PerspectiveAndSumsAreExact) {
  SearchValue child{0.25f, 0.5f, 3.0f};
  SearchValue parent = child.Parent();
  EXPECT_FLOAT_EQ(parent.q, -0.25f);
  EXPECT_FLOAT_EQ(parent.d, 0.5f);
  EXPECT_FLOAT_EQ(parent.m, 4.0f);
  ValueStats stats;
  stats.Add(0.25f, 0.5f, 3.0f);
  stats.Add(-0.25f, 0.25f, 5.0f);
  EXPECT_EQ(stats.visits, 2u);
  EXPECT_FLOAT_EQ(stats.Q(), 0.0f);
  EXPECT_FLOAT_EQ(stats.D(), 0.375f);
  EXPECT_FLOAT_EQ(stats.M(), 4.0f);
}

TEST(Lc5GraphTest, GenerationChecksProtectRecreatedNodes) {
  GameGraph graph;
  NodeKey key{123};
  auto first = graph.FindOrCreateMaterializing(key, 7);
  ASSERT_TRUE(first.created);
  ExpansionPayload payload{.moves = {M(kFileA, kFileA)}, .priors = {1.0f}};
  EXPECT_EQ(graph.InstallPayload(key, 7, payload), first.generation);
  EXPECT_TRUE(graph.Erase(key));
  auto second = graph.FindOrCreateMaterializing(key, 8);
  EXPECT_NE(first.generation, second.generation);
  EXPECT_EQ(graph.UpdateNodeValue(key, first.generation, {}),
            UpdateResult::kStale);
  EXPECT_EQ(graph.UpdateNodeValue(key, second.generation, {}),
            UpdateResult::kApplied);
}

TEST(Lc5GraphTest, ReservationsAreExactAndCancellationDoesNotCommit) {
  GameGraph graph;
  NodeKey key{456};
  auto created = graph.FindOrCreateMaterializing(key, 1);
  Move first = M(kFileA, kFileA);
  Move second = M(kFileB, kFileB);
  ExpansionPayload payload{.moves = {first, second},
                           .priors = {0.5f, 0.5f}};
  graph.InstallPayload(key, 1, payload);
  auto settings = TestSettings();
  auto selection1 =
      graph.SelectAndReserve(key, created.generation, settings);
  auto selection2 =
      graph.SelectAndReserve(key, created.generation, settings);
  EXPECT_EQ(selection1.move, first);
  EXPECT_EQ(selection2.move, second);
  EXPECT_EQ(graph.CancelEdge(key, created.generation, first),
            UpdateResult::kApplied);
  EXPECT_EQ(graph.CompleteEdge(key, created.generation, second,
                               SearchValue{0.4f, 0.2f, 2.0f}),
            UpdateResult::kApplied);
  auto node = graph.SnapshotNode(key);
  ASSERT_TRUE(node);
  EXPECT_EQ(node->edges[0].visits, 0u);
  EXPECT_EQ(node->edges[0].in_flight, 0u);
  EXPECT_EQ(node->edges[1].visits, 1u);
  EXPECT_FLOAT_EQ(node->edges[1].Q(), 0.4f);
  EXPECT_EQ(graph.CancelEdge(key, created.generation, first),
            UpdateResult::kUnderflow);
}

TEST(Lc5GraphTest, IncomingParentsKeepIndependentEdgeStatistics) {
  GameGraph graph;
  Move move = M(kFileC, kFileC);
  ExpansionPayload payload{.moves = {move}, .priors = {1.0f}};
  auto p1 = graph.FindOrCreateMaterializing(NodeKey{1}, 1);
  graph.FindOrCreateMaterializing(NodeKey{2}, 2);
  graph.InstallPayload(NodeKey{1}, 1, payload);
  graph.InstallPayload(NodeKey{2}, 2, payload);
  auto settings = TestSettings();
  graph.SelectAndReserve(NodeKey{1}, p1.generation, settings);
  graph.CompleteEdge(NodeKey{1}, p1.generation, move, {});
  EXPECT_EQ(graph.SnapshotNode(NodeKey{1})->edges[0].visits, 1u);
  EXPECT_EQ(graph.SnapshotNode(NodeKey{2})->edges[0].visits, 0u);
}

}  // namespace
}  // namespace lczero::lc5

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
