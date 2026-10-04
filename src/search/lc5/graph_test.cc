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

TEST(Lc5GraphTest, MetadataSnapshotTracksPublicationByValue) {
  GameGraph graph;
  const NodeKey key{123};
  EXPECT_FALSE(graph.SnapshotNodeMetadata(key));

  const auto created = graph.FindOrCreateMaterializing(key, 7);
  const auto materializing = graph.SnapshotNodeMetadata(key);
  ASSERT_TRUE(materializing);
  EXPECT_EQ(materializing->generation, created.generation);
  EXPECT_EQ(materializing->lifecycle, NodeLifecycle::kMaterializing);
  EXPECT_EQ(materializing->terminal, TerminalKind::kNonTerminal);

  const ExpansionPayload payload{.moves = {M(kFileA, kFileA)},
                                 .priors = {1.0f}};
  graph.InstallPayload(key, 7, payload);
  const auto expanded = graph.SnapshotNodeMetadata(key);
  ASSERT_TRUE(expanded);
  EXPECT_EQ(expanded->generation, created.generation);
  EXPECT_EQ(expanded->lifecycle, NodeLifecycle::kExpanded);
  EXPECT_EQ(expanded->terminal, TerminalKind::kNonTerminal);
  // Publication does not change a previously returned value.
  EXPECT_EQ(materializing->lifecycle, NodeLifecycle::kMaterializing);

  const auto selected =
      graph.SelectAndReserve(key, expanded->generation, TestSettings());
  EXPECT_EQ(selected.status, SelectStatus::kSelected);
  const auto full = graph.SnapshotNode(key);
  ASSERT_TRUE(full);
  ASSERT_EQ(full->edges.size(), 1u);
  EXPECT_EQ(full->edges[0].in_flight, 1u);
  EXPECT_EQ(full->generation, expanded->generation);
  EXPECT_EQ(full->lifecycle, expanded->lifecycle);
  EXPECT_EQ(full->terminal, expanded->terminal);
}

TEST(Lc5GraphTest, MetadataSnapshotIncludesExpandedTerminalKind) {
  GameGraph graph;
  const NodeKey key{123};
  const auto created = graph.FindOrCreateMaterializing(key, 7);
  graph.InstallPayload(
      key, 7,
      ExpansionPayload{
          .terminal = TerminalKind::kCheckmate, .moves = {}, .priors = {}});
  const auto terminal = graph.SnapshotNodeMetadata(key);
  ASSERT_TRUE(terminal);
  EXPECT_EQ(terminal->generation, created.generation);
  EXPECT_EQ(terminal->lifecycle, NodeLifecycle::kExpanded);
  EXPECT_EQ(terminal->terminal, TerminalKind::kCheckmate);
  EXPECT_EQ(
      graph.SelectAndReserve(key, terminal->generation, TestSettings()).status,
      SelectStatus::kTerminal);
}

TEST(Lc5GraphTest, MetadataSnapshotPreservesStaleGenerationAfterRecreation) {
  GameGraph graph;
  const NodeKey key{123};
  graph.FindOrCreateMaterializing(key, 7);
  const ExpansionPayload payload{.moves = {M(kFileA, kFileA)},
                                 .priors = {1.0f}};
  graph.InstallPayload(key, 7, payload);
  const auto old = graph.SnapshotNodeMetadata(key);
  ASSERT_TRUE(old);
  ASSERT_TRUE(graph.Erase(key));
  EXPECT_FALSE(graph.SnapshotNodeMetadata(key));

  const auto recreated = graph.FindOrCreateMaterializing(key, 8);
  graph.InstallPayload(key, 8, payload);
  const auto current = graph.SnapshotNodeMetadata(key);
  ASSERT_TRUE(current);
  EXPECT_NE(current->generation, old->generation);
  EXPECT_EQ(current->generation, recreated.generation);
  EXPECT_EQ(current->lifecycle, NodeLifecycle::kExpanded);
  EXPECT_EQ(old->lifecycle, NodeLifecycle::kExpanded);
  const auto stale =
      graph.SelectAndReserve(key, old->generation, TestSettings());
  EXPECT_EQ(stale.status, SelectStatus::kStale);
  EXPECT_EQ(stale.generation, current->generation);
  const auto full = graph.SnapshotNode(key);
  ASSERT_TRUE(full);
  ASSERT_EQ(full->edges.size(), 1u);
  EXPECT_EQ(full->edges[0].in_flight, 0u);
}

TEST(Lc5GraphTest, SizeCountsOnlyNewNodes) {
  GameGraph graph;
  EXPECT_EQ(graph.Size(), 0u);
  EXPECT_TRUE(graph.FindOrCreateMaterializing(NodeKey{1}, 7).created);
  EXPECT_EQ(graph.Size(), 1u);
  EXPECT_FALSE(graph.FindOrCreateMaterializing(NodeKey{1}, 8).created);
  EXPECT_EQ(graph.Size(), 1u);
  EXPECT_TRUE(graph.FindOrCreateMaterializing(NodeKey{2}, 9).created);
  EXPECT_EQ(graph.Size(), 2u);

  ExpansionPayload payload{.moves = {M(kFileA, kFileA)}, .priors = {1.0f}};
  graph.InstallPayload(NodeKey{1}, 8, payload);
  EXPECT_EQ(graph.SnapshotNode(NodeKey{1})->lifecycle,
            NodeLifecycle::kMaterializing);
  EXPECT_EQ(graph.Size(), 2u);
  graph.InstallPayload(NodeKey{1}, 7, payload);
  EXPECT_EQ(graph.SnapshotNode(NodeKey{1})->lifecycle,
            NodeLifecycle::kExpanded);
  EXPECT_EQ(graph.Size(), 2u);
  graph.InstallPayload(NodeKey{1}, 7, payload);
  EXPECT_EQ(graph.Size(), 2u);
  EXPECT_FALSE(graph.FindOrCreateMaterializing(NodeKey{1}, 10).created);
  EXPECT_EQ(graph.Size(), 2u);
}

TEST(Lc5GraphTest, SizeCountsPayloadInsertionAndSuccessfulErasure) {
  GameGraph graph;
  NodeKey key{123};
  ExpansionPayload payload{.moves = {M(kFileA, kFileA)}, .priors = {1.0f}};
  const auto generation = graph.InstallPayload(key, 7, payload);
  EXPECT_EQ(graph.Size(), 1u);
  EXPECT_EQ(graph.InstallPayload(key, 7, payload), generation);
  EXPECT_EQ(graph.Size(), 1u);
  EXPECT_FALSE(graph.Erase(NodeKey{456}));
  EXPECT_EQ(graph.Size(), 1u);
  EXPECT_TRUE(graph.Erase(key));
  EXPECT_EQ(graph.Size(), 0u);
  EXPECT_FALSE(graph.Erase(key));
  EXPECT_EQ(graph.Size(), 0u);
  EXPECT_NE(graph.InstallPayload(key, 8, payload), generation);
  EXPECT_EQ(graph.Size(), 1u);
  EXPECT_TRUE(graph.FindOrCreateMaterializing(NodeKey{456}, 9).created);
  EXPECT_EQ(graph.Size(), 2u);
  EXPECT_TRUE(graph.Erase(NodeKey{456}));
  EXPECT_EQ(graph.Size(), 1u);
}

TEST(Lc5GraphTest, ClearResetsSizeAndAllowsReuse) {
  GameGraph graph;
  graph.Clear();
  EXPECT_EQ(graph.Size(), 0u);
  ExpansionPayload payload{.moves = {M(kFileA, kFileA)}, .priors = {1.0f}};
  // Populate multiple shards with both materializing and expanded nodes.
  for (uint64_t hash = 0; hash < 32; ++hash) {
    if (hash % 2 == 0) {
      graph.FindOrCreateMaterializing(NodeKey{hash}, hash + 1);
    } else {
      graph.InstallPayload(NodeKey{hash}, hash + 1, payload);
    }
  }
  EXPECT_EQ(graph.Size(), 32u);
  EXPECT_EQ(graph.Size(), graph.SnapshotAllForTesting().size());
  graph.Clear();
  EXPECT_EQ(graph.Size(), 0u);
  EXPECT_TRUE(graph.SnapshotAllForTesting().empty());
  graph.Clear();
  EXPECT_EQ(graph.Size(), 0u);
  EXPECT_TRUE(graph.FindOrCreateMaterializing(NodeKey{0}, 33).created);
  EXPECT_EQ(graph.Size(), 1u);
  graph.InstallPayload(NodeKey{1}, 34, payload);
  EXPECT_EQ(graph.Size(), 2u);
  EXPECT_TRUE(graph.Erase(NodeKey{0}));
  EXPECT_EQ(graph.Size(), 1u);
  graph.Clear();
  EXPECT_EQ(graph.Size(), 0u);
}

TEST(Lc5GraphTest, LateMaterializationPreservesPublishedEdges) {
  GameGraph graph;
  const NodeKey key{123};
  const auto created = graph.FindOrCreateMaterializing(key, 7);
  ASSERT_TRUE(created.created);
  const auto stale_snapshot = graph.SnapshotNode(key);
  ASSERT_TRUE(stale_snapshot);
  ASSERT_EQ(stale_snapshot->lifecycle, NodeLifecycle::kMaterializing);

  const Move move = M(kFileA, kFileA);
  const ExpansionPayload payload{.moves = {move}, .priors = {1.0f}};
  graph.InstallPayload(key, 7, payload);
  ASSERT_EQ(graph.SelectAndReserve(key, created.generation, TestSettings()).status,
            SelectStatus::kSelected);
  ASSERT_EQ(graph.CompleteEdge(key, created.generation, move,
                               SearchValue{0.4f, 0.2f, 2.0f}),
            UpdateResult::kApplied);
  ASSERT_EQ(graph.UpdateNodeValue(key, created.generation,
                                  SearchValue{0.4f, 0.2f, 2.0f}),
            UpdateResult::kApplied);
  ASSERT_EQ(graph.SelectAndReserve(key, created.generation, TestSettings()).status,
            SelectStatus::kSelected);

  // A caller with the old materializing snapshot must observe publication,
  // not create a second materialization. Even a late payload cannot reset it.
  const auto late = graph.FindOrCreateMaterializing(key, 8);
  EXPECT_FALSE(late.created);
  EXPECT_EQ(late.lifecycle, NodeLifecycle::kExpanded);
  EXPECT_EQ(late.ticket, 0u);
  EXPECT_EQ(late.generation, created.generation);
  const ExpansionPayload replacement{.moves = {M(kFileB, kFileB)},
                                     .priors = {1.0f}};
  for (const auto ticket : {7u, 8u}) {
    EXPECT_EQ(graph.InstallPayload(key, ticket, replacement), created.generation);
    const auto node = graph.SnapshotNode(key);
    ASSERT_TRUE(node);
    ASSERT_EQ(node->edges.size(), 1u);
    EXPECT_EQ(node->edges[0].move, move);
    EXPECT_FLOAT_EQ(node->edges[0].prior, 1.0f);
    EXPECT_EQ(node->edges[0].visits, 1u);
    EXPECT_EQ(node->edges[0].in_flight, 1u);
    EXPECT_FLOAT_EQ(node->edges[0].Q(), 0.4f);
    EXPECT_EQ(node->value.visits, 1u);
  }
  EXPECT_EQ(graph.CancelEdge(key, created.generation, move),
            UpdateResult::kApplied);
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
