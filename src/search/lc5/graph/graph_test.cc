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

#include "search/lc5/graph/graph.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "search/lc5/graph/node_record.h"
#include "search/lc5/policies/puct.h"
#include "search/lc5/policies/puct_dag.h"

namespace lczero::lc5 {
namespace {

Move M(File from, File to) {
  return Move::White(Square(from, kRank2), Square(to, kRank3));
}

using Graph = GameGraph<PuctAveraging>;

// The statistics of edge `i`; an untried edge has none, so it reads as empty.
const TriedEdge<PuctAveraging>& TriedAt(const NodeState<PuctAveraging>& node,
                                        size_t i) {
  static const TriedEdge<PuctAveraging> kUntried;
  return i < node.tried.size() ? node.tried[i] : kUntried;
}

PuctAveraging::Params TestParams() {
  return {.cpuct = 1.0f,
          .cpuct_base = 100.0f,
          .cpuct_factor = 0.0f,
          .fpu_strategy = FpuStrategy::kAbsolute,
          .fpu_value = 0.0f};
}

TEST(Lc5GraphTest, PackedPriorRoundsMonotonically) {
  for (const float exact : {0.0f, 1.0f, 0.5f, 0.25f}) {
    EXPECT_EQ(float(PackedPrior(exact)), exact);
  }
  EXPECT_NEAR(float(PackedPrior(1e-6f)), 1e-6f, 6e-8f);
  EXPECT_NEAR(float(PackedPrior(0.3f)), 0.3f, 0.3f / 1024);
  float previous = 0.0f;
  for (int i = 0; i <= 100000; ++i) {
    const float packed = PackedPrior(i / 100000.0f);
    EXPECT_GE(packed, previous);
    previous = packed;
  }
}

TEST(Lc5GraphTest, CancellationOnlyErasesMatchingPlaceholder) {
  Graph graph;
  const NodeKey key{123};
  graph.FindOrCreateMaterialization(key, 7);
  EXPECT_FALSE(graph.CancelMaterialization(key, 8));
  EXPECT_TRUE(graph.CancelMaterialization(key, 7));
  EXPECT_EQ(graph.Size(), 0u);
  graph.FindOrCreateMaterialization(key, 8);
  EXPECT_FALSE(graph.CancelMaterialization(key, 7));
  graph.CompleteMaterialization(
      key, 8, {.moves = {M(kFileA, kFileA)}, .priors = {1.0f}});
  EXPECT_FALSE(graph.CancelMaterialization(key, 8));
  ASSERT_TRUE(graph.SnapshotNode(key));
  EXPECT_EQ(graph.SnapshotNode(key)->lifecycle, NodeLifecycle::kExpanded);
}

TEST(Lc5ValueTest, PerspectiveAndMeans) {
  SearchValue child{0.25f, 0.5f, 3.0f};
  SearchValue parent = child.Parent();
  EXPECT_FLOAT_EQ(parent.q, -0.25f);
  EXPECT_FLOAT_EQ(parent.d, 0.5f);
  EXPECT_FLOAT_EQ(parent.m, 4.0f);
  ValueStats stats;
  stats.Add({0.25f, 0.5f, 3.0f});
  stats.Add({-0.25f, 0.25f, 5.0f});
  EXPECT_EQ(stats.visits, 2u);
  EXPECT_FLOAT_EQ(stats.Q(), 0.0f);
  EXPECT_FLOAT_EQ(stats.D(), 0.375f);
  EXPECT_FLOAT_EQ(stats.M(), 4.0f);
}

TEST(Lc5ValueTest, RunningMeansStayAccurate) {
  ValueStats empty;
  EXPECT_EQ(empty.Q(), 0.0f);
  EXPECT_EQ(empty.D(), 0.0f);
  EXPECT_EQ(empty.M(), 0.0f);

  ValueStats constant;
  for (int i = 0; i < 100000; ++i) constant.Add({0.3f, 0.2f, 7.0f});
  EXPECT_EQ(constant.visits, 100000u);
  EXPECT_EQ(constant.Q(), 0.3f);
  EXPECT_EQ(constant.D(), 0.2f);
  EXPECT_EQ(constant.M(), 7.0f);

  // Q cycles through -0.5, 0, 0.5, 1 with mean 0.25.
  ValueStats sequence;
  for (int i = 0; i < 4000; ++i) {
    sequence.Add({-0.5f + 0.5f * (i % 4), 0.0f, static_cast<float>(i % 3)});
  }
  EXPECT_NEAR(sequence.Q(), 0.25f, 1e-5f);
  EXPECT_NEAR(sequence.M(), 3999.0f / 4000.0f, 1e-5f);
}

TEST(Lc5GraphTest, PublicationAcceptanceIsIndependentOfReturnedGeneration) {
  Graph graph;
  const NodeKey key{123};
  const Move move = M(kFileA, kFileA);
  const ExpansionPayload payload{.moves = {move}, .priors = {1.0f}};
  const ExpansionPayload replacement{.moves = {M(kFileB, kFileB)},
                                     .priors = {0.5f}};
  const auto created = graph.FindOrCreateMaterialization(key, 7);

  auto installed = graph.CompleteMaterialization(key, 8, replacement);
  EXPECT_EQ(installed.generation, created.generation);
  EXPECT_FALSE(installed.accepted);
  auto node = graph.SnapshotNode(key);
  ASSERT_TRUE(node);
  EXPECT_EQ(node->lifecycle, NodeLifecycle::kMaterializing);
  EXPECT_EQ(node->ticket, 7u);
  EXPECT_TRUE(node->edges.empty());

  installed = graph.CompleteMaterialization(key, 7, payload);
  EXPECT_EQ(installed.generation, created.generation);
  EXPECT_TRUE(installed.accepted);
  ASSERT_EQ(graph.SelectAndReserve(key, TestParams()).status,
            SelectStatus::kSelected);
  ASSERT_EQ(
      graph.BackupNode(key, created.generation, {0.25f, 0.5f, 3.0f}, move).edge,
      UpdateResult::kApplied);
  ASSERT_EQ(graph.SelectAndReserve(key, TestParams()).status,
            SelectStatus::kSelected);
  // Both duplicate and wrong-ticket publications return the same generation
  // as success; only `accepted` distinguishes them. Rejection must not
  // disturb statistics or live reservations.
  for (const auto ticket : {7u, 8u}) {
    installed = graph.CompleteMaterialization(key, ticket, replacement);
    EXPECT_EQ(installed.generation, created.generation);
    EXPECT_FALSE(installed.accepted);
    node = graph.SnapshotNode(key);
    ASSERT_TRUE(node);
    EXPECT_EQ(node->lifecycle, NodeLifecycle::kExpanded);
    EXPECT_EQ(node->ticket, 0u);
    ASSERT_EQ(node->edges.size(), 1u);
    EXPECT_EQ(node->edges[0].move, move);
    EXPECT_FLOAT_EQ(node->edges[0].prior, 1.0f);
    EXPECT_EQ(TriedAt(*node, 0).stats.visits, 1u);
    EXPECT_EQ(TriedAt(*node, 0).in_flight, 1u);
    EXPECT_FLOAT_EQ(TriedAt(*node, 0).stats.Q(), 0.25f);
    EXPECT_EQ(node->stats.visits, 2u);
  }
  EXPECT_EQ(graph.CancelEdge(key, created.generation, move),
            UpdateResult::kApplied);

  ASSERT_TRUE(graph.Erase(key));
  const auto recreated = graph.FindOrCreateMaterialization(key, 9);
  ASSERT_NE(recreated.generation, created.generation);
  installed = graph.CompleteMaterialization(key, 7, payload);
  EXPECT_EQ(installed.generation, recreated.generation);
  EXPECT_FALSE(installed.accepted);
  EXPECT_EQ(graph.SnapshotNode(key)->ticket, 9u);
  installed = graph.CompleteMaterialization(key, 9, replacement);
  EXPECT_EQ(installed.generation, recreated.generation);
  EXPECT_TRUE(installed.accepted);

  // The insertion path also reports acceptance, not just existing tickets.
  EXPECT_TRUE(
      graph.CompleteMaterialization(NodeKey{456}, 10, payload).accepted);
  EXPECT_EQ(graph.Size(), 2u);
}

TEST(Lc5GraphTest, SelectionReportsLifecycleAndGeneration) {
  Graph graph;
  const NodeKey key{123};
  EXPECT_EQ(graph.SelectAndReserve(key, TestParams()).status,
            SelectStatus::kMissing);

  const auto created = graph.FindOrCreateMaterialization(key, 7);
  const auto materializing = graph.SelectAndReserve(key, TestParams());
  EXPECT_EQ(materializing.status, SelectStatus::kMaterializing);
  EXPECT_EQ(materializing.generation, created.generation);
  EXPECT_EQ(materializing.terminal, TerminalKind::kNonTerminal);

  const ExpansionPayload payload{.moves = {M(kFileA, kFileA)},
                                 .priors = {1.0f}};
  graph.CompleteMaterialization(key, 7, payload);
  const auto selected = graph.SelectAndReserve(key, TestParams());
  EXPECT_EQ(selected.status, SelectStatus::kSelected);
  EXPECT_EQ(selected.move, M(kFileA, kFileA));
  EXPECT_EQ(selected.generation, created.generation);
  const auto full = graph.SnapshotNode(key);
  ASSERT_TRUE(full);
  ASSERT_EQ(full->edges.size(), 1u);
  EXPECT_EQ(TriedAt(*full, 0).in_flight, 1u);
  EXPECT_EQ(full->generation, created.generation);
  EXPECT_EQ(full->lifecycle, NodeLifecycle::kExpanded);
  EXPECT_EQ(full->terminal, TerminalKind::kNonTerminal);
}

TEST(Lc5GraphTest, SelectionReportsExpandedTerminalKind) {
  Graph graph;
  const NodeKey key{123};
  const auto created = graph.FindOrCreateMaterialization(key, 7);
  graph.CompleteMaterialization(
      key, 7,
      ExpansionPayload{
          .terminal = TerminalKind::kCheckmate, .moves = {}, .priors = {}});
  const auto node = graph.SnapshotNode(key);
  ASSERT_TRUE(node);
  EXPECT_EQ(node->lifecycle, NodeLifecycle::kExpanded);
  EXPECT_EQ(node->terminal, TerminalKind::kCheckmate);
  const auto terminal = graph.SelectAndReserve(key, TestParams());
  EXPECT_EQ(terminal.status, SelectStatus::kTerminal);
  EXPECT_EQ(terminal.generation, created.generation);
  EXPECT_EQ(terminal.terminal, TerminalKind::kCheckmate);
}

TEST(Lc5GraphTest, ProvenBoundsGoUpAndStopSelectionWhenAsked) {
  Graph graph;
  const NodeKey parent{1};
  const NodeKey child{2};
  const Move move = M(kFileA, kFileA);
  const auto parent_created = graph.FindOrCreateMaterialization(parent, 1);
  graph.CompleteMaterialization(parent, 1, {.moves = {move}, .priors = {1.0f}});
  const auto child_created = graph.FindOrCreateMaterialization(child, 2);
  graph.CompleteMaterialization(
      child, 2,
      {.terminal = TerminalKind::kCheckmate,
       .leaf = TerminalValue(TerminalKind::kCheckmate),
       .moves = {},
       .priors = {}});
  // The install proves the checkmate; the leaf's own backup keeps it.
  ASSERT_EQ(graph.SelectAndReserve(parent, TestParams()).move, move);
  const auto from_child = graph.BackupNode(
      child, child_created.generation,
      PuctAveraging::Leaf(TerminalValue(TerminalKind::kCheckmate), true),
      std::nullopt);
  ASSERT_TRUE(from_child.to_parent);
  EXPECT_EQ(from_child.to_parent->bounds,
            (Bounds{.lower = 1, .upper = 1, .plies = 1}));
  const auto up = graph.BackupNode(parent, parent_created.generation,
                                   *from_child.to_parent, move);
  ASSERT_TRUE(up.to_parent);
  EXPECT_EQ(up.to_parent->bounds,
            (Bounds{.lower = -1, .upper = -1, .plies = 2}));
  EXPECT_EQ(graph.SnapshotNode(parent)->stats.bounds,
            (Bounds{.lower = 1, .upper = 1, .plies = 1}));

  // A proven node stops selection only with StickyEndgames and when asked.
  auto params = TestParams();
  const auto select = [&](bool stop_at_proven) {
    const auto selected =
        graph.SelectAndReserve(parent, params, stop_at_proven);
    if (selected.status == SelectStatus::kSelected) {
      graph.CancelEdge(parent, parent_created.generation, move);
    }
    return selected;
  };
  EXPECT_EQ(select(true).status, SelectStatus::kSelected);
  params.sticky_endgames = true;
  EXPECT_EQ(select(false).status, SelectStatus::kSelected);
  const auto proven = select(true);
  EXPECT_EQ(proven.status, SelectStatus::kProven);
  EXPECT_EQ(proven.generation, parent_created.generation);
  EXPECT_EQ(proven.value.q, 1.0f);
  EXPECT_EQ(proven.value.m, 1.0f);
  EXPECT_EQ(graph.SnapshotNode(parent)->tried[0].in_flight, 0u);
}

TEST(Lc5GraphTest, SelectionReturnsNewGenerationAfterRecreation) {
  Graph graph;
  const NodeKey key{123};
  const auto created = graph.FindOrCreateMaterialization(key, 7);
  const ExpansionPayload payload{.moves = {M(kFileA, kFileA)},
                                 .priors = {1.0f}};
  graph.CompleteMaterialization(key, 7, payload);
  ASSERT_TRUE(graph.Erase(key));
  EXPECT_FALSE(graph.SnapshotNode(key));

  const auto recreated = graph.FindOrCreateMaterialization(key, 8);
  ASSERT_NE(recreated.generation, created.generation);
  graph.CompleteMaterialization(key, 8, payload);
  const auto selected = graph.SelectAndReserve(key, TestParams());
  EXPECT_EQ(selected.status, SelectStatus::kSelected);
  EXPECT_EQ(selected.generation, recreated.generation);
  // Updates computed against the erased node are still rejected.
  EXPECT_EQ(graph.CancelEdge(key, created.generation, M(kFileA, kFileA)),
            UpdateResult::kStale);
  EXPECT_EQ(TriedAt(*graph.SnapshotNode(key), 0).in_flight, 1u);
}

TEST(Lc5GraphTest, SizeCountsOnlyNewNodes) {
  Graph graph;
  EXPECT_EQ(graph.Size(), 0u);
  EXPECT_TRUE(graph.FindOrCreateMaterialization(NodeKey{1}, 7).created);
  EXPECT_EQ(graph.Size(), 1u);
  EXPECT_FALSE(graph.FindOrCreateMaterialization(NodeKey{1}, 8).created);
  EXPECT_EQ(graph.Size(), 1u);
  EXPECT_TRUE(graph.FindOrCreateMaterialization(NodeKey{2}, 9).created);
  EXPECT_EQ(graph.Size(), 2u);

  ExpansionPayload payload{.moves = {M(kFileA, kFileA)}, .priors = {1.0f}};
  graph.CompleteMaterialization(NodeKey{1}, 8, payload);
  EXPECT_EQ(graph.SnapshotNode(NodeKey{1})->lifecycle,
            NodeLifecycle::kMaterializing);
  EXPECT_EQ(graph.Size(), 2u);
  graph.CompleteMaterialization(NodeKey{1}, 7, payload);
  EXPECT_EQ(graph.SnapshotNode(NodeKey{1})->lifecycle,
            NodeLifecycle::kExpanded);
  EXPECT_EQ(graph.Size(), 2u);
  graph.CompleteMaterialization(NodeKey{1}, 7, payload);
  EXPECT_EQ(graph.Size(), 2u);
  EXPECT_FALSE(graph.FindOrCreateMaterialization(NodeKey{1}, 10).created);
  EXPECT_EQ(graph.Size(), 2u);
}

TEST(Lc5GraphTest, SizeCountsPayloadInsertionAndSuccessfulErasure) {
  Graph graph;
  NodeKey key{123};
  ExpansionPayload payload{.moves = {M(kFileA, kFileA)}, .priors = {1.0f}};
  const auto generation =
      graph.CompleteMaterialization(key, 7, payload).generation;
  EXPECT_EQ(graph.Size(), 1u);
  EXPECT_EQ(graph.CompleteMaterialization(key, 7, payload).generation,
            generation);
  EXPECT_EQ(graph.Size(), 1u);
  EXPECT_FALSE(graph.Erase(NodeKey{456}));
  EXPECT_EQ(graph.Size(), 1u);
  EXPECT_TRUE(graph.Erase(key));
  EXPECT_EQ(graph.Size(), 0u);
  EXPECT_FALSE(graph.Erase(key));
  EXPECT_EQ(graph.Size(), 0u);
  EXPECT_NE(graph.CompleteMaterialization(key, 8, payload).generation,
            generation);
  EXPECT_EQ(graph.Size(), 1u);
  EXPECT_TRUE(graph.FindOrCreateMaterialization(NodeKey{456}, 9).created);
  EXPECT_EQ(graph.Size(), 2u);
  EXPECT_TRUE(graph.Erase(NodeKey{456}));
  EXPECT_EQ(graph.Size(), 1u);
}

TEST(Lc5GraphTest, ClearResetsSizeAndAllowsReuse) {
  Graph graph;
  graph.Clear();
  EXPECT_EQ(graph.Size(), 0u);
  ExpansionPayload payload{.moves = {M(kFileA, kFileA)}, .priors = {1.0f}};
  // Populate multiple shards with both materializing and expanded nodes.
  for (uint64_t hash = 0; hash < 32; ++hash) {
    if (hash % 2 == 0) {
      graph.FindOrCreateMaterialization(NodeKey{hash}, hash + 1);
    } else {
      graph.CompleteMaterialization(NodeKey{hash}, hash + 1, payload);
    }
  }
  EXPECT_EQ(graph.Size(), 32u);
  EXPECT_EQ(graph.Size(), graph.SnapshotAllForTesting().size());
  graph.Clear();
  EXPECT_EQ(graph.Size(), 0u);
  EXPECT_TRUE(graph.SnapshotAllForTesting().empty());
  graph.Clear();
  EXPECT_EQ(graph.Size(), 0u);
  EXPECT_TRUE(graph.FindOrCreateMaterialization(NodeKey{0}, 33).created);
  EXPECT_EQ(graph.Size(), 1u);
  graph.CompleteMaterialization(NodeKey{1}, 34, payload);
  EXPECT_EQ(graph.Size(), 2u);
  EXPECT_TRUE(graph.Erase(NodeKey{0}));
  EXPECT_EQ(graph.Size(), 1u);
  graph.Clear();
  EXPECT_EQ(graph.Size(), 0u);
}

TEST(Lc5GraphTest, LateMaterializationPreservesPublishedEdges) {
  Graph graph;
  const NodeKey key{123};
  const auto created = graph.FindOrCreateMaterialization(key, 7);
  ASSERT_TRUE(created.created);
  const auto stale_snapshot = graph.SnapshotNode(key);
  ASSERT_TRUE(stale_snapshot);
  ASSERT_EQ(stale_snapshot->lifecycle, NodeLifecycle::kMaterializing);

  const Move move = M(kFileA, kFileA);
  const ExpansionPayload payload{.moves = {move}, .priors = {1.0f}};
  graph.CompleteMaterialization(key, 7, payload);
  ASSERT_EQ(graph.SelectAndReserve(key, TestParams()).status,
            SelectStatus::kSelected);
  ASSERT_EQ(
      graph.BackupNode(key, created.generation, {{0.4f, 0.2f, 2.0f}}, move)
          .edge,
      UpdateResult::kApplied);
  ASSERT_EQ(graph.SelectAndReserve(key, TestParams()).status,
            SelectStatus::kSelected);

  // A caller with the old materializing snapshot must observe publication,
  // not create a second materialization. Even a late payload cannot reset it.
  const auto late = graph.FindOrCreateMaterialization(key, 8);
  EXPECT_FALSE(late.created);
  EXPECT_EQ(late.lifecycle, NodeLifecycle::kExpanded);
  EXPECT_EQ(late.ticket, 0u);
  EXPECT_EQ(late.generation, created.generation);
  const ExpansionPayload replacement{.moves = {M(kFileB, kFileB)},
                                     .priors = {1.0f}};
  for (const auto ticket : {7u, 8u}) {
    EXPECT_EQ(
        graph.CompleteMaterialization(key, ticket, replacement).generation,
        created.generation);
    const auto node = graph.SnapshotNode(key);
    ASSERT_TRUE(node);
    ASSERT_EQ(node->edges.size(), 1u);
    EXPECT_EQ(node->edges[0].move, move);
    EXPECT_FLOAT_EQ(node->edges[0].prior, 1.0f);
    EXPECT_EQ(TriedAt(*node, 0).stats.visits, 1u);
    EXPECT_EQ(TriedAt(*node, 0).in_flight, 1u);
    EXPECT_FLOAT_EQ(TriedAt(*node, 0).stats.Q(), 0.4f);
    EXPECT_EQ(node->stats.visits, 2u);
  }
  EXPECT_EQ(graph.CancelEdge(key, created.generation, move),
            UpdateResult::kApplied);
}

TEST(Lc5GraphTest, AcceptedInstallCountsLeafValueOnce) {
  Graph graph;
  const NodeKey key{321};
  const ExpansionPayload payload{.leaf = {0.5f, 0.25f, 6.0f},
                                 .moves = {M(kFileA, kFileA)},
                                 .priors = {1.0f}};
  graph.FindOrCreateMaterialization(key, 7);
  EXPECT_FALSE(graph.CompleteMaterialization(key, 8, payload).accepted);
  auto node = graph.SnapshotNode(key);
  ASSERT_TRUE(node);
  EXPECT_EQ(node->stats.visits, 0u);

  EXPECT_TRUE(graph.CompleteMaterialization(key, 7, payload).accepted);
  node = graph.SnapshotNode(key);
  ASSERT_TRUE(node);
  EXPECT_EQ(node->stats.visits, 1u);
  EXPECT_FLOAT_EQ(node->stats.Q(), 0.5f);
  EXPECT_FLOAT_EQ(node->stats.D(), 0.25f);
  EXPECT_FLOAT_EQ(node->stats.M(), 6.0f);

  // Duplicate and wrong-ticket publications do not count the leaf again.
  for (const auto ticket : {7u, 8u}) {
    EXPECT_FALSE(graph.CompleteMaterialization(key, ticket, payload).accepted);
    node = graph.SnapshotNode(key);
    ASSERT_TRUE(node);
    EXPECT_EQ(node->stats.visits, 1u);
    EXPECT_FLOAT_EQ(node->stats.Q(), 0.5f);
  }
}

TEST(Lc5GraphTest, GenerationChecksProtectRecreatedNodes) {
  Graph graph;
  NodeKey key{123};
  auto first = graph.FindOrCreateMaterialization(key, 7);
  ASSERT_TRUE(first.created);
  ExpansionPayload payload{.moves = {M(kFileA, kFileA)}, .priors = {1.0f}};
  EXPECT_EQ(graph.CompleteMaterialization(key, 7, payload).generation,
            first.generation);
  EXPECT_TRUE(graph.Erase(key));
  auto second = graph.FindOrCreateMaterialization(key, 8);
  EXPECT_NE(first.generation, second.generation);
  const auto stale = graph.BackupNode(key, first.generation, {}, std::nullopt);
  EXPECT_EQ(stale.node, UpdateResult::kStale);
  EXPECT_FALSE(stale.to_parent);
  EXPECT_EQ(graph.BackupNode(key, second.generation, {}, std::nullopt).node,
            UpdateResult::kApplied);
}

TEST(Lc5GraphTest, BackupCommitsNodeAndOptionalEdgeSums) {
  Graph graph;
  NodeKey key{456};
  Move move = M(kFileA, kFileA);
  auto created = graph.FindOrCreateMaterialization(key, 1);
  graph.CompleteMaterialization(key, 1, {.moves = {move}, .priors = {1.0f}});
  auto result = graph.BackupNode(key, created.generation, {0.25f, 0.5f, 3.0f},
                                 std::nullopt);
  EXPECT_EQ(result.node, UpdateResult::kApplied);
  EXPECT_EQ(result.edge, UpdateResult::kApplied);
  auto node = graph.SnapshotNode(key);
  ASSERT_TRUE(node);
  EXPECT_EQ(node->stats.visits, 2u);
  EXPECT_FLOAT_EQ(node->stats.Q(), 0.125f);
  EXPECT_FLOAT_EQ(node->stats.D(), 0.25f);
  EXPECT_FLOAT_EQ(node->stats.M(), 1.5f);
  EXPECT_EQ(TriedAt(*node, 0).stats.visits, 0u);
  ASSERT_EQ(graph.SelectAndReserve(key, TestParams()).status,
            SelectStatus::kSelected);
  result =
      graph.BackupNode(key, created.generation, {-0.5f, 0.25f, 4.0f}, move);
  EXPECT_EQ(result.node, UpdateResult::kApplied);
  EXPECT_EQ(result.edge, UpdateResult::kApplied);
  ASSERT_TRUE(result.to_parent);
  EXPECT_FLOAT_EQ(result.to_parent->value.q, 0.5f);
  EXPECT_FLOAT_EQ(result.to_parent->value.d, 0.25f);
  EXPECT_FLOAT_EQ(result.to_parent->value.m, 5.0f);
  node = graph.SnapshotNode(key);
  ASSERT_TRUE(node);
  EXPECT_EQ(node->stats.visits, 3u);
  EXPECT_FLOAT_EQ(node->stats.Q(), -0.25f / 3);
  EXPECT_FLOAT_EQ(node->stats.D(), 0.25f);
  EXPECT_FLOAT_EQ(node->stats.M(), 7.0f / 3);
  EXPECT_EQ(TriedAt(*node, 0).stats.visits, 1u);
  EXPECT_EQ(TriedAt(*node, 0).in_flight, 0u);
  EXPECT_FLOAT_EQ(TriedAt(*node, 0).stats.Q(), -0.5f);
  EXPECT_FLOAT_EQ(TriedAt(*node, 0).stats.D(), 0.25f);
  EXPECT_FLOAT_EQ(TriedAt(*node, 0).stats.M(), 4.0f);
}

TEST(Lc5GraphTest, BackupCanSendTheNodesCurrentValue) {
  Graph graph;
  NodeKey key{457};
  Move move = M(kFileA, kFileA);
  auto created = graph.FindOrCreateMaterialization(key, 1);
  // The install counts the leaf value once: Q 0.25, D 0.5, M 2.
  graph.CompleteMaterialization(key, 1,
                                {.leaf = {.q = 0.25f, .d = 0.5f, .m = 2.0f},
                                 .moves = {move},
                                 .priors = {1.0f}});
  ASSERT_EQ(graph.SelectAndReserve(key, TestParams()).status,
            SelectStatus::kSelected);
  const auto result = graph.BackupNode(key, created.generation,
                                       {-0.75f, 0.0f, 4.0f}, move, true);
  EXPECT_EQ(result.node, UpdateResult::kApplied);
  EXPECT_EQ(result.edge, UpdateResult::kApplied);
  const auto node = graph.SnapshotNode(key);
  ASSERT_TRUE(node);
  EXPECT_EQ(node->stats.visits, 2u);
  EXPECT_FLOAT_EQ(node->stats.Q(), -0.25f);
  EXPECT_FLOAT_EQ(node->stats.D(), 0.25f);
  EXPECT_FLOAT_EQ(node->stats.M(), 3.0f);
  EXPECT_EQ(TriedAt(*node, 0).stats.visits, 1u);
  EXPECT_EQ(TriedAt(*node, 0).in_flight, 0u);
  EXPECT_FLOAT_EQ(TriedAt(*node, 0).stats.Q(), -0.75f);
  // The parent receives the node's new mean, not the backed-up value.
  ASSERT_TRUE(result.to_parent);
  EXPECT_FLOAT_EQ(result.to_parent->value.q, 0.25f);
  EXPECT_FLOAT_EQ(result.to_parent->value.d, 0.25f);
  EXPECT_FLOAT_EQ(result.to_parent->value.m, 4.0f);
  EXPECT_EQ(result.to_parent->bounds, Bounds{});
}

TEST(Lc5GraphTest, BackupRejectsMissingAndStaleGenerations) {
  Graph graph;
  NodeKey key{456};
  Move move = M(kFileA, kFileA);
  auto first = graph.FindOrCreateMaterialization(key, 1);
  ASSERT_TRUE(graph.Erase(key));
  auto result =
      graph.BackupNode(key, first.generation, {0.25f, 0.5f, 3.0f}, move);
  EXPECT_EQ(result.node, UpdateResult::kMissing);
  EXPECT_EQ(result.edge, UpdateResult::kMissing);
  EXPECT_FALSE(graph.SnapshotNode(key));
  auto second = graph.FindOrCreateMaterialization(key, 2);
  ASSERT_NE(first.generation, second.generation);
  graph.CompleteMaterialization(key, 2, {.moves = {move}, .priors = {1.0f}});
  ASSERT_EQ(graph.SelectAndReserve(key, TestParams()).status,
            SelectStatus::kSelected);
  result = graph.BackupNode(key, first.generation, {0.25f, 0.5f, 3.0f}, move);
  EXPECT_EQ(result.node, UpdateResult::kStale);
  EXPECT_EQ(result.edge, UpdateResult::kStale);
  result = graph.BackupNode(key, first.generation, {}, std::nullopt);
  EXPECT_EQ(result.node, UpdateResult::kStale);
  EXPECT_EQ(result.edge, UpdateResult::kApplied);
  auto node = graph.SnapshotNode(key);
  ASSERT_TRUE(node);
  EXPECT_EQ(node->stats.visits, 1u);
  EXPECT_FLOAT_EQ(node->stats.Q(), 0.0f);
  EXPECT_FLOAT_EQ(node->stats.D(), 0.0f);
  EXPECT_FLOAT_EQ(node->stats.M(), 0.0f);
  EXPECT_EQ(TriedAt(*node, 0).stats.visits, 0u);
  EXPECT_EQ(TriedAt(*node, 0).in_flight, 1u);
  EXPECT_FLOAT_EQ(TriedAt(*node, 0).stats.Q(), 0.0f);
  EXPECT_FLOAT_EQ(TriedAt(*node, 0).stats.D(), 0.0f);
  EXPECT_FLOAT_EQ(TriedAt(*node, 0).stats.M(), 0.0f);
}

TEST(Lc5GraphTest, BackupCommitsNodeDespiteEdgeFailure) {
  Graph graph;
  NodeKey key{456};
  Move move = M(kFileA, kFileA);
  auto created = graph.FindOrCreateMaterialization(key, 1);
  graph.CompleteMaterialization(key, 1, {.moves = {move}, .priors = {1.0f}});
  auto result =
      graph.BackupNode(key, created.generation, {0.25f, 0.5f, 3.0f}, move);
  EXPECT_EQ(result.node, UpdateResult::kApplied);
  EXPECT_EQ(result.edge, UpdateResult::kUnderflow);
  EXPECT_TRUE(result.to_parent);
  result = graph.BackupNode(key, created.generation, {0.25f, 0.5f, 3.0f},
                            M(kFileB, kFileB));
  EXPECT_EQ(result.node, UpdateResult::kApplied);
  EXPECT_EQ(result.edge, UpdateResult::kEdgeMissing);
  auto node = graph.SnapshotNode(key);
  ASSERT_TRUE(node);
  EXPECT_EQ(node->stats.visits, 3u);
  EXPECT_FLOAT_EQ(node->stats.Q(), 0.5f / 3);
  EXPECT_FLOAT_EQ(node->stats.D(), 1.0f / 3);
  EXPECT_FLOAT_EQ(node->stats.M(), 2.0f);
  EXPECT_EQ(TriedAt(*node, 0).stats.visits, 0u);
  EXPECT_EQ(TriedAt(*node, 0).in_flight, 0u);
  EXPECT_FLOAT_EQ(TriedAt(*node, 0).stats.Q(), 0.0f);
  EXPECT_FLOAT_EQ(TriedAt(*node, 0).stats.D(), 0.0f);
  EXPECT_FLOAT_EQ(TriedAt(*node, 0).stats.M(), 0.0f);
}

TEST(Lc5GraphTest, ReservationsAreExactAndCancellationDoesNotCommit) {
  Graph graph;
  NodeKey key{456};
  auto created = graph.FindOrCreateMaterialization(key, 1);
  Move first = M(kFileA, kFileA);
  Move second = M(kFileB, kFileB);
  ExpansionPayload payload{.moves = {first, second}, .priors = {0.5f, 0.5f}};
  graph.CompleteMaterialization(key, 1, payload);
  auto params = TestParams();
  auto selection1 = graph.SelectAndReserve(key, params);
  auto selection2 = graph.SelectAndReserve(key, params);
  EXPECT_EQ(selection1.move, first);
  EXPECT_EQ(selection2.move, second);
  EXPECT_EQ(graph.CancelEdge(key, created.generation, first),
            UpdateResult::kApplied);
  EXPECT_EQ(
      graph.BackupNode(key, created.generation, {{0.4f, 0.2f, 2.0f}}, second)
          .edge,
      UpdateResult::kApplied);
  auto node = graph.SnapshotNode(key);
  ASSERT_TRUE(node);
  EXPECT_EQ(TriedAt(*node, 0).stats.visits, 0u);
  EXPECT_EQ(TriedAt(*node, 0).in_flight, 0u);
  EXPECT_EQ(TriedAt(*node, 1).stats.visits, 1u);
  EXPECT_FLOAT_EQ(TriedAt(*node, 1).stats.Q(), 0.4f);
  EXPECT_EQ(graph.CancelEdge(key, created.generation, first),
            UpdateResult::kUnderflow);
}

// Priors in descending order, with moves out of encoding order.
ExpansionPayload FourEdges() {
  return {.moves = {M(kFileD, kFileD), M(kFileA, kFileA), M(kFileC, kFileC),
                    M(kFileB, kFileB)},
          .priors = {0.4f, 0.3f, 0.2f, 0.1f}};
}

TEST(Lc5GraphTest, TriedPrefixGrowsInPriorOrder) {
  Graph graph;
  const NodeKey key{456};
  const auto created = graph.FindOrCreateMaterialization(key, 1);
  const auto payload = FourEdges();
  graph.CompleteMaterialization(key, 1, payload);
  // Untried edges score above every tried one, so each is tried in turn.
  auto params = TestParams();
  params.fpu_value = 1.0f;
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_EQ(graph.SnapshotNode(key)->tried.size(), i);
    const auto selection = graph.SelectAndReserve(key, params);
    EXPECT_EQ(selection.move, payload.moves[i]);
    EXPECT_EQ(graph.SnapshotNode(key)->tried.size(), i + 1);
    graph.BackupNode(key, created.generation, {}, selection.move);
    graph.BackupNode(key, created.generation, {}, std::nullopt);
    EXPECT_EQ(graph.SnapshotNode(key)->tried.size(), i + 1);
  }
  EXPECT_EQ(graph.SelectAndReserve(key, params).move, payload.moves[0]);
  EXPECT_EQ(graph.SnapshotNode(key)->tried.size(), 4u);
}

TEST(Lc5GraphTest, CancelledTriedEdgeIsStillSelectable) {
  Graph graph;
  const NodeKey key{456};
  const auto created = graph.FindOrCreateMaterialization(key, 1);
  const auto payload = FourEdges();
  graph.CompleteMaterialization(key, 1, payload);
  auto params = TestParams();
  params.fpu_value = 1.0f;
  EXPECT_EQ(graph.SelectAndReserve(key, params).move, payload.moves[0]);
  EXPECT_EQ(graph.SelectAndReserve(key, params).move, payload.moves[1]);
  EXPECT_EQ(graph.CancelEdge(key, created.generation, payload.moves[0]),
            UpdateResult::kApplied);
  EXPECT_EQ(graph.SnapshotNode(key)->tried.size(), 2u);
  EXPECT_EQ(graph.SelectAndReserve(key, params).move, payload.moves[0]);
  const auto node = graph.SnapshotNode(key);
  EXPECT_EQ(node->tried.size(), 2u);
  EXPECT_EQ(TriedAt(*node, 0).in_flight, 1u);
}

TEST(Lc5GraphTest, FirstUntriedEdgeCompetesWithTriedOnes) {
  for (const float q : {0.5f, -0.5f}) {
    SCOPED_TRACE(q);
    Graph graph;
    const NodeKey key{456};
    const auto created = graph.FindOrCreateMaterialization(key, 1);
    const ExpansionPayload payload{
        .moves = {M(kFileA, kFileA), M(kFileB, kFileB), M(kFileC, kFileC)},
        .priors = {0.5f, 0.3f, 0.2f}};
    graph.CompleteMaterialization(key, 1, payload);
    const auto params = TestParams();
    ASSERT_EQ(graph.SelectAndReserve(key, params).move, payload.moves[0]);
    graph.BackupNode(key, created.generation, {q, 0.0f, 0.0f},
                     payload.moves[0]);
    // The tried edge scores q + 0.25 and the first untried one 0.3.
    const bool tried_wins = q > 0.05f;
    EXPECT_EQ(graph.SelectAndReserve(key, params).move,
              payload.moves[tried_wins ? 0 : 1]);
    EXPECT_EQ(graph.SnapshotNode(key)->tried.size(), tried_wins ? 1u : 2u);
  }
}

TEST(Lc5GraphTest, ReselectionMovesOnlyTheWaitersReservation) {
  Graph graph;
  const NodeKey key{456};
  const auto created = graph.FindOrCreateMaterialization(key, 1);
  const ExpansionPayload payload{
      .moves = {M(kFileA, kFileA), M(kFileB, kFileB), M(kFileC, kFileC)},
      .priors = {0.8f, 0.1f, 0.1f}};
  graph.CompleteMaterialization(key, 1, payload);
  const Move a = payload.moves[0];
  const Move b = payload.moves[1];
  const auto params = TestParams();
  // Edge A scores 0.8, then 0.8 / 2 = 0.4 against B's 0.1.
  ASSERT_EQ(graph.SelectAndReserve(key, params).move, a);
  ASSERT_EQ(graph.SelectAndReserve(key, params).move, a);
  // N = 2 and sqrt(N) = 1.414. A good child keeps edge A, 0.9 + 0.8 * 1.414 /
  // 3 = 1.277 against B's 0.141, and its reservation.
  auto result = graph.ReselectAndReserve(key, created.generation, a,
                                         {.value = {.q = 0.9f}}, params);
  EXPECT_EQ(result.status, SelectStatus::kSelected);
  EXPECT_EQ(result.move, a);
  EXPECT_EQ(result.generation, created.generation);
  auto node = graph.SnapshotNode(key);
  EXPECT_EQ(TriedAt(*node, 0).in_flight, 2u);
  EXPECT_EQ(TriedAt(*node, 0).stats.visits, 0u);
  // A bad child moves it to edge B: A scores -0.9 + 0.377 = -0.523.
  result = graph.ReselectAndReserve(key, created.generation, a,
                                    {.value = {.q = -0.9f}}, params);
  EXPECT_EQ(result.status, SelectStatus::kSelected);
  EXPECT_EQ(result.move, b);
  node = graph.SnapshotNode(key);
  EXPECT_EQ(TriedAt(*node, 0).in_flight, 1u);
  EXPECT_EQ(TriedAt(*node, 1).in_flight, 1u);
  EXPECT_EQ(TriedAt(*node, 0).stats.visits, 0u);
  // Nothing changes for a stale generation or an edge without reservations.
  EXPECT_EQ(graph
                .ReselectAndReserve(key, created.generation + 1, a,
                                    {.value = {.q = -0.9f}}, params)
                .status,
            SelectStatus::kMissing);
  EXPECT_EQ(graph
                .ReselectAndReserve(key, created.generation, payload.moves[2],
                                    {.value = {.q = -0.9f}}, params)
                .status,
            SelectStatus::kMissing);
  node = graph.SnapshotNode(key);
  EXPECT_EQ(node->tried.size(), 2u);
  EXPECT_EQ(TriedAt(*node, 0).in_flight, 1u);
  EXPECT_EQ(TriedAt(*node, 1).in_flight, 1u);
}

TEST(Lc5GraphTest, IncomingParentsKeepIndependentEdgeStatistics) {
  Graph graph;
  Move move = M(kFileC, kFileC);
  ExpansionPayload payload{.moves = {move}, .priors = {1.0f}};
  auto p1 = graph.FindOrCreateMaterialization(NodeKey{1}, 1);
  graph.FindOrCreateMaterialization(NodeKey{2}, 2);
  graph.CompleteMaterialization(NodeKey{1}, 1, payload);
  graph.CompleteMaterialization(NodeKey{2}, 2, payload);
  auto params = TestParams();
  graph.SelectAndReserve(NodeKey{1}, params);
  graph.BackupNode(NodeKey{1}, p1.generation, {}, move);
  EXPECT_EQ(TriedAt(*graph.SnapshotNode(NodeKey{1}), 0).stats.visits, 1u);
  EXPECT_EQ(TriedAt(*graph.SnapshotNode(NodeKey{2}), 0).stats.visits, 0u);
}

// A node with two of its three edges tried, visits and proven bounds.
Graph::Node BackedUpNode(Graph& graph, NodeKey key) {
  graph.CompleteMaterialization(
      key, 1,
      {.leaf = {0.5f, 0.25f, 6.0f},
       .moves = {M(kFileA, kFileA), M(kFileB, kFileB), M(kFileC, kFileC)},
       .priors = {0.5f, 0.3f, 0.2f}});
  const auto params = TestParams();
  for (int i = 0; i < 2; ++i) {
    const auto selected = graph.SelectAndReserve(key, params);
    EXPECT_EQ(selected.status, SelectStatus::kSelected);
    graph.BackupNode(key, selected.generation,
                     PuctAveraging::Leaf({-0.75f, 0.25f, 3.0f}, i == 0),
                     selected.move);
  }
  return *graph.SnapshotNode(key);
}

TEST(Lc5NodeRecordTest, RoundTripKeepsStatisticsAndEdges) {
  Graph graph;
  const NodeKey key{42};
  const auto node = BackedUpNode(graph, key);
  ASSERT_EQ(node.tried.size(), 2u);
  const auto decoded = DecodeNode<PuctAveraging>(EncodeNode(node));
  ASSERT_TRUE(decoded);
  EXPECT_EQ(decoded->lifecycle, NodeLifecycle::kExpanded);
  EXPECT_EQ(decoded->terminal, node.terminal);
  EXPECT_EQ(decoded->stats.visits, node.stats.visits);
  EXPECT_EQ(decoded->stats.Q(), node.stats.Q());
  EXPECT_EQ(decoded->stats.D(), node.stats.D());
  EXPECT_EQ(decoded->stats.M(), node.stats.M());
  EXPECT_EQ(decoded->stats.bounds, node.stats.bounds);
  ASSERT_EQ(decoded->edges.size(), node.edges.size());
  for (size_t i = 0; i < node.edges.size(); ++i) {
    EXPECT_EQ(decoded->edges[i].move, node.edges[i].move);
    EXPECT_EQ(decoded->edges[i].prior.bits, node.edges[i].prior.bits);
  }
  ASSERT_EQ(decoded->tried.size(), node.tried.size());
  for (size_t i = 0; i < node.tried.size(); ++i) {
    EXPECT_EQ(decoded->tried[i].in_flight, 0u);
    EXPECT_EQ(decoded->tried[i].stats.visits, node.tried[i].stats.visits);
    EXPECT_EQ(decoded->tried[i].stats.Q(), node.tried[i].stats.Q());
    EXPECT_EQ(decoded->tried[i].stats.bounds, node.tried[i].stats.bounds);
  }
  // The first backup was exact, so that edge is proven.
  EXPECT_TRUE(decoded->tried[0].stats.bounds.Proven());

  // A terminal node has no edges.
  graph.CompleteMaterialization(
      NodeKey{43}, 1,
      {.terminal = TerminalKind::kCheckmate,
       .leaf = TerminalValue(TerminalKind::kCheckmate),
       .moves = {},
       .priors = {}});
  const auto terminal =
      DecodeNode<PuctAveraging>(EncodeNode(*graph.SnapshotNode(NodeKey{43})));
  ASSERT_TRUE(terminal);
  EXPECT_EQ(terminal->terminal, TerminalKind::kCheckmate);
  EXPECT_TRUE(terminal->edges.empty());
  EXPECT_EQ(terminal->stats.visits, 1u);
}

TEST(Lc5NodeRecordTest, RejectsOtherPoliciesAndMalformedRecords) {
  Graph graph;
  const auto record = EncodeNode(BackedUpNode(graph, NodeKey{42}));
  EXPECT_FALSE(DecodeNode<PuctDag>(record));
  EXPECT_TRUE(DecodeNode<PuctAveraging>(record));
  EXPECT_FALSE(DecodeNode<PuctAveraging>({}));
  auto truncated = record;
  truncated.pop_back();
  EXPECT_FALSE(DecodeNode<PuctAveraging>(truncated));
  auto longer = record;
  longer.push_back(0);
  EXPECT_FALSE(DecodeNode<PuctAveraging>(longer));
  auto version = record;
  version[offsetof(RecordHeader, version)] ^= 1;
  EXPECT_FALSE(DecodeNode<PuctAveraging>(version));
}

TEST(Lc5GraphTest, RestoreKeepsStatisticsUnderANewGeneration) {
  Graph graph;
  const NodeKey key{42};
  const auto original = BackedUpNode(graph, key);
  const auto record = EncodeNode(original);
  ASSERT_TRUE(
      graph.EraseIfUnchanged(key, original.generation, original.version));
  EXPECT_EQ(graph.Size(), 0u);

  const auto placeholder = graph.FindOrCreateMaterialization(key, 9);
  ASSERT_TRUE(placeholder.created);
  EXPECT_NE(placeholder.generation, original.generation);
  auto decoded = DecodeNode<PuctAveraging>(record);
  ASSERT_TRUE(decoded);
  // Another ticket's restore is rejected.
  EXPECT_FALSE(graph.RestoreNode(key, 8, *decoded).accepted);
  const auto restored = graph.RestoreNode(key, 9, *decoded);
  ASSERT_TRUE(restored.accepted);
  EXPECT_EQ(restored.generation, placeholder.generation);
  // One-shot per ticket.
  EXPECT_FALSE(graph.RestoreNode(key, 9, *decoded).accepted);

  const auto node = graph.SnapshotNode(key);
  ASSERT_TRUE(node);
  EXPECT_EQ(node->lifecycle, NodeLifecycle::kExpanded);
  EXPECT_EQ(node->ticket, 0u);
  EXPECT_EQ(node->stats.visits, original.stats.visits);
  EXPECT_EQ(node->stats.Q(), original.stats.Q());
  ASSERT_EQ(node->tried.size(), original.tried.size());
  EXPECT_EQ(node->tried[0].stats.visits, original.tried[0].stats.visits);
  // Updates for the evicted node are rejected.
  EXPECT_EQ(graph
                .BackupNode(key, original.generation,
                            PuctAveraging::Leaf({0.0f, 1.0f, 0.0f}, false),
                            std::nullopt)
                .node,
            UpdateResult::kStale);
  EXPECT_EQ(graph
                .BackupNode(key, restored.generation,
                            PuctAveraging::Leaf({0.0f, 1.0f, 0.0f}, false),
                            std::nullopt)
                .node,
            UpdateResult::kApplied);
  EXPECT_EQ(graph.SnapshotNode(key)->stats.visits, original.stats.visits + 1);
}

TEST(Lc5GraphTest, EraseIfUnchangedRefusesAfterAnyChange) {
  Graph graph;
  const NodeKey key{42};
  const auto params = TestParams();
  auto node = BackedUpNode(graph, key);
  // A backup.
  graph.BackupNode(key, node.generation,
                   PuctAveraging::Leaf({0.0f, 1.0f, 0.0f}, false),
                   std::nullopt);
  EXPECT_FALSE(graph.EraseIfUnchanged(key, node.generation, node.version));
  // A reservation, which also makes the node not evictable.
  node = *graph.SnapshotNode(key);
  const auto selected = graph.SelectAndReserve(key, params);
  EXPECT_FALSE(graph.EraseIfUnchanged(key, node.generation, node.version));
  node = *graph.SnapshotNode(key);
  EXPECT_FALSE(graph.EraseIfUnchanged(key, node.generation, node.version));
  // A cancelled reservation.
  graph.CancelEdge(key, selected.generation, selected.move);
  EXPECT_FALSE(graph.EraseIfUnchanged(key, node.generation, node.version));
  // Another generation.
  node = *graph.SnapshotNode(key);
  EXPECT_FALSE(graph.EraseIfUnchanged(key, node.generation + 1, node.version));
  EXPECT_TRUE(graph.EraseIfUnchanged(key, node.generation, node.version));
  EXPECT_FALSE(graph.SnapshotNode(key));
  EXPECT_EQ(graph.Size(), 0u);
}

TEST(Lc5GraphTest, EvictCandidatesSkipReservedMaterializingAndKeptNodes) {
  Graph graph;
  const auto params = TestParams();
  const NodeKey keep{1}, reserved{2}, materializing{3}, idle{4};
  for (const NodeKey key : {keep, reserved, idle}) {
    graph.CompleteMaterialization(
        key, 1, {.moves = {M(kFileA, kFileA)}, .priors = {1.0f}});
  }
  graph.FindOrCreateMaterialization(materializing, 5);
  ASSERT_EQ(graph.SelectAndReserve(reserved, params).status,
            SelectStatus::kSelected);
  size_t cursor = 0;
  const auto candidates = graph.EvictCandidates(&cursor, 10, 1.0, keep);
  ASSERT_EQ(candidates.size(), 1u);
  EXPECT_EQ(candidates[0].first, idle);
  const auto node = graph.SnapshotNode(idle);
  EXPECT_EQ(candidates[0].second.generation, node->generation);
  EXPECT_EQ(candidates[0].second.version, node->version);
  EXPECT_EQ(EncodeNode(candidates[0].second), EncodeNode(*node));
  // Nothing is erased until the eviction is confirmed.
  EXPECT_EQ(graph.Size(), 4u);
}

TEST(Lc5GraphTest, EvictCandidatesPreferFewVisits) {
  Graph graph;
  const auto params = TestParams();
  // All keys in one shard (the graph's shard index is the top 10 bits of a
  // Fibonacci hash), so the fraction applies to them together.
  std::vector<NodeKey> keys;
  for (uint64_t i = 0; keys.size() < 4; ++i) {
    if ((i * 11400714819323198485ull) >> 54 == 0) keys.push_back(NodeKey{i});
  }
  for (size_t i = 0; i < keys.size(); ++i) {
    graph.CompleteMaterialization(
        keys[i], 1, {.moves = {M(kFileA, kFileA)}, .priors = {1.0f}});
    for (size_t visit = 0; visit < 3 - i % 3 + (i == 3 ? 5 : 0); ++visit) {
      const auto selected = graph.SelectAndReserve(keys[i], params);
      graph.BackupNode(keys[i], selected.generation,
                       PuctAveraging::Leaf({0.0f, 1.0f, 0.0f}, false),
                       selected.move);
    }
  }
  size_t cursor = 0;
  const auto candidates =
      graph.EvictCandidates(&cursor, 2, 0.5, NodeKey{~0ull});
  ASSERT_EQ(candidates.size(), 2u);
  std::vector<uint32_t> visits;
  for (const auto& candidate : candidates) {
    visits.push_back(graph.SnapshotNode(candidate.first)->stats.visits);
  }
  std::ranges::sort(visits);
  // Visits are 4, 3, 2 and 9; the two lowest are taken.
  EXPECT_EQ(visits, (std::vector<uint32_t>{2, 3}));
}

}  // namespace
}  // namespace lczero::lc5

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
