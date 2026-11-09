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
const TriedEdge<PuctAveraging>& TriedAt(const Node<PuctAveraging>& node,
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

TEST(Lc5ValueTest, PerspectiveAndMeans) {
  VisitValue child{.q = 0.25f, .d = 0.5f, .m = 3.0f};
  VisitValue parent = child.Parent();
  EXPECT_FLOAT_EQ(parent.q, -0.25f);
  EXPECT_FLOAT_EQ(parent.d, 0.5f);
  EXPECT_FLOAT_EQ(parent.m, 4.0f);
  ValueStats stats;
  stats.Add({.q = 0.25f, .d = 0.5f, .m = 3.0f});
  stats.Add({.q = -0.25f, .d = 0.25f, .m = 5.0f});
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
  for (int i = 0; i < 100000; ++i) {
    constant.Add({.q = 0.3f, .d = 0.2f, .m = 7.0f});
  }
  EXPECT_EQ(constant.visits, 100000u);
  EXPECT_EQ(constant.Q(), 0.3f);
  EXPECT_EQ(constant.D(), 0.2f);
  EXPECT_EQ(constant.M(), 7.0f);

  // Q cycles through -0.5, 0, 0.5, 1 with mean 0.25.
  ValueStats sequence;
  for (int i = 0; i < 4000; ++i) {
    sequence.Add({.q = -0.5f + 0.5f * (i % 4),
                  .d = 0.0f,
                  .m = static_cast<float>(i % 3)});
  }
  EXPECT_NEAR(sequence.Q(), 0.25f, 1e-5f);
  EXPECT_NEAR(sequence.M(), 3999.0f / 4000.0f, 1e-5f);
}

TEST(Lc5GraphTest, DuplicatePublicationDoesNotDisturbExistingNode) {
  Graph graph;
  const NodeKey key{123};
  const Move move = M(kFileA, kFileA);
  const NodeEval eval{.moves = {move}, .priors = {1.0f}};
  const NodeEval replacement{.moves = {M(kFileB, kFileB)}, .priors = {0.5f}};

  EXPECT_FALSE(graph.ContainsNode(key));
  EXPECT_TRUE(graph.InsertNode(key, eval));
  EXPECT_TRUE(graph.ContainsNode(key));
  ASSERT_EQ(graph.SelectAndReserveEdge(key, TestParams()).status,
            SelectResult::kEdgeSelected);
  ASSERT_TRUE(graph.UpdateNode(key, {0.25f, 0.5f, 3.0f}, move).to_parent);
  ASSERT_EQ(graph.SelectAndReserveEdge(key, TestParams()).status,
            SelectResult::kEdgeSelected);
  // Duplicate publications are rejected and must not disturb statistics or
  // live reservations.
  EXPECT_FALSE(graph.InsertNode(key, replacement));
  auto node = graph.SnapshotNode(key);
  ASSERT_TRUE(node);
  ASSERT_EQ(node->edges.size(), 1u);
  EXPECT_EQ(node->edges[0].move, move);
  EXPECT_FLOAT_EQ(node->edges[0].prior, 1.0f);
  EXPECT_EQ(TriedAt(*node, 0).stats.visits, 1u);
  EXPECT_EQ(TriedAt(*node, 0).in_flight, 1u);
  EXPECT_FLOAT_EQ(TriedAt(*node, 0).stats.Q(), 0.25f);
  EXPECT_EQ(node->stats.visits, 2u);
  EXPECT_TRUE(graph.ReleaseEdge(key, move));

  ASSERT_TRUE(graph.EraseNode(key, graph.SnapshotNode(key)->version));
  EXPECT_FALSE(graph.ContainsNode(key));
  EXPECT_TRUE(graph.InsertNode(key, replacement));

  EXPECT_TRUE(graph.InsertNode(NodeKey{456}, eval));
  EXPECT_EQ(graph.Size(), 2u);
}

TEST(Lc5GraphTest, SelectionReservesMaterializedEdge) {
  Graph graph;
  const NodeKey key{123};
  EXPECT_EQ(graph.SelectAndReserveEdge(key, TestParams()).status,
            SelectResult::kNodeMissing);

  const NodeEval eval{.moves = {M(kFileA, kFileA)}, .priors = {1.0f}};
  graph.InsertNode(key, eval);
  const auto selected = graph.SelectAndReserveEdge(key, TestParams());
  EXPECT_EQ(selected.status, SelectResult::kEdgeSelected);
  EXPECT_EQ(selected.move, M(kFileA, kFileA));
  const auto full = graph.SnapshotNode(key);
  ASSERT_TRUE(full);
  ASSERT_EQ(full->edges.size(), 1u);
  EXPECT_EQ(TriedAt(*full, 0).in_flight, 1u);
  EXPECT_EQ(full->terminal, TerminalKind::kNonTerminal);
}

TEST(Lc5GraphTest, SelectionReportsMaterializedTerminalKind) {
  Graph graph;
  const NodeKey key{123};
  graph.InsertNode(
      key,
      NodeEval{
          .terminal = TerminalKind::kCheckmate, .moves = {}, .priors = {}});
  const auto node = graph.SnapshotNode(key);
  ASSERT_TRUE(node);
  EXPECT_EQ(node->terminal, TerminalKind::kCheckmate);
  const auto terminal = graph.SelectAndReserveEdge(key, TestParams());
  EXPECT_EQ(terminal.status, SelectResult::kNodeTerminal);
  EXPECT_EQ(terminal.terminal, TerminalKind::kCheckmate);
}

TEST(Lc5GraphTest, ProvenBoundsGoUpAndStopSelectionWhenAsked) {
  Graph graph;
  const NodeKey parent{1};
  const NodeKey child{2};
  const Move move = M(kFileA, kFileA);
  graph.InsertNode(parent, {.moves = {move}, .priors = {1.0f}});
  graph.InsertNode(
      child, {.terminal = TerminalKind::kCheckmate,
              .leaf = TerminalValue(TerminalKind::kCheckmate),
              .moves = {},
              .priors = {}});
  // The install proves the checkmate; the leaf's own backup keeps it.
  ASSERT_EQ(graph.SelectAndReserveEdge(parent, TestParams()).move, move);
  const auto from_child = graph.UpdateNode(
      child, PuctAveraging::Leaf(TerminalValue(TerminalKind::kCheckmate), true),
      std::nullopt);
  ASSERT_TRUE(from_child.to_parent);
  EXPECT_EQ(from_child.to_parent->bounds,
            (Bounds{.lower = 1, .upper = 1, .plies = 1}));
  const auto up = graph.UpdateNode(parent, *from_child.to_parent, move);
  ASSERT_TRUE(up.to_parent);
  EXPECT_EQ(up.to_parent->bounds,
            (Bounds{.lower = -1, .upper = -1, .plies = 2}));
  EXPECT_EQ(graph.SnapshotNode(parent)->stats.bounds,
            (Bounds{.lower = 1, .upper = 1, .plies = 1}));

  // A proven node stops selection only with StickyEndgames and when asked.
  auto params = TestParams();
  const auto select = [&](bool stop_at_proven) {
    const auto selected =
        graph.SelectAndReserveEdge(parent, params, stop_at_proven);
    if (selected.status == SelectResult::kEdgeSelected) {
      graph.ReleaseEdge(parent, move);
    }
    return selected;
  };
  EXPECT_EQ(select(true).status, SelectResult::kEdgeSelected);
  params.sticky_endgames = true;
  EXPECT_EQ(select(false).status, SelectResult::kEdgeSelected);
  const auto proven = select(true);
  EXPECT_EQ(proven.status, SelectResult::kNodeProven);
  EXPECT_EQ(proven.value.q, 1.0f);
  EXPECT_EQ(proven.value.m, 1.0f);
  EXPECT_EQ(graph.SnapshotNode(parent)->tried[0].in_flight, 0u);
}

TEST(Lc5GraphTest, RecreatedNodeGetsNewVersion) {
  Graph graph;
  const NodeKey key{123};
  const NodeEval eval{.moves = {M(kFileA, kFileA)}, .priors = {1.0f}};
  graph.InsertNode(key, eval);
  const uint32_t first_version = graph.SnapshotNode(key)->version;
  ASSERT_TRUE(graph.EraseNode(key, first_version));
  EXPECT_FALSE(graph.SnapshotNode(key));

  graph.InsertNode(key, eval);
  EXPECT_GT(graph.SnapshotNode(key)->version, first_version);
  const auto selected = graph.SelectAndReserveEdge(key, TestParams());
  EXPECT_EQ(selected.status, SelectResult::kEdgeSelected);
  EXPECT_EQ(TriedAt(*graph.SnapshotNode(key), 0).in_flight, 1u);
}

TEST(Lc5GraphTest, SizeCountsEvalInsertionAndSuccessfulErasure) {
  Graph graph;
  NodeKey key{123};
  NodeEval eval{.moves = {M(kFileA, kFileA)}, .priors = {1.0f}};
  EXPECT_TRUE(graph.InsertNode(key, eval));
  EXPECT_EQ(graph.Size(), 1u);
  EXPECT_FALSE(graph.InsertNode(key, eval));
  EXPECT_EQ(graph.Size(), 1u);
  EXPECT_FALSE(graph.EraseNode(NodeKey{456}, 0));
  EXPECT_EQ(graph.Size(), 1u);
  const uint32_t version = graph.SnapshotNode(key)->version;
  EXPECT_TRUE(graph.EraseNode(key, version));
  EXPECT_EQ(graph.Size(), 0u);
  EXPECT_FALSE(graph.EraseNode(key, version));
  EXPECT_EQ(graph.Size(), 0u);
  EXPECT_TRUE(graph.InsertNode(key, eval));
  EXPECT_EQ(graph.Size(), 1u);
  EXPECT_TRUE(graph.InsertNode(NodeKey{456}, eval));
  EXPECT_EQ(graph.Size(), 2u);
  EXPECT_TRUE(
      graph.EraseNode(NodeKey{456}, graph.SnapshotNode(NodeKey{456})->version));
  EXPECT_EQ(graph.Size(), 1u);
}

TEST(Lc5GraphTest, ClearResetsSizeAndAllowsReuse) {
  Graph graph;
  graph.Clear();
  EXPECT_EQ(graph.Size(), 0u);
  NodeEval eval{.moves = {M(kFileA, kFileA)}, .priors = {1.0f}};
  for (uint64_t hash = 0; hash < 32; ++hash) {
    graph.InsertNode(NodeKey{hash}, eval);
  }
  EXPECT_EQ(graph.Size(), 32u);
  size_t cursor = 0;
  EXPECT_EQ(graph.Size(),
            graph.FindEvictableNodes(&cursor, 64, 1.0, NodeKey{~0ull}).size());
  graph.Clear();
  EXPECT_EQ(graph.Size(), 0u);
  EXPECT_TRUE(
      graph.FindEvictableNodes(&cursor, 64, 1.0, NodeKey{~0ull}).empty());
  graph.Clear();
  EXPECT_EQ(graph.Size(), 0u);
  EXPECT_TRUE(graph.InsertNode(NodeKey{0}, eval));
  EXPECT_EQ(graph.Size(), 1u);
  graph.InsertNode(NodeKey{1}, eval);
  EXPECT_EQ(graph.Size(), 2u);
  EXPECT_TRUE(
      graph.EraseNode(NodeKey{0}, graph.SnapshotNode(NodeKey{0})->version));
  EXPECT_EQ(graph.Size(), 1u);
  graph.Clear();
  EXPECT_EQ(graph.Size(), 0u);
}

TEST(Lc5GraphTest, AcceptedInstallCountsLeafValueOnce) {
  Graph graph;
  const NodeKey key{321};
  const NodeEval eval{.leaf = {.q = 0.5f, .d = 0.25f, .m = 6.0f},
                      .moves = {M(kFileA, kFileA)},
                      .priors = {1.0f}};
  EXPECT_TRUE(graph.InsertNode(key, eval));
  auto node = graph.SnapshotNode(key);
  ASSERT_TRUE(node);
  EXPECT_EQ(node->stats.visits, 1u);
  EXPECT_FLOAT_EQ(node->stats.Q(), 0.5f);
  EXPECT_FLOAT_EQ(node->stats.D(), 0.25f);
  EXPECT_FLOAT_EQ(node->stats.M(), 6.0f);

  // Duplicate publications do not count the leaf again.
  EXPECT_FALSE(graph.InsertNode(key, eval));
  node = graph.SnapshotNode(key);
  ASSERT_TRUE(node);
  EXPECT_EQ(node->stats.visits, 1u);
  EXPECT_FLOAT_EQ(node->stats.Q(), 0.5f);
}

TEST(Lc5GraphTest, BackupCommitsNodeAndOptionalEdgeSums) {
  Graph graph;
  NodeKey key{456};
  Move move = M(kFileA, kFileA);
  graph.InsertNode(key, {.moves = {move}, .priors = {1.0f}});
  auto result = graph.UpdateNode(key, {0.25f, 0.5f, 3.0f}, std::nullopt);
  EXPECT_TRUE(result.to_parent);
  auto node = graph.SnapshotNode(key);
  ASSERT_TRUE(node);
  EXPECT_EQ(node->stats.visits, 2u);
  EXPECT_FLOAT_EQ(node->stats.Q(), 0.125f);
  EXPECT_FLOAT_EQ(node->stats.D(), 0.25f);
  EXPECT_FLOAT_EQ(node->stats.M(), 1.5f);
  EXPECT_EQ(TriedAt(*node, 0).stats.visits, 0u);
  ASSERT_EQ(graph.SelectAndReserveEdge(key, TestParams()).status,
            SelectResult::kEdgeSelected);
  result = graph.UpdateNode(key, {-0.5f, 0.25f, 4.0f}, move);
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
  // The install counts the leaf value once: Q 0.25, D 0.5, M 2.
  graph.InsertNode(key, {.leaf = {.q = 0.25f, .d = 0.5f, .m = 2.0f},
                         .moves = {move},
                         .priors = {1.0f}});
  ASSERT_EQ(graph.SelectAndReserveEdge(key, TestParams()).status,
            SelectResult::kEdgeSelected);
  const auto result = graph.UpdateNode(key, {-0.75f, 0.0f, 4.0f}, move, true);
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

TEST(Lc5GraphTest, BackupRejectsMissingNodes) {
  Graph graph;
  NodeKey key{456};
  Move move = M(kFileA, kFileA);
  auto result = graph.UpdateNode(key, {0.25f, 0.5f, 3.0f}, move);
  EXPECT_FALSE(result.to_parent);
  EXPECT_FALSE(graph.SnapshotNode(key));
}

TEST(Lc5GraphTest, BackupCommitsNodeDespiteEdgeFailure) {
  Graph graph;
  NodeKey key{456};
  Move move = M(kFileA, kFileA);
  graph.InsertNode(key, {.moves = {move}, .priors = {1.0f}});
  auto result = graph.UpdateNode(key, {0.25f, 0.5f, 3.0f}, move);
  EXPECT_TRUE(result.to_parent);
  result = graph.UpdateNode(key, {0.25f, 0.5f, 3.0f}, M(kFileB, kFileB));
  EXPECT_TRUE(result.to_parent);
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
  Move first = M(kFileA, kFileA);
  Move second = M(kFileB, kFileB);
  NodeEval eval{.moves = {first, second}, .priors = {0.5f, 0.5f}};
  graph.InsertNode(key, eval);
  auto params = TestParams();
  auto selection1 = graph.SelectAndReserveEdge(key, params);
  auto selection2 = graph.SelectAndReserveEdge(key, params);
  EXPECT_EQ(selection1.move, first);
  EXPECT_EQ(selection2.move, second);
  EXPECT_TRUE(graph.ReleaseEdge(key, first));
  EXPECT_TRUE(graph.UpdateNode(key, {{0.4f, 0.2f, 2.0f}}, second).to_parent);
  auto node = graph.SnapshotNode(key);
  ASSERT_TRUE(node);
  EXPECT_EQ(TriedAt(*node, 0).stats.visits, 0u);
  EXPECT_EQ(TriedAt(*node, 0).in_flight, 0u);
  EXPECT_EQ(TriedAt(*node, 1).stats.visits, 1u);
  EXPECT_FLOAT_EQ(TriedAt(*node, 1).stats.Q(), 0.4f);
  EXPECT_FALSE(graph.ReleaseEdge(key, first));
}

// Priors in descending order, with moves out of encoding order.
NodeEval FourEdges() {
  return {.moves = {M(kFileD, kFileD), M(kFileA, kFileA), M(kFileC, kFileC),
                    M(kFileB, kFileB)},
          .priors = {0.4f, 0.3f, 0.2f, 0.1f}};
}

TEST(Lc5GraphTest, TriedPrefixGrowsInPriorOrder) {
  Graph graph;
  const NodeKey key{456};
  const auto eval = FourEdges();
  graph.InsertNode(key, eval);
  // Untried edges score above every tried one, so each is tried in turn.
  auto params = TestParams();
  params.fpu_value = 1.0f;
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_EQ(graph.SnapshotNode(key)->tried.size(), i);
    const auto selection = graph.SelectAndReserveEdge(key, params);
    EXPECT_EQ(selection.move, eval.moves[i]);
    EXPECT_EQ(graph.SnapshotNode(key)->tried.size(), i + 1);
    graph.UpdateNode(key, {}, selection.move);
    graph.UpdateNode(key, {}, std::nullopt);
    EXPECT_EQ(graph.SnapshotNode(key)->tried.size(), i + 1);
  }
  EXPECT_EQ(graph.SelectAndReserveEdge(key, params).move, eval.moves[0]);
  EXPECT_EQ(graph.SnapshotNode(key)->tried.size(), 4u);
}

TEST(Lc5GraphTest, CancelledTriedEdgeIsStillSelectable) {
  Graph graph;
  const NodeKey key{456};
  const auto eval = FourEdges();
  graph.InsertNode(key, eval);
  auto params = TestParams();
  params.fpu_value = 1.0f;
  EXPECT_EQ(graph.SelectAndReserveEdge(key, params).move, eval.moves[0]);
  EXPECT_EQ(graph.SelectAndReserveEdge(key, params).move, eval.moves[1]);
  EXPECT_TRUE(graph.ReleaseEdge(key, eval.moves[0]));
  EXPECT_EQ(graph.SnapshotNode(key)->tried.size(), 2u);
  EXPECT_EQ(graph.SelectAndReserveEdge(key, params).move, eval.moves[0]);
  const auto node = graph.SnapshotNode(key);
  EXPECT_EQ(node->tried.size(), 2u);
  EXPECT_EQ(TriedAt(*node, 0).in_flight, 1u);
}

TEST(Lc5GraphTest, FirstUntriedEdgeCompetesWithTriedOnes) {
  for (const float q : {0.5f, -0.5f}) {
    SCOPED_TRACE(q);
    Graph graph;
    const NodeKey key{456};
    const NodeEval eval{
        .moves = {M(kFileA, kFileA), M(kFileB, kFileB), M(kFileC, kFileC)},
        .priors = {0.5f, 0.3f, 0.2f}};
    graph.InsertNode(key, eval);
    const auto params = TestParams();
    ASSERT_EQ(graph.SelectAndReserveEdge(key, params).move, eval.moves[0]);
    graph.UpdateNode(key, {q, 0.0f, 0.0f}, eval.moves[0]);
    // The tried edge scores q + 0.25 and the first untried one 0.3.
    const bool tried_wins = q > 0.05f;
    EXPECT_EQ(graph.SelectAndReserveEdge(key, params).move,
              eval.moves[tried_wins ? 0 : 1]);
    EXPECT_EQ(graph.SnapshotNode(key)->tried.size(), tried_wins ? 1u : 2u);
  }
}

TEST(Lc5GraphTest, ReselectionMovesOnlyTheWaitersReservation) {
  Graph graph;
  const NodeKey key{456};
  const NodeEval eval{
      .moves = {M(kFileA, kFileA), M(kFileB, kFileB), M(kFileC, kFileC)},
      .priors = {0.8f, 0.1f, 0.1f}};
  graph.InsertNode(key, eval);
  const Move a = eval.moves[0];
  const Move b = eval.moves[1];
  const auto params = TestParams();
  // Edge A scores 0.8, then 0.8 / 2 = 0.4 against B's 0.1.
  ASSERT_EQ(graph.SelectAndReserveEdge(key, params).move, a);
  ASSERT_EQ(graph.SelectAndReserveEdge(key, params).move, a);
  // N = 2 and sqrt(N) = 1.414. A good child keeps edge A, 0.9 + 0.8 * 1.414 /
  // 3 = 1.277 against B's 0.141, and its reservation.
  EXPECT_EQ(graph.ReselectAndReserveEdge(key, a, {.value = {.q = 0.9f}}, params),
            a);
  auto node = graph.SnapshotNode(key);
  EXPECT_EQ(TriedAt(*node, 0).in_flight, 2u);
  EXPECT_EQ(TriedAt(*node, 0).stats.visits, 0u);
  // A bad child moves it to edge B: A scores -0.9 + 0.377 = -0.523.
  EXPECT_EQ(
      graph.ReselectAndReserveEdge(key, a, {.value = {.q = -0.9f}}, params), b);
  node = graph.SnapshotNode(key);
  EXPECT_EQ(TriedAt(*node, 0).in_flight, 1u);
  EXPECT_EQ(TriedAt(*node, 1).in_flight, 1u);
  EXPECT_EQ(TriedAt(*node, 0).stats.visits, 0u);
  // Nothing changes for a missing node or an edge without reservations.
  EXPECT_EQ(graph.ReselectAndReserveEdge(NodeKey{999}, a,
                                         {.value = {.q = -0.9f}}, params),
            std::nullopt);
  EXPECT_EQ(graph.ReselectAndReserveEdge(key, eval.moves[2],
                                         {.value = {.q = -0.9f}}, params),
            std::nullopt);
  node = graph.SnapshotNode(key);
  EXPECT_EQ(node->tried.size(), 2u);
  EXPECT_EQ(TriedAt(*node, 0).in_flight, 1u);
  EXPECT_EQ(TriedAt(*node, 1).in_flight, 1u);
}

TEST(Lc5GraphTest, IncomingParentsKeepIndependentEdgeStatistics) {
  Graph graph;
  Move move = M(kFileC, kFileC);
  NodeEval eval{.moves = {move}, .priors = {1.0f}};
  graph.InsertNode(NodeKey{1}, eval);
  graph.InsertNode(NodeKey{2}, eval);
  auto params = TestParams();
  graph.SelectAndReserveEdge(NodeKey{1}, params);
  graph.UpdateNode(NodeKey{1}, {}, move);
  EXPECT_EQ(TriedAt(*graph.SnapshotNode(NodeKey{1}), 0).stats.visits, 1u);
  EXPECT_EQ(TriedAt(*graph.SnapshotNode(NodeKey{2}), 0).stats.visits, 0u);
}

// A node with two of its three edges tried, visits and proven bounds.
Graph::Node BackedUpNode(Graph& graph, NodeKey key) {
  graph.InsertNode(
      key, {.leaf = {.q = 0.5f, .d = 0.25f, .m = 6.0f},
            .moves = {M(kFileA, kFileA), M(kFileB, kFileB), M(kFileC, kFileC)},
            .priors = {0.5f, 0.3f, 0.2f}});
  const auto params = TestParams();
  for (int i = 0; i < 2; ++i) {
    const auto selected = graph.SelectAndReserveEdge(key, params);
    EXPECT_EQ(selected.status, SelectResult::kEdgeSelected);
    graph.UpdateNode(
        key, PuctAveraging::Leaf({.q = -0.75f, .d = 0.25f, .m = 3.0f}, i == 0),
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
  graph.InsertNode(
      NodeKey{43}, {.terminal = TerminalKind::kCheckmate,
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

TEST(Lc5GraphTest, RestoreKeepsStatisticsUnderANewVersion) {
  Graph graph;
  const NodeKey key{42};
  const auto original = BackedUpNode(graph, key);
  const auto record = EncodeNode(original);
  ASSERT_TRUE(graph.EraseNode(key, original.version));
  EXPECT_EQ(graph.Size(), 0u);

  auto decoded = DecodeNode<PuctAveraging>(record);
  ASSERT_TRUE(decoded);
  EXPECT_TRUE(graph.InsertNode(key, *decoded));
  EXPECT_EQ(graph.Size(), 1u);
  // Duplicate restore is rejected.
  EXPECT_FALSE(graph.InsertNode(key, *decoded));
  EXPECT_EQ(graph.Size(), 1u);

  const auto node = graph.SnapshotNode(key);
  ASSERT_TRUE(node);
  EXPECT_GT(node->version, original.version);
  EXPECT_EQ(node->stats.visits, original.stats.visits);
  EXPECT_EQ(node->stats.Q(), original.stats.Q());
  ASSERT_EQ(node->tried.size(), original.tried.size());
  EXPECT_EQ(node->tried[0].stats.visits, original.tried[0].stats.visits);
  // A stale eviction for the old version is rejected.
  EXPECT_FALSE(graph.EraseNode(key, original.version));
  EXPECT_TRUE(graph
                  .UpdateNode(key,
                              PuctAveraging::Leaf(
                                  {.q = 0.0f, .d = 1.0f, .m = 0.0f}, false),
                              std::nullopt)
                  .to_parent);
  EXPECT_EQ(graph.SnapshotNode(key)->stats.visits, original.stats.visits + 1);
}

TEST(Lc5GraphTest, EraseNodeRefusesAfterAnyChange) {
  Graph graph;
  const NodeKey key{42};
  const auto params = TestParams();
  auto node = BackedUpNode(graph, key);
  // A backup.
  graph.UpdateNode(
      key, PuctAveraging::Leaf({.q = 0.0f, .d = 1.0f, .m = 0.0f}, false),
      std::nullopt);
  EXPECT_FALSE(graph.EraseNode(key, node.version));
  // A reservation, which also makes the node not evictable.
  node = *graph.SnapshotNode(key);
  const auto selected = graph.SelectAndReserveEdge(key, params);
  EXPECT_FALSE(graph.EraseNode(key, node.version));
  node = *graph.SnapshotNode(key);
  EXPECT_FALSE(graph.EraseNode(key, node.version));
  // A cancelled reservation.
  graph.ReleaseEdge(key, selected.move);
  EXPECT_FALSE(graph.EraseNode(key, node.version));
  // Another version.
  node = *graph.SnapshotNode(key);
  EXPECT_FALSE(graph.EraseNode(key, node.version + 1));
  EXPECT_TRUE(graph.EraseNode(key, node.version));
  EXPECT_FALSE(graph.SnapshotNode(key));
  EXPECT_EQ(graph.Size(), 0u);
}

TEST(Lc5GraphTest, FindEvictableNodesSkipReservedAndKeptNodes) {
  Graph graph;
  const auto params = TestParams();
  const NodeKey keep{1}, reserved{2}, idle{4};
  for (const NodeKey key : {keep, reserved, idle}) {
    graph.InsertNode(key, {.moves = {M(kFileA, kFileA)}, .priors = {1.0f}});
  }
  ASSERT_EQ(graph.SelectAndReserveEdge(reserved, params).status,
            SelectResult::kEdgeSelected);
  size_t cursor = 0;
  const auto candidates = graph.FindEvictableNodes(&cursor, 10, 1.0, keep);
  ASSERT_EQ(candidates.size(), 1u);
  EXPECT_EQ(candidates[0].first, idle);
  const auto node = graph.SnapshotNode(idle);
  EXPECT_EQ(candidates[0].second.version, node->version);
  EXPECT_EQ(EncodeNode(candidates[0].second), EncodeNode(*node));
  // Nothing is erased until the eviction is confirmed.
  EXPECT_EQ(graph.Size(), 3u);
}

TEST(Lc5GraphTest, FindEvictableNodesPreferFewVisits) {
  Graph graph;
  const auto params = TestParams();
  // All keys in one shard (the graph's shard index is the top 10 bits of a
  // Fibonacci hash), so the fraction applies to them together.
  std::vector<NodeKey> keys;
  for (uint64_t i = 0; keys.size() < 4; ++i) {
    if ((i * 11400714819323198485ull) >> 54 == 0) keys.push_back(NodeKey{i});
  }
  for (size_t i = 0; i < keys.size(); ++i) {
    graph.InsertNode(keys[i], {.moves = {M(kFileA, kFileA)}, .priors = {1.0f}});
    for (size_t visit = 0; visit < 3 - i % 3 + (i == 3 ? 5 : 0); ++visit) {
      const auto selected = graph.SelectAndReserveEdge(keys[i], params);
      graph.UpdateNode(
          keys[i],
          PuctAveraging::Leaf({.q = 0.0f, .d = 1.0f, .m = 0.0f}, false),
          selected.move);
    }
  }
  size_t cursor = 0;
  const auto candidates =
      graph.FindEvictableNodes(&cursor, 2, 0.5, NodeKey{~0ull});
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
