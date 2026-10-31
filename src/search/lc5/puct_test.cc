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

#include "search/lc5/puct.h"

#include <gtest/gtest.h>

#include "search/lc5/policies.h"

namespace lczero::lc5 {
namespace {

using Node = NodeState<PuctAveraging>;
using Tried = TriedEdge<PuctAveraging>;

Move M(File from, File to) {
  return Move::White(Square(from, kRank2), Square(to, kRank3));
}

LegalEdge MakeEdge(File file, float prior) {
  return {.move = M(file, file), .prior = prior};
}

Tried MakeTried(uint32_t visits, float q = 0.0f, float m = 0.0f) {
  return {.stats = {.visits = visits, .q = q, .m = m}};
}

BackendAttributes MakeAttributes(bool has_mlh) {
  return {.has_mlh = has_mlh,
          .has_wdl = true,
          .runs_on_cpu = false,
          .suggested_num_search_threads = 1,
          .recommended_batch_size = 1,
          .maximum_batch_size = 1};
}

// A backend with a moves-left head.
const BackendAttributes kMlh = MakeAttributes(true);

PuctAveraging::Params TestParams() {
  return {.cpuct = 1.0f,
          .cpuct_base = 100.0f,
          .cpuct_factor = 0.0f,
          .fpu_strategy = FpuStrategy::kAbsolute,
          .fpu_value = 0.0f};
}

TEST(Lc5PuctTest, OptionDefaultsMatchParams) {
  OptionsParser parser;
  PopulatePolicyOptions(&parser);
  EXPECT_EQ(PolicyName(parser.GetOptionsDict()), "puct");
  const auto params =
      PuctAveraging::Params::FromOptions(parser.GetOptionsDict(), kMlh);
  const PuctAveraging::Params defaults;
  EXPECT_EQ(params.cpuct, defaults.cpuct);
  EXPECT_EQ(params.cpuct_base, defaults.cpuct_base);
  EXPECT_EQ(params.cpuct_factor, defaults.cpuct_factor);
  EXPECT_EQ(params.fpu_strategy, FpuStrategy::kReduction);
  EXPECT_EQ(params.fpu_value, defaults.fpu_value);
  EXPECT_EQ(params.totals, PuctTotals::kStarted);
  EXPECT_EQ(params.waiter_resume, WaiterResume::kSelect);
  EXPECT_EQ(PuctAveraging::OnResume(params), WaiterResume::kSelect);
  EXPECT_FALSE(params.moves_left.enabled);
  EXPECT_EQ(params.moves_left.max_effect, defaults.moves_left.max_effect);
  EXPECT_EQ(params.moves_left.threshold, defaults.moves_left.threshold);
  EXPECT_EQ(params.moves_left.slope, defaults.moves_left.slope);
  EXPECT_EQ(params.moves_left.constant_factor,
            defaults.moves_left.constant_factor);
  EXPECT_EQ(params.moves_left.scaled_factor, defaults.moves_left.scaled_factor);
  EXPECT_EQ(params.moves_left.quadratic_factor,
            defaults.moves_left.quadratic_factor);
  EXPECT_FALSE(params.sticky_endgames);
  EXPECT_EQ(params.two_fold_draws, TwoFoldDraws::kRoot);
  EXPECT_FALSE(PuctAveraging::ContainTwoFold(params));
}

TEST(Lc5PuctTest, OptionsReachParams) {
  OptionsParser parser;
  PopulatePolicyOptions(&parser);
  parser.SetUciOption("CPuct", "3");
  parser.SetUciOption("FpuStrategy", "absolute");
  parser.SetUciOption("PuctTotals", "completed");
  parser.SetUciOption("WaiterResume", "restart");
  parser.SetUciOption("MovesLeft", "true");
  parser.SetUciOption("MovesLeftSlope", "0.01");
  parser.SetUciOption("StickyEndgames", "true");
  parser.SetUciOption("TwoFoldDraws", "contained");
  const auto params =
      PuctAveraging::Params::FromOptions(parser.GetOptionsDict(), kMlh);
  EXPECT_EQ(params.cpuct, 3.0f);
  EXPECT_EQ(params.fpu_strategy, FpuStrategy::kAbsolute);
  EXPECT_EQ(params.totals, PuctTotals::kCompleted);
  EXPECT_EQ(PuctAveraging::OnResume(params), WaiterResume::kRestart);
  EXPECT_TRUE(params.moves_left.enabled);
  EXPECT_EQ(params.moves_left.slope, 0.01f);
  EXPECT_TRUE(params.sticky_endgames);
  EXPECT_EQ(params.two_fold_draws, TwoFoldDraws::kContained);
  EXPECT_TRUE(PuctAveraging::ContainTwoFold(params));
  parser.SetUciOption("WaiterResume", "reselect");
  EXPECT_EQ(PuctAveraging::Params::FromOptions(parser.GetOptionsDict(), kMlh)
                .waiter_resume,
            WaiterResume::kReselect);
  // Without a moves-left head the option has no effect.
  EXPECT_FALSE(PuctAveraging::Params::FromOptions(parser.GetOptionsDict(),
                                                  MakeAttributes(false))
                   .moves_left.enabled);
}

TEST(Lc5PuctTest, UnstartedNodePicksTopPrior) {
  const Node node{.lifecycle = NodeLifecycle::kExpanded,
                  .stats = {.visits = 1},
                  .edges = {MakeEdge(kFileA, 0.5f), MakeEdge(kFileB, 0.3f),
                            MakeEdge(kFileC, 0.2f)}};
  EXPECT_EQ(PuctAveraging::Select(node, TestParams()), 0u);
}

TEST(Lc5PuctTest, FirstUntriedEdgeCompetesWithTriedOnes) {
  for (const float q : {0.5f, -0.5f}) {
    SCOPED_TRACE(q);
    const Node node{.lifecycle = NodeLifecycle::kExpanded,
                    .stats = {.visits = 2},
                    .edges = {MakeEdge(kFileA, 0.5f), MakeEdge(kFileB, 0.3f),
                              MakeEdge(kFileC, 0.2f)},
                    .tried = {MakeTried(1, q)}};
    // The tried edge scores q + 0.25 and the first untried one 0.3.
    EXPECT_EQ(PuctAveraging::Select(node, TestParams()), q > 0.05f ? 0u : 1u);
  }
}

TEST(Lc5PuctTest, InFlightEdgeLosesUrgency) {
  Node node{.lifecycle = NodeLifecycle::kExpanded,
            .stats = {.visits = 1},
            .edges = {MakeEdge(kFileA, 0.5f), MakeEdge(kFileB, 0.3f)},
            .tried = {Tried{.in_flight = 1}}};
  // Edge A scores 0.5 / 2 with one visit in flight; edge B scores 0.3.
  EXPECT_EQ(PuctAveraging::Select(node, TestParams()), 1u);
}

TEST(Lc5PuctTest, InFlightCountsInTotalsOnlyWhenStarted) {
  Tried tried = MakeTried(1, 0.25f);
  tried.in_flight = 3;
  const Node node{.lifecycle = NodeLifecycle::kExpanded,
                  .stats = {.visits = 2},
                  .edges = {MakeEdge(kFileA, 0.5f), MakeEdge(kFileB, 0.3f)},
                  .tried = {tried}};
  auto params = TestParams();
  // Started totals: sqrt(N) = 2, edge A scores 0.25 + 0.5 * 2 / 5 = 0.45 and
  // edge B 0.3 * 2 = 0.6.
  EXPECT_EQ(PuctAveraging::Select(node, params), 1u);
  // Completed totals: sqrt(N) = 1, edge A scores 0.25 + 0.5 / 5 = 0.35 and
  // edge B 0.3.
  params.totals = PuctTotals::kCompleted;
  EXPECT_EQ(PuctAveraging::Select(node, params), 0u);
}

TEST(Lc5PuctTest, ReselectCountsWaiterAsCompleted) {
  // The owner's visit is still in flight on edge A; the waiter's reservation
  // has been released and counts as completed with the child's value.
  const Node node{.lifecycle = NodeLifecycle::kExpanded,
                  .stats = {.visits = 1},
                  .edges = {MakeEdge(kFileA, 0.5f), MakeEdge(kFileB, 0.3f)},
                  .tried = {Tried{.in_flight = 1}}};
  // N = 2, so sqrt(N) = 1.414. Edge A scores q + 0.5 * 1.414 / 3 = q + 0.236
  // and edge B 0.3 * 1.414 = 0.424.
  // q = -0.9: A -0.664 < B 0.424.
  EXPECT_EQ(
      PuctAveraging::Reselect(node, TestParams(), 0, {.value = {.q = -0.9f}}),
      1u);
  // q = 0.9: A 1.136 > B 0.424.
  EXPECT_EQ(
      PuctAveraging::Reselect(node, TestParams(), 0, {.value = {.q = 0.9f}}),
      0u);
}

TEST(Lc5PuctTest, MovesLeftPrefersShorterWinsAndLongerLosses) {
  auto params = TestParams();
  params.moves_left.enabled = true;
  const auto select = [&](float node_q, float edge_q) {
    // Equal priors and visits, so without moves-left the tie goes to edge A.
    // The node has M 20; edge A's child has M 30 (10 plies longer) and edge
    // B's child M 10 (10 plies shorter).
    const Node node{
        .lifecycle = NodeLifecycle::kExpanded,
        .stats = {.visits = 3, .q = node_q, .m = 20.0f},
        .edges = {MakeEdge(kFileA, 0.5f), MakeEdge(kFileB, 0.5f)},
        .tried = {MakeTried(1, edge_q, 31.0f), MakeTried(1, edge_q, 11.0f)}};
    return PuctAveraging::Select(node, params);
  };
  // Edge |Q| 0.9 is (0.9 - 0.8) / 0.2 = 0.5 above the threshold, so the
  // scaling is 1.6521 * 0.5 - 0.6521 * 0.25 = 0.663. The slope gives
  // 0.0027 * 10 = 0.027, so the edges get +-0.0179.
  // Winning: B (shorter) gets the bonus.
  EXPECT_EQ(select(0.9f, 0.9f), 1u);
  // Losing: A (longer) gets the bonus.
  EXPECT_EQ(select(-0.9f, -0.9f), 0u);
  // Node |Q| 0.5 is below the threshold: tie.
  EXPECT_EQ(select(0.5f, 0.9f), 0u);
  params.moves_left.enabled = false;
  EXPECT_EQ(select(0.9f, 0.9f), 0u);
}

TEST(Lc5PuctTest, UpdateAveragesAndFlipsForParent) {
  Node node{.lifecycle = NodeLifecycle::kExpanded,
            .stats = {.visits = 1, .q = 0.5f, .d = 0.5f, .m = 2.0f},
            .edges = {MakeEdge(kFileA, 1.0f)},
            .tried = {Tried{}}};
  const PuctAveraging::Message message{
      .value = {.q = -0.5f, .d = 0.25f, .m = 4.0f}};
  const auto up = PuctAveraging::Update(node, &node.tried[0], message);
  const auto& edge = node.tried[0].stats;
  EXPECT_EQ(node.stats.visits, 2u);
  EXPECT_FLOAT_EQ(node.stats.Q(), 0.0f);
  EXPECT_FLOAT_EQ(node.stats.D(), 0.375f);
  EXPECT_FLOAT_EQ(node.stats.M(), 3.0f);
  EXPECT_EQ(edge.visits, 1u);
  EXPECT_FLOAT_EQ(edge.Q(), -0.5f);
  EXPECT_FLOAT_EQ(up.value.q, 0.5f);
  EXPECT_FLOAT_EQ(up.value.d, 0.25f);
  EXPECT_FLOAT_EQ(up.value.m, 5.0f);
  EXPECT_EQ(up.bounds, Bounds{});

  PuctAveraging::Update(node, nullptr, message);
  EXPECT_EQ(node.stats.visits, 3u);
  EXPECT_EQ(edge.visits, 1u);
  const auto skipped = PuctAveraging::Skip(message);
  EXPECT_FLOAT_EQ(skipped.value.q, 0.5f);
  EXPECT_FLOAT_EQ(skipped.value.m, 5.0f);
}

TEST(Lc5PuctTest, CurrentSendsNodeMeanAndBoundsFlipped) {
  Node node{.lifecycle = NodeLifecycle::kExpanded,
            .stats = {.visits = 4,
                      .q = 0.25f,
                      .d = 0.5f,
                      .m = 6.0f,
                      .bounds = {.lower = 0, .upper = 1}},
            .edges = {MakeEdge(kFileA, 1.0f)},
            .tried = {Tried{}}};
  const auto up = PuctAveraging::Current(node);
  EXPECT_FLOAT_EQ(up.value.q, -0.25f);
  EXPECT_FLOAT_EQ(up.value.d, 0.5f);
  EXPECT_FLOAT_EQ(up.value.m, 7.0f);
  EXPECT_EQ(up.bounds, (Bounds{.lower = -1, .upper = 0}));
}

// An expanded node with `edges` edges, of which the first ones are tried with
// one visit each and `bounds`, in the node's frame.
Node MakeBoundedNode(size_t edges, std::vector<Bounds> bounds) {
  Node node{.lifecycle = NodeLifecycle::kExpanded,
            .stats = {.visits = 1 + static_cast<uint32_t>(bounds.size())}};
  for (size_t i = 0; i < edges; ++i) {
    node.edges.push_back(MakeEdge(File::FromIdx(i), 1.0f / edges));
  }
  for (const Bounds& b : bounds) {
    node.tried.push_back(MakeTried(1, b.lower));
    node.tried.back().stats.bounds = b;
  }
  return node;
}

constexpr Bounds kWin{.lower = 1, .upper = 1};
constexpr Bounds kDraw{.lower = 0, .upper = 0};
constexpr Bounds kLoss{.lower = -1, .upper = -1};
Bounds WithPlies(Bounds bounds, uint8_t plies) {
  bounds.plies = plies;
  return bounds;
}

TEST(Lc5PuctTest, TerminalLossProvesParentWin) {
  // The child is checkmated: its install leaf is exact.
  Node child{.lifecycle = NodeLifecycle::kExpanded,
             .terminal = TerminalKind::kCheckmate};
  const auto leaf =
      PuctAveraging::Leaf(TerminalValue(TerminalKind::kCheckmate), true);
  const auto from_child = PuctAveraging::Update(child, nullptr, leaf);
  EXPECT_EQ(child.stats.bounds, kLoss);
  EXPECT_EQ(from_child.bounds, WithPlies(kWin, 1));

  // A parent with an untried edge left is still proven by one win.
  Node parent = MakeBoundedNode(2, {Bounds{}});
  const auto up = PuctAveraging::Update(parent, &parent.tried[0], from_child);
  EXPECT_EQ(parent.tried[0].stats.bounds, WithPlies(kWin, 1));
  EXPECT_EQ(parent.stats.bounds, WithPlies(kWin, 1));
  EXPECT_EQ(up.bounds, WithPlies(kLoss, 2));
}

TEST(Lc5PuctTest, NodeProvesBoundsFromItsEdges) {
  const auto prove = [](size_t edges, std::vector<Bounds> bounds) {
    // Re-sends the last edge's bounds as a change.
    Node node = MakeBoundedNode(edges, bounds);
    const Bounds last = node.tried.back().stats.bounds;
    node.tried.back().stats.bounds = {};
    PuctAveraging::Update(
        node, &node.tried.back(),
        {.value = {.q = static_cast<float>(last.lower)}, .bounds = last});
    return node.stats.bounds;
  };
  // All edges lose: the node loses, by the longest one.
  EXPECT_EQ(prove(2, {WithPlies(kLoss, 3), WithPlies(kLoss, 5)}),
            WithPlies(kLoss, 5));
  // An untried edge is unknown.
  EXPECT_EQ(prove(3, {WithPlies(kLoss, 3), WithPlies(kLoss, 5)}),
            (Bounds{.lower = -1, .upper = 1}));
  // The shortest win.
  EXPECT_EQ(prove(3, {WithPlies(kWin, 7), WithPlies(kWin, 3)}),
            WithPlies(kWin, 3));
  EXPECT_EQ(prove(2, {kDraw, WithPlies(kLoss, 3)}), kDraw);
  EXPECT_EQ(prove(2, {kDraw, Bounds{.lower = -1, .upper = 0}}), kDraw);
  EXPECT_EQ(prove(2, {Bounds{}, kDraw}), (Bounds{.lower = 0, .upper = 1}));
}

TEST(Lc5PuctTest, StaleDropsBoundsAndSkipKeepsThem) {
  const PuctAveraging::Message message{.value = {.q = -1.0f}, .bounds = kLoss};
  EXPECT_EQ(PuctAveraging::Skip(message).bounds, WithPlies(kWin, 1));
  EXPECT_EQ(PuctAveraging::Stale(message).bounds, Bounds{});
  EXPECT_FLOAT_EQ(PuctAveraging::Stale(message).value.q, 1.0f);
  // Only exact leaves are proven.
  EXPECT_EQ(PuctAveraging::Leaf({.d = 1.0f}, true).bounds, kDraw);
  EXPECT_EQ(PuctAveraging::Leaf({.d = 1.0f}, false).bounds, Bounds{});
}

TEST(Lc5PuctTest, StickySelectionUsesBounds) {
  auto params = TestParams();
  // Edge A has the higher prior and Q but is a proven loss.
  Node node = MakeBoundedNode(2, {kLoss, Bounds{}});
  node.edges[0].prior = 0.9f;
  node.edges[1].prior = 0.1f;
  node.tried[0].stats.q = 0.5f;
  node.tried[1].stats.q = 0.0f;
  EXPECT_EQ(PuctAveraging::Select(node, params), 0u);
  params.sticky_endgames = true;
  EXPECT_EQ(PuctAveraging::Select(node, params), 1u);

  // Edge B's Q -0.5 is clamped to its lower bound 0, which beats A's -0.1
  // (equal priors and visits, so U ties).
  node = MakeBoundedNode(2, {Bounds{}, Bounds{.lower = 0, .upper = 1}});
  node.tried[0].stats.q = -0.1f;
  node.tried[1].stats.q = -0.5f;
  EXPECT_EQ(PuctAveraging::Select(node, params), 1u);
  params.sticky_endgames = false;
  EXPECT_EQ(PuctAveraging::Select(node, params), 0u);
}

TEST(Lc5PuctTest, StickyChooseEdgePrefersShortWinsAndLongLosses) {
  auto params = TestParams();
  params.sticky_endgames = true;
  // A proven win with fewer visits beats a more-visited unknown edge.
  Node node = MakeBoundedNode(2, {Bounds{}, WithPlies(kWin, 5)});
  node.tried[0].stats.visits = 10;
  EXPECT_EQ(PuctAveraging::ChooseEdge(node, params), 1u);
  EXPECT_EQ(PuctAveraging::ChooseEdge(node, TestParams()), 0u);
  // Of two proven wins, the shorter one.
  node = MakeBoundedNode(2, {WithPlies(kWin, 5), WithPlies(kWin, 3)});
  node.tried[0].stats.visits = 10;
  EXPECT_EQ(PuctAveraging::ChooseEdge(node, params), 1u);
  // A proven loss is chosen only if all edges lose; then the longest.
  node = MakeBoundedNode(2, {WithPlies(kLoss, 1), Bounds{}});
  node.tried[0].stats.visits = 10;
  EXPECT_EQ(PuctAveraging::ChooseEdge(node, params), 1u);
  node = MakeBoundedNode(2, {WithPlies(kLoss, 1), WithPlies(kLoss, 3)});
  node.tried[0].stats.visits = 10;
  EXPECT_EQ(PuctAveraging::ChooseEdge(node, params), 1u);
}

TEST(Lc5PuctTest, ProvenValueNeedsStickyEndgames) {
  auto params = TestParams();
  Node node = MakeBoundedNode(1, {WithPlies(kWin, 3)});
  node.stats.bounds = WithPlies(kWin, 3);
  EXPECT_EQ(PuctAveraging::ProvenValue(node, params), std::nullopt);
  EXPECT_FALSE(PuctAveraging::Decided(node, params));
  params.sticky_endgames = true;
  const auto value = PuctAveraging::ProvenValue(node, params);
  ASSERT_TRUE(value);
  EXPECT_EQ(value->q, 1.0f);
  EXPECT_EQ(value->d, 0.0f);
  EXPECT_EQ(value->m, 3.0f);
  EXPECT_TRUE(PuctAveraging::Decided(node, params));
  node.stats.bounds = {.lower = 0, .upper = 1};
  EXPECT_EQ(PuctAveraging::ProvenValue(node, params), std::nullopt);
}

TEST(Lc5PuctTest, ChooseEdgeRanksVisitsThenQThenPrior) {
  Node node{.lifecycle = NodeLifecycle::kExpanded,
            .edges = {MakeEdge(kFileA, 0.5f), MakeEdge(kFileB, 0.3f),
                      MakeEdge(kFileC, 0.2f)}};
  EXPECT_EQ(PuctAveraging::ChooseEdge(node, TestParams()), std::nullopt);

  // Tried edges without a visit don't count.
  node.tried = {Tried{.in_flight = 1}, Tried{}, MakeTried(1, -0.9f)};
  EXPECT_EQ(PuctAveraging::ChooseEdge(node, TestParams()), 2u);

  node.tried[1].stats = {.visits = 2, .q = -0.5f};
  node.tried[2].stats = {.visits = 2, .q = 0.1f};
  EXPECT_EQ(PuctAveraging::ChooseEdge(node, TestParams()), 2u);

  node.tried[1].stats.q = 0.1f;
  EXPECT_EQ(PuctAveraging::ChooseEdge(node, TestParams()), 1u);

  node.tried[0].stats = {.visits = 3, .q = -1.0f};
  EXPECT_EQ(PuctAveraging::ChooseEdge(node, TestParams()), 0u);
}

TEST(Lc5PuctTest, ChooseMoveFallsBackToTopPrior) {
  Node node{.lifecycle = NodeLifecycle::kExpanded};
  EXPECT_EQ(PuctAveraging::ChooseMove(node, TestParams()), std::nullopt);

  node.edges = {MakeEdge(kFileC, 0.5f), MakeEdge(kFileA, 0.3f),
                MakeEdge(kFileB, 0.2f)};
  EXPECT_EQ(PuctAveraging::ChooseEdge(node, TestParams()), std::nullopt);
  EXPECT_EQ(PuctAveraging::ChooseMove(node, TestParams()), 0u);

  node.tried = {Tried{}, Tried{}, MakeTried(1)};
  EXPECT_EQ(PuctAveraging::ChooseMove(node, TestParams()), 2u);
}

TEST(Lc5PuctTest, DescribeNeedsVisits) {
  const auto describe = [](ValueStats stats, ThinkingInfo& info) {
    PuctAveraging::Describe(Node{.stats = stats}, TestParams(), info);
  };
  ThinkingInfo info;
  describe({}, info);
  EXPECT_FALSE(info.score);
  EXPECT_FALSE(info.wdl);

  describe({.visits = 1, .q = 0.0f, .d = 1.0f}, info);
  ASSERT_TRUE(info.score);
  EXPECT_EQ(*info.score, 0);
  ASSERT_TRUE(info.wdl);
  EXPECT_EQ(info.wdl->w, 0);
  EXPECT_EQ(info.wdl->d, 1000);
  EXPECT_EQ(info.wdl->l, 0);

  describe({.visits = 1, .q = 0.5f}, info);
  EXPECT_GT(*info.score, 0);
  EXPECT_EQ(info.wdl->w, 750);
  EXPECT_EQ(info.wdl->l, 250);
}

TEST(Lc5PuctTest, DescribeShowsProvenMates) {
  auto params = TestParams();
  params.sticky_endgames = true;
  const auto describe = [&](Bounds bounds) {
    ThinkingInfo info;
    PuctAveraging::Describe(
        Node{.stats = {.visits = 2,
                       .q = static_cast<float>(bounds.lower),
                       .bounds = bounds}},
        params, info);
    return info;
  };
  // Mate in one move (one ply), and mated after our move and theirs.
  ThinkingInfo info = describe(WithPlies(kWin, 1));
  EXPECT_EQ(info.mate, 1);
  EXPECT_FALSE(info.score);
  EXPECT_EQ(describe(WithPlies(kWin, 3)).mate, 2);
  info = describe(WithPlies(kLoss, 2));
  EXPECT_EQ(info.mate, -1);
  EXPECT_FALSE(info.score);
  // A proven draw keeps its centipawns.
  info = describe(kDraw);
  EXPECT_FALSE(info.mate);
  EXPECT_EQ(info.score, 0);
  // Not without the option.
  params.sticky_endgames = false;
  info = describe(WithPlies(kWin, 1));
  EXPECT_FALSE(info.mate);
  EXPECT_TRUE(info.score);
}

}  // namespace
}  // namespace lczero::lc5

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
