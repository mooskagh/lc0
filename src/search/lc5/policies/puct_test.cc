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

#include "search/lc5/policies/puct.h"

#include <gtest/gtest.h>

#include "search/lc5/policies/puct_dag.h"
#include "search/lc5/policies/registry.h"
#include "search/lc5/visit_engine.h"

namespace lczero::lc5 {
namespace {

using Node = lc5::Node<PuctAveraging>;
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
  EXPECT_TRUE(params.moves_left.enabled);
  EXPECT_EQ(params.moves_left.max_effect, defaults.moves_left.max_effect);
  EXPECT_EQ(params.moves_left.threshold, defaults.moves_left.threshold);
  EXPECT_EQ(params.moves_left.slope, defaults.moves_left.slope);
  EXPECT_EQ(params.moves_left.constant_factor,
            defaults.moves_left.constant_factor);
  EXPECT_EQ(params.moves_left.scaled_factor, defaults.moves_left.scaled_factor);
  EXPECT_EQ(params.moves_left.quadratic_factor,
            defaults.moves_left.quadratic_factor);
  EXPECT_FALSE(params.sticky_endgames);
}

TEST(Lc5PuctTest, OptionsReachParams) {
  OptionsParser parser;
  PopulatePolicyOptions(&parser);
  parser.SetUciOption("CPuct", "3");
  parser.SetUciOption("FpuStrategy", "absolute");
  parser.SetUciOption("MovesLeft", "true");
  parser.SetUciOption("MovesLeftSlope", "0.01");
  parser.SetUciOption("StickyEndgames", "true");
  const auto params =
      PuctAveraging::Params::FromOptions(parser.GetOptionsDict(), kMlh);
  EXPECT_EQ(params.cpuct, 3.0f);
  EXPECT_EQ(params.fpu_strategy, FpuStrategy::kAbsolute);
  EXPECT_TRUE(params.moves_left.enabled);
  EXPECT_EQ(params.moves_left.slope, 0.01f);
  EXPECT_TRUE(params.sticky_endgames);
  // Without a moves-left head the option has no effect.
  EXPECT_FALSE(PuctAveraging::Params::FromOptions(parser.GetOptionsDict(),
                                                  MakeAttributes(false))
                   .moves_left.enabled);
  parser.SetUciOption("MovesLeft", "false");
  EXPECT_FALSE(PuctAveraging::Params::FromOptions(parser.GetOptionsDict(), kMlh)
                   .moves_left.enabled);
}

TEST(Lc5PuctTest, UnstartedNodePicksTopPrior) {
  const Node node{.stats = {.visits = 1},
                  .edges = {MakeEdge(kFileA, 0.5f), MakeEdge(kFileB, 0.3f),
                            MakeEdge(kFileC, 0.2f)}};
  EXPECT_EQ(PuctAveraging::PickEdgeToVisit(node, TestParams()), 0u);
}

TEST(Lc5PuctTest, FirstUntriedEdgeCompetesWithTriedOnes) {
  for (const float q : {0.5f, -0.5f}) {
    SCOPED_TRACE(q);
    const Node node{.stats = {.visits = 2},
                    .edges = {MakeEdge(kFileA, 0.5f), MakeEdge(kFileB, 0.3f),
                              MakeEdge(kFileC, 0.2f)},
                    .tried = {MakeTried(1, q)}};
    // The tried edge scores q + 0.25 and the first untried one 0.3.
    EXPECT_EQ(PuctAveraging::PickEdgeToVisit(node, TestParams()),
              q > 0.05f ? 0u : 1u);
  }
}

TEST(Lc5PuctTest, InFlightEdgeLosesUrgency) {
  Node node{.stats = {.visits = 1},
            .edges = {MakeEdge(kFileA, 0.5f), MakeEdge(kFileB, 0.3f)},
            .tried = {Tried{.in_flight = 1}}};
  // Edge A scores 0.5 / 2 with one visit in flight; edge B scores 0.3.
  EXPECT_EQ(PuctAveraging::PickEdgeToVisit(node, TestParams()), 1u);
}

TEST(Lc5PuctTest, InFlightCountsInTotals) {
  Tried tried = MakeTried(1, 0.25f);
  tried.in_flight = 3;
  const Node node{.stats = {.visits = 2},
                  .edges = {MakeEdge(kFileA, 0.5f), MakeEdge(kFileB, 0.3f)},
                  .tried = {tried}};
  // Started totals: sqrt(N) = 2, edge A scores 0.25 + 0.5 * 2 / 5 = 0.45 and
  // edge B 0.3 * 2 = 0.6. (Counting only completed visits, sqrt(N) = 1 would
  // pick A: 0.35 against 0.3.)
  EXPECT_EQ(PuctAveraging::PickEdgeToVisit(node, TestParams()), 1u);
}

TEST(Lc5PuctTest, MovesLeftPrefersShorterWinsAndLongerLosses) {
  auto params = TestParams();
  params.moves_left.enabled = true;
  const auto select = [&](float node_q, float edge_q) {
    // Equal priors and visits, so without moves-left the tie goes to edge A.
    // The node has M 20; edge A's child has M 30 (10 plies longer) and edge
    // B's child M 10 (10 plies shorter).
    const Node node{
        .stats = {.visits = 3, .q = node_q, .m = 20.0f},
        .edges = {MakeEdge(kFileA, 0.5f), MakeEdge(kFileB, 0.5f)},
        .tried = {MakeTried(1, edge_q, 31.0f), MakeTried(1, edge_q, 11.0f)}};
    return PuctAveraging::PickEdgeToVisit(node, params);
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

TEST(Lc5PuctTest, ApplyMessageAveragesAndFlipsForParent) {
  Node node{.stats = {.visits = 1, .q = 0.5f, .d = 0.5f, .m = 2.0f},
            .edges = {MakeEdge(kFileA, 1.0f)},
            .tried = {Tried{}}};
  const PuctAveraging::Message message{
      .value = {.q = -0.5f, .d = 0.25f, .m = 4.0f}};
  const auto up = PuctAveraging::ApplyMessage(node, &node.tried[0], message);
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
  EXPECT_EQ(up.bounds, GameResultBounds{});

  PuctAveraging::ApplyMessage(node, nullptr, message);
  EXPECT_EQ(node.stats.visits, 3u);
  EXPECT_EQ(edge.visits, 1u);
  const auto skipped = PuctAveraging::MakeSkipMessage(message);
  EXPECT_FLOAT_EQ(skipped.value.q, 0.5f);
  EXPECT_FLOAT_EQ(skipped.value.m, 5.0f);
}

TEST(Lc5PuctTest, MakeCurrentMessageSendsNodeMeanAndBoundsFlipped) {
  lc5::Node<PuctDag> node{.stats = {.visits = 4,
                                    .q = 0.25f,
                                    .d = 0.5f,
                                    .m = 6.0f,
                                    .bounds = {.lower = 0, .upper = 1}},
                          .edges = {MakeEdge(kFileA, 1.0f)},
                          .tried = {TriedEdge<PuctDag>{}}};
  const auto up = PuctDag::MakeCurrentMessage(node);
  EXPECT_FLOAT_EQ(up.value.q, -0.25f);
  EXPECT_FLOAT_EQ(up.value.d, 0.5f);
  EXPECT_FLOAT_EQ(up.value.m, 7.0f);
  EXPECT_EQ(up.bounds, (GameResultBounds{.lower = -1, .upper = 0}));
}

// A node with `edges` edges, of which the first ones are tried with one visit
// each and `bounds`, in the node's frame.
Node MakeBoundedNode(size_t edges, std::vector<GameResultBounds> bounds) {
  Node node{.stats = {.visits = 1 + static_cast<uint32_t>(bounds.size())}};
  for (size_t i = 0; i < edges; ++i) {
    node.edges.push_back(MakeEdge(File::FromIdx(i), 1.0f / edges));
  }
  for (const GameResultBounds& b : bounds) {
    node.tried.push_back(MakeTried(1, b.lower));
    node.tried.back().stats.bounds = b;
  }
  return node;
}

constexpr GameResultBounds kWin{.lower = 1, .upper = 1};
constexpr GameResultBounds kDraw{.lower = 0, .upper = 0};
constexpr GameResultBounds kLoss{.lower = -1, .upper = -1};
GameResultBounds WithPlies(GameResultBounds bounds, uint8_t plies) {
  bounds.plies = plies;
  return bounds;
}

TEST(Lc5PuctTest, TerminalLossProvesParentWin) {
  // The child is checkmated: its install leaf is exact.
  Node child{.terminal = TerminalKind::kCheckmate};
  const auto leaf = PuctAveraging::MakeLeafMessage(
      TerminalValue(TerminalKind::kCheckmate), true);
  const auto from_child = PuctAveraging::ApplyMessage(child, nullptr, leaf);
  EXPECT_EQ(child.stats.bounds, kLoss);
  EXPECT_EQ(from_child.bounds, WithPlies(kWin, 1));

  // A parent with an untried edge left is still proven by one win.
  Node parent = MakeBoundedNode(2, {GameResultBounds{}});
  const auto up =
      PuctAveraging::ApplyMessage(parent, &parent.tried[0], from_child);
  EXPECT_EQ(parent.tried[0].stats.bounds, WithPlies(kWin, 1));
  EXPECT_EQ(parent.stats.bounds, WithPlies(kWin, 1));
  EXPECT_EQ(up.bounds, WithPlies(kLoss, 2));
}

TEST(Lc5PuctTest, NodeProvesBoundsFromItsEdges) {
  const auto prove = [](size_t edges, std::vector<GameResultBounds> bounds) {
    // Re-sends the last edge's bounds as a change.
    Node node = MakeBoundedNode(edges, bounds);
    const GameResultBounds last = node.tried.back().stats.bounds;
    node.tried.back().stats.bounds = {};
    PuctAveraging::ApplyMessage(
        node, &node.tried.back(),
        {.value = {.q = static_cast<float>(last.lower)}, .bounds = last});
    return node.stats.bounds;
  };
  // All edges lose: the node loses, by the longest one.
  EXPECT_EQ(prove(2, {WithPlies(kLoss, 3), WithPlies(kLoss, 5)}),
            WithPlies(kLoss, 5));
  // An untried edge is unknown.
  EXPECT_EQ(prove(3, {WithPlies(kLoss, 3), WithPlies(kLoss, 5)}),
            (GameResultBounds{.lower = -1, .upper = 1}));
  // The shortest win.
  EXPECT_EQ(prove(3, {WithPlies(kWin, 7), WithPlies(kWin, 3)}),
            WithPlies(kWin, 3));
  EXPECT_EQ(prove(2, {kDraw, WithPlies(kLoss, 3)}), kDraw);
  EXPECT_EQ(prove(2, {kDraw, GameResultBounds{.lower = -1, .upper = 0}}),
            kDraw);
  EXPECT_EQ(prove(2, {GameResultBounds{}, kDraw}),
            (GameResultBounds{.lower = 0, .upper = 1}));
}

TEST(Lc5PuctTest, StaleDropsBoundsAndSkipKeepsThem) {
  const PuctAveraging::Message message{.value = {.q = -1.0f}, .bounds = kLoss};
  EXPECT_EQ(PuctAveraging::MakeSkipMessage(message).bounds, WithPlies(kWin, 1));
  EXPECT_EQ(PuctAveraging::MakeStaleMessage(message).bounds,
            GameResultBounds{});
  EXPECT_FLOAT_EQ(PuctAveraging::MakeStaleMessage(message).value.q, 1.0f);
  // Only exact leaves are proven.
  EXPECT_EQ(PuctAveraging::MakeLeafMessage({.d = 1.0f}, true).bounds, kDraw);
  EXPECT_EQ(PuctAveraging::MakeLeafMessage({.d = 1.0f}, false).bounds,
            GameResultBounds{});
}

TEST(Lc5PuctTest, StickySelectionUsesBounds) {
  auto params = TestParams();
  // Edge A has the higher prior and Q but is a proven loss.
  Node node = MakeBoundedNode(2, {kLoss, GameResultBounds{}});
  node.edges[0].prior = 0.9f;
  node.edges[1].prior = 0.1f;
  node.tried[0].stats.q = 0.5f;
  node.tried[1].stats.q = 0.0f;
  EXPECT_EQ(PuctAveraging::PickEdgeToVisit(node, params), 0u);
  params.sticky_endgames = true;
  EXPECT_EQ(PuctAveraging::PickEdgeToVisit(node, params), 1u);

  // Edge B's Q -0.5 is clamped to its lower bound 0, which beats A's -0.1
  // (equal priors and visits, so U ties).
  node = MakeBoundedNode(
      2, {GameResultBounds{}, GameResultBounds{.lower = 0, .upper = 1}});
  node.tried[0].stats.q = -0.1f;
  node.tried[1].stats.q = -0.5f;
  EXPECT_EQ(PuctAveraging::PickEdgeToVisit(node, params), 1u);
  params.sticky_endgames = false;
  EXPECT_EQ(PuctAveraging::PickEdgeToVisit(node, params), 0u);
}

TEST(Lc5PuctTest, StickyGetBestMovePrefersShortWinsAndLongLosses) {
  auto params = TestParams();
  params.sticky_endgames = true;
  // A proven win with fewer visits beats a more-visited unknown edge.
  Node node = MakeBoundedNode(2, {GameResultBounds{}, WithPlies(kWin, 5)});
  node.tried[0].stats.visits = 10;
  EXPECT_EQ(PuctAveraging::GetBestMove(node, params), node.edges[1].move);
  EXPECT_EQ(PuctAveraging::GetBestMove(node, TestParams()), node.edges[0].move);
  // Of two proven wins, the shorter one.
  node = MakeBoundedNode(2, {WithPlies(kWin, 5), WithPlies(kWin, 3)});
  node.tried[0].stats.visits = 10;
  EXPECT_EQ(PuctAveraging::GetBestMove(node, params), node.edges[1].move);
  // A proven loss is chosen only if all edges lose; then the longest.
  node = MakeBoundedNode(2, {WithPlies(kLoss, 1), GameResultBounds{}});
  node.tried[0].stats.visits = 10;
  EXPECT_EQ(PuctAveraging::GetBestMove(node, params), node.edges[1].move);
  node = MakeBoundedNode(2, {WithPlies(kLoss, 1), WithPlies(kLoss, 3)});
  node.tried[0].stats.visits = 10;
  EXPECT_EQ(PuctAveraging::GetBestMove(node, params), node.edges[1].move);
}

TEST(Lc5PuctTest, GetProvenNodeValueNeedsStickyEndgames) {
  auto params = TestParams();
  Node node = MakeBoundedNode(1, {WithPlies(kWin, 3)});
  node.stats.bounds = WithPlies(kWin, 3);
  EXPECT_EQ(PuctAveraging::GetProvenNodeValue(node, params), std::nullopt);
  EXPECT_FALSE(PuctAveraging::IsNodeDecided(node, params));
  params.sticky_endgames = true;
  const auto value = PuctAveraging::GetProvenNodeValue(node, params);
  ASSERT_TRUE(value);
  EXPECT_EQ(value->q, 1.0f);
  EXPECT_EQ(value->d, 0.0f);
  EXPECT_EQ(value->m, 3.0f);
  EXPECT_TRUE(PuctAveraging::IsNodeDecided(node, params));
  node.stats.bounds = {.lower = 0, .upper = 1};
  EXPECT_EQ(PuctAveraging::GetProvenNodeValue(node, params), std::nullopt);
}

TEST(Lc5PuctTest, GetBestMoveRanksVisitsThenQThenPrior) {
  Node node{.edges = {MakeEdge(kFileA, 0.5f), MakeEdge(kFileB, 0.3f),
                      MakeEdge(kFileC, 0.2f)}};
  EXPECT_EQ(PuctAveraging::GetBestMove(node, TestParams()), std::nullopt);

  // Tried edges without a visit don't count.
  node.tried = {Tried{.in_flight = 1}, Tried{}, MakeTried(1, -0.9f)};
  EXPECT_EQ(PuctAveraging::GetBestMove(node, TestParams()), node.edges[2].move);

  node.tried[1].stats = {.visits = 2, .q = -0.5f};
  node.tried[2].stats = {.visits = 2, .q = 0.1f};
  EXPECT_EQ(PuctAveraging::GetBestMove(node, TestParams()), node.edges[2].move);

  node.tried[1].stats.q = 0.1f;
  EXPECT_EQ(PuctAveraging::GetBestMove(node, TestParams()), node.edges[1].move);

  node.tried[0].stats = {.visits = 3, .q = -1.0f};
  EXPECT_EQ(PuctAveraging::GetBestMove(node, TestParams()), node.edges[0].move);
}

TEST(Lc5PuctTest, GetBestMoveFallsBackToTopPriorWhenAllowed) {
  Node node;
  EXPECT_EQ(PuctAveraging::GetBestMove(node, TestParams(), true), std::nullopt);

  node.edges = {MakeEdge(kFileC, 0.5f), MakeEdge(kFileA, 0.3f),
                MakeEdge(kFileB, 0.2f)};
  EXPECT_EQ(PuctAveraging::GetBestMove(node, TestParams()), std::nullopt);
  EXPECT_EQ(PuctAveraging::GetBestMove(node, TestParams(), true),
            node.edges[0].move);

  node.tried = {Tried{}, Tried{}, MakeTried(1)};
  EXPECT_EQ(PuctAveraging::GetBestMove(node, TestParams(), true),
            node.edges[2].move);
}

TEST(Lc5PuctTest, FillThinkingInfoNeedsVisits) {
  const auto describe = [](ValueStats stats, ThinkingInfo& info) {
    PuctAveraging::FillThinkingInfo(Node{.stats = stats}, TestParams(), info);
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

TEST(Lc5PuctTest, FillThinkingInfoShowsProvenMates) {
  auto params = TestParams();
  params.sticky_endgames = true;
  const auto describe = [&](GameResultBounds bounds) {
    ThinkingInfo info;
    PuctAveraging::FillThinkingInfo(
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

TEST(Lc5PuctTest, PoliciesShareThePuctOptions) {
  OptionsParser parser;
  PopulatePolicyOptions(&parser);
  parser.SetUciOption("Lc5Policy", "puct-dag");
  parser.SetUciOption("CPuct", "3");
  EXPECT_EQ(PolicyName(parser.GetOptionsDict()), "puct-dag");
  EXPECT_EQ(PuctDag::Params::FromOptions(parser.GetOptionsDict(), kMlh).cpuct,
            3.0f);
  EXPECT_EQ(MakeVisitEngine("puct")->GetName(), "puct");
  EXPECT_EQ(MakeVisitEngine("puct-dag")->GetName(), "puct-dag");
}

using DagNode = lc5::Node<PuctDag>;
using DagTried = TriedEdge<PuctDag>;

TEST(Lc5PuctTest, DagEdgesCacheTheChildValue) {
  DagNode node{.edges = {MakeEdge(kFileA, 0.5f), MakeEdge(kFileB, 0.5f)},
               .tried = {DagTried{}, DagTried{}}};
  const VisitValue leaf{.q = 0.2f, .d = 0.4f, .m = 10.0f};
  PuctDag::ApplyMessage(node, nullptr, PuctDag::MakeLeafMessage(leaf, false));
  EXPECT_EQ(node.stats.visits, 1u);
  EXPECT_FLOAT_EQ(node.stats.Q(), 0.2f);

  // Visits through the edges with changing child values, in the edge frame.
  struct Step {
    size_t edge;
    VisitValue value;
  };
  const Step steps[] = {
      {0, {.q = 0.5f, .d = 0.3f, .m = 8.0f}},
      {1, {.q = -0.4f, .d = 0.6f, .m = 12.0f}},
      {0, {.q = 0.1f, .d = 0.2f, .m = 7.0f}},
      {0, {.q = -0.7f, .d = 0.1f, .m = 6.0f}},
      {1, {.q = 0.9f, .d = 0.05f, .m = 3.0f}},
      {1, {.q = 0.3f, .d = 0.5f, .m = 9.0f}},
  };
  for (const Step& step : steps) {
    const uint32_t before = node.tried[step.edge].stats.visits;
    const auto up = PuctDag::ApplyMessage(node, &node.tried[step.edge],
                                          {.value = step.value});
    const ValueStats& edge = node.tried[step.edge].stats;
    EXPECT_EQ(edge.visits, before + 1);
    EXPECT_EQ(edge.Q(), step.value.q);
    EXPECT_EQ(edge.D(), step.value.d);
    EXPECT_EQ(edge.M(), step.value.m);
    // The node is its own sample plus each edge's cached value, weighted by
    // the edge's visits.
    float visits = 1.0f, q = leaf.q, d = leaf.d, m = leaf.m;
    for (const DagTried& tried : node.tried) {
      visits += tried.stats.visits;
      q += tried.stats.visits * tried.stats.Q();
      d += tried.stats.visits * tried.stats.D();
      m += tried.stats.visits * tried.stats.M();
    }
    EXPECT_EQ(node.stats.visits, static_cast<uint32_t>(visits));
    EXPECT_NEAR(node.stats.Q(), q / visits, 1e-5f);
    EXPECT_NEAR(node.stats.D(), d / visits, 1e-5f);
    EXPECT_NEAR(node.stats.M(), m / visits, 1e-4f);
    const auto current = PuctDag::MakeCurrentMessage(node);
    EXPECT_EQ(up.value.q, current.value.q);
    EXPECT_EQ(up.value.d, current.value.d);
    EXPECT_EQ(up.value.m, current.value.m);
    EXPECT_EQ(up.bounds, current.bounds);
  }
}

TEST(Lc5PuctTest, DagApplyMessageReplacesTheEdgeValue) {
  // Edge A has 3 visits at 0.8; a fourth visit brings 0.2. With N = 4, sqrt(N)
  // = 2: A scores q + 0.5 * 2 / 5 = q + 0.2 and the untried B 0.3 * 2 = 0.6.
  // Averaging gives A q = 0.65 (0.85 > 0.6); replacing gives 0.2 (0.4 < 0.6).
  Node node{.stats = {.visits = 4, .q = 0.8f},
            .edges = {MakeEdge(kFileA, 0.5f), MakeEdge(kFileB, 0.3f)},
            .tried = {MakeTried(3, 0.8f)}};
  DagNode dag{.stats = {.visits = 4, .q = 0.8f},
              .edges = {MakeEdge(kFileA, 0.5f), MakeEdge(kFileB, 0.3f)},
              .tried = {DagTried{.stats = {.visits = 3, .q = 0.8f}}}};
  const VisitValue from_child{.q = 0.2f};
  PuctAveraging::ApplyMessage(node, &node.tried[0], {.value = from_child});
  PuctDag::ApplyMessage(dag, &dag.tried[0], {.value = from_child});
  EXPECT_EQ(PuctAveraging::PickEdgeToVisit(node, TestParams()), 0u);
  EXPECT_EQ(PuctDag::PickEdgeToVisit(dag, TestParams()), 1u);
}

TEST(Lc5PuctTest, DagTerminalLossProvesParentWin) {
  DagNode child{.terminal = TerminalKind::kCheckmate};
  const auto from_child = PuctDag::ApplyMessage(
      child, nullptr,
      PuctDag::MakeLeafMessage(TerminalValue(TerminalKind::kCheckmate), true));
  EXPECT_EQ(child.stats.bounds, kLoss);
  EXPECT_EQ(from_child.bounds, WithPlies(kWin, 1));
  EXPECT_FLOAT_EQ(from_child.value.q, 1.0f);

  DagNode parent{.stats = {.visits = 1},
                 .edges = {MakeEdge(kFileA, 0.5f), MakeEdge(kFileB, 0.5f)},
                 .tried = {DagTried{}}};
  const auto up = PuctDag::ApplyMessage(parent, &parent.tried[0], from_child);
  EXPECT_EQ(parent.tried[0].stats.bounds, WithPlies(kWin, 1));
  EXPECT_EQ(parent.stats.bounds, WithPlies(kWin, 1));
  EXPECT_EQ(up.bounds, WithPlies(kLoss, 2));
}

}  // namespace
}  // namespace lczero::lc5

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
