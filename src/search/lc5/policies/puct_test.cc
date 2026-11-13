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

#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "chess/bitboard.h"
#include "chess/board.h"
#include "chess/position.h"
#include "search/lc5/policies/registry.h"
#include "search/lc5/settings.h"
#include "search/lc5/visit_engine.h"

namespace lczero::lc5 {
namespace {

struct TestPolicy : PuctPolicy<TestPolicy> {};

using Node = lc5::Node<TestPolicy>;
using Tried = TriedEdge<TestPolicy>;

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

PuctParams TestParams() {
  return {.cpuct = 1.0f,
          .cpuct_base = 100.0f,
          .cpuct_factor = 0.0f,
          .fpu_strategy = FpuStrategy::kAbsolute,
          .fpu_value = 0.0f,
          .sticky_endgames = false};
}

std::unique_ptr<VisitEngine> MakeConfiguredEngine(
    std::string_view name, bool sticky_endgames = false) {
  OptionsParser parser;
  Settings::Populate(&parser);
  PopulatePolicyOptions(&parser);
  parser.SetUciOption("CPuct", "1");
  parser.SetUciOption("CPuctBase", "100");
  parser.SetUciOption("CPuctFactor", "0");
  parser.SetUciOption("FpuStrategy", "absolute");
  parser.SetUciOption("FpuValue", "0");
  parser.SetUciOption("MovesLeft", "false");
  parser.SetUciOption("StickyEndgames", sticky_endgames ? "true" : "false");
  auto engine = MakeVisitEngine(name);
  engine->Configure(parser.GetOptionsDict(), kMlh);
  return engine;
}

VisitOrigin StartOrigin() {
  PositionHistory history;
  history.Reset(Position::FromFen(ChessBoard::kStartposFen));
  return {.key = MakeNodeKey(history, 7),
          .history = std::move(history),
          .backup_prefix = {}};
}

TEST(Lc5PuctTest, OptionDefaultsMatchParams) {
  OptionsParser parser;
  PopulatePolicyOptions(&parser);
  EXPECT_EQ(PolicyName(parser.GetOptionsDict()), "puct");
  const auto params = PuctParams::FromOptions(parser.GetOptionsDict(), kMlh);
  const PuctParams defaults;
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
  EXPECT_TRUE(params.sticky_endgames);
}

TEST(Lc5PuctTest, OptionsReachParams) {
  OptionsParser parser;
  PopulatePolicyOptions(&parser);
  parser.SetUciOption("CPuct", "3");
  parser.SetUciOption("FpuStrategy", "absolute");
  parser.SetUciOption("MovesLeft", "true");
  parser.SetUciOption("MovesLeftSlope", "0.01");
  parser.SetUciOption("StickyEndgames", "false");
  const auto params = PuctParams::FromOptions(parser.GetOptionsDict(), kMlh);
  EXPECT_EQ(params.cpuct, 3.0f);
  EXPECT_EQ(params.fpu_strategy, FpuStrategy::kAbsolute);
  EXPECT_TRUE(params.moves_left.enabled);
  EXPECT_EQ(params.moves_left.slope, 0.01f);
  EXPECT_FALSE(params.sticky_endgames);
  // Without a moves-left head the option has no effect.
  EXPECT_FALSE(
      PuctParams::FromOptions(parser.GetOptionsDict(), MakeAttributes(false))
          .moves_left.enabled);
  parser.SetUciOption("MovesLeft", "false");
  EXPECT_FALSE(PuctParams::FromOptions(parser.GetOptionsDict(), kMlh)
                   .moves_left.enabled);
}

TEST(Lc5PuctTest, UnstartedNodePicksTopPrior) {
  const Node node{.stats = {.visits = 1},
                  .edges = {MakeEdge(kFileA, 0.5f), MakeEdge(kFileB, 0.3f),
                            MakeEdge(kFileC, 0.2f)}};
  EXPECT_EQ(TestPolicy::PickEdgeToVisit(node, TestParams()), 0u);
}

TEST(Lc5PuctTest, FirstUntriedEdgeCompetesWithTriedOnes) {
  for (const float q : {0.5f, -0.5f}) {
    SCOPED_TRACE(q);
    const Node node{.stats = {.visits = 2},
                    .edges = {MakeEdge(kFileA, 0.5f), MakeEdge(kFileB, 0.3f),
                              MakeEdge(kFileC, 0.2f)},
                    .tried = {MakeTried(1, q)}};
    // The tried edge scores q + 0.25 and the first untried one 0.3.
    EXPECT_EQ(TestPolicy::PickEdgeToVisit(node, TestParams()),
              q > 0.05f ? 0u : 1u);
  }
}

TEST(Lc5PuctTest, InFlightEdgeLosesUrgency) {
  Node node{.stats = {.visits = 1},
            .edges = {MakeEdge(kFileA, 0.5f), MakeEdge(kFileB, 0.3f)},
            .tried = {Tried{.in_flight = 1}}};
  // Edge A scores 0.5 / 2 with one visit in flight; edge B scores 0.3.
  EXPECT_EQ(TestPolicy::PickEdgeToVisit(node, TestParams()), 1u);
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
  EXPECT_EQ(TestPolicy::PickEdgeToVisit(node, TestParams()), 1u);
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
    return TestPolicy::PickEdgeToVisit(node, params);
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
  TestPolicy::InitNodeStats(child,
                            {.terminal = TerminalKind::kCheckmate,
                             .leaf = TerminalValue(TerminalKind::kCheckmate),
                             .moves = {},
                             .priors = {}});
  EXPECT_EQ(child.stats.visits, 1u);
  EXPECT_EQ(child.stats.bounds, kLoss);
  const GameResultBounds from_child = child.stats.bounds.Parent();
  EXPECT_EQ(from_child, WithPlies(kWin, 1));

  // A parent with an untried edge left is still proven by one win.
  Node parent = MakeBoundedNode(2, {GameResultBounds{}});
  TestPolicy::UpdateBounds(parent, &parent.tried[0], from_child);
  EXPECT_EQ(parent.tried[0].stats.bounds, WithPlies(kWin, 1));
  EXPECT_EQ(parent.stats.bounds, WithPlies(kWin, 1));
  EXPECT_EQ(parent.stats.bounds.Parent(), WithPlies(kLoss, 2));
}

TEST(Lc5PuctTest, NodeProvesBoundsFromItsEdges) {
  const auto prove = [](size_t edges, std::vector<GameResultBounds> bounds) {
    // Re-sends the last edge's bounds as a change.
    Node node = MakeBoundedNode(edges, bounds);
    const GameResultBounds last = node.tried.back().stats.bounds;
    node.tried.back().stats.bounds = {};
    TestPolicy::UpdateBounds(node, &node.tried.back(), last);
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
  const TestPolicy::BackupValue backup{.value = {.q = -1.0f}, .bounds = kLoss};
  EXPECT_EQ(TestPolicy::MakeSkipBackup(backup).bounds, WithPlies(kWin, 1));
  EXPECT_EQ(TestPolicy::MakeStaleBackup(backup).bounds, GameResultBounds{});
  EXPECT_FLOAT_EQ(TestPolicy::MakeStaleBackup(backup).value.q, 1.0f);
  // Only exact leaves are proven.
  EXPECT_EQ(TestPolicy::MakeLeafBackup({.d = 1.0f}, true).bounds, kDraw);
  EXPECT_EQ(TestPolicy::MakeLeafBackup({.d = 1.0f}, false).bounds,
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
  EXPECT_EQ(TestPolicy::PickEdgeToVisit(node, params), 0u);
  params.sticky_endgames = true;
  EXPECT_EQ(TestPolicy::PickEdgeToVisit(node, params), 1u);

  // Edge B's Q -0.5 is clamped to its lower bound 0, which beats A's -0.1
  // (equal priors and visits, so U ties).
  node = MakeBoundedNode(
      2, {GameResultBounds{}, GameResultBounds{.lower = 0, .upper = 1}});
  node.tried[0].stats.q = -0.1f;
  node.tried[1].stats.q = -0.5f;
  EXPECT_EQ(TestPolicy::PickEdgeToVisit(node, params), 1u);
  params.sticky_endgames = false;
  EXPECT_EQ(TestPolicy::PickEdgeToVisit(node, params), 0u);
}

TEST(Lc5PuctTest, StickyGetBestMovePrefersShortWinsAndLongLosses) {
  auto params = TestParams();
  params.sticky_endgames = true;
  // A proven win with fewer visits beats a more-visited unknown edge.
  Node node = MakeBoundedNode(2, {GameResultBounds{}, WithPlies(kWin, 5)});
  node.tried[0].stats.visits = 10;
  EXPECT_EQ(TestPolicy::GetBestMove(node, params), node.edges[1].move);
  EXPECT_EQ(TestPolicy::GetBestMove(node, TestParams()), node.edges[0].move);
  // Of two proven wins, the shorter one.
  node = MakeBoundedNode(2, {WithPlies(kWin, 5), WithPlies(kWin, 3)});
  node.tried[0].stats.visits = 10;
  EXPECT_EQ(TestPolicy::GetBestMove(node, params), node.edges[1].move);
  // A proven loss is chosen only if all edges lose; then the longest.
  node = MakeBoundedNode(2, {WithPlies(kLoss, 1), GameResultBounds{}});
  node.tried[0].stats.visits = 10;
  EXPECT_EQ(TestPolicy::GetBestMove(node, params), node.edges[1].move);
  node = MakeBoundedNode(2, {WithPlies(kLoss, 1)});
  node.tried[0].stats.visits = 10;
  EXPECT_EQ(TestPolicy::GetBestMove(node, params), node.edges[1].move);
  node = MakeBoundedNode(2, {WithPlies(kLoss, 1), WithPlies(kLoss, 3)});
  node.tried[0].stats.visits = 10;
  EXPECT_EQ(TestPolicy::GetBestMove(node, params), node.edges[1].move);
}

TEST(Lc5PuctTest, GetProvenNodeValueNeedsStickyEndgames) {
  auto params = TestParams();
  Node node = MakeBoundedNode(1, {WithPlies(kWin, 3)});
  node.stats.bounds = WithPlies(kWin, 3);
  EXPECT_EQ(TestPolicy::GetProvenNodeValue(node, params), std::nullopt);
  EXPECT_FALSE(TestPolicy::IsNodeDecided(node, params));
  params.sticky_endgames = true;
  const auto value = TestPolicy::GetProvenNodeValue(node, params);
  ASSERT_TRUE(value);
  EXPECT_EQ(value->q, 1.0f);
  EXPECT_EQ(value->d, 0.0f);
  EXPECT_EQ(value->m, 3.0f);
  EXPECT_TRUE(TestPolicy::IsNodeDecided(node, params));
  node.stats.bounds = WithPlies(kLoss, 2);
  EXPECT_FALSE(TestPolicy::IsNodeDecided(node, params));
  node.stats.bounds = kDraw;
  EXPECT_FALSE(TestPolicy::IsNodeDecided(node, params));
  node.stats.bounds = {.lower = 0, .upper = 1};
  EXPECT_EQ(TestPolicy::GetProvenNodeValue(node, params), std::nullopt);
}

TEST(Lc5PuctTest, GetBestMoveRanksVisitsThenQThenPrior) {
  Node node{.edges = {MakeEdge(kFileA, 0.5f), MakeEdge(kFileB, 0.3f),
                      MakeEdge(kFileC, 0.2f)}};
  // With no visits yet, falls back to the highest-prior edge.
  EXPECT_EQ(TestPolicy::GetBestMove(node, TestParams()), node.edges[0].move);
  node.tried = {Tried{.in_flight = 1}};
  EXPECT_EQ(TestPolicy::GetBestMove(node, TestParams()), node.edges[0].move);

  // Tried edges without a visit don't beat a visited edge.
  node.tried = {Tried{.in_flight = 1}, Tried{}, MakeTried(1, -0.9f)};
  EXPECT_EQ(TestPolicy::GetBestMove(node, TestParams()), node.edges[2].move);

  node.tried[1].stats = {.visits = 2, .q = -0.5f};
  node.tried[2].stats = {.visits = 2, .q = 0.1f};
  EXPECT_EQ(TestPolicy::GetBestMove(node, TestParams()), node.edges[2].move);

  node.tried[1].stats.q = 0.1f;
  EXPECT_EQ(TestPolicy::GetBestMove(node, TestParams()), node.edges[1].move);

  node.tried[0].stats = {.visits = 3, .q = -1.0f};
  EXPECT_EQ(TestPolicy::GetBestMove(node, TestParams()), node.edges[0].move);
}

TEST(Lc5PuctTest, WdlFromValueInvariantsAndEdgeCases) {
  const auto check = [](float q, float d) {
    const auto wdl = WdlFromValue(q, d);
    EXPECT_GE(wdl.w, 0);
    EXPECT_LE(wdl.w, 1000);
    EXPECT_GE(wdl.d, 0);
    EXPECT_LE(wdl.d, 1000);
    EXPECT_GE(wdl.l, 0);
    EXPECT_LE(wdl.l, 1000);
    EXPECT_EQ(wdl.w + wdl.d + wdl.l, 1000);
    const auto flipped = WdlFromValue(-q, d);
    EXPECT_EQ(wdl.w, flipped.l);
    EXPECT_EQ(wdl.d, flipped.d);
    EXPECT_EQ(wdl.l, flipped.w);
    return std::tuple(wdl.w, wdl.d, wdl.l);
  };

  // High-draw edge case: q > 0 should not flip to L > W by truncation.
  EXPECT_EQ(check(0.0005f, 0.999f), std::tuple(1, 999, 0));
  EXPECT_EQ(check(-0.0005f, 0.999f), std::tuple(0, 999, 1));

  // Float representation slightly below integer (0.29f * 1000 < 290) and
  // odd-draw equal positions (0.375 -> 375 permille) keep W == L.
  EXPECT_EQ(check(0.0f, 0.29f), std::tuple(355, 290, 355));
  EXPECT_EQ(check(0.0f, 0.375f), std::tuple(313, 374, 313));
  check(0.0f, 0.003f);

  // Exact .5 ties on both W and L when d == 0 (562.5 + 437.5).
  EXPECT_EQ(check(0.125f, 0.0f), std::tuple(562, 0, 438));

  // Out-of-simplex / out-of-range float drift still stays in [0, 1000] and
  // sums to 1000.
  EXPECT_EQ(check(0.003f, 0.999f), std::tuple(2, 998, 0));
  EXPECT_EQ(check(-0.003f, 0.999f), std::tuple(0, 998, 2));
  EXPECT_EQ(check(0.002f, 1.0f), std::tuple(1, 999, 0));
  EXPECT_EQ(check(1.5f, -0.1f), std::tuple(1000, 0, 0));

  for (int qi = -120; qi <= 120; ++qi) {
    for (int di = -10; di <= 110; ++di) {
      check(qi * 0.01f, di * 0.01f);
    }
  }
}

TEST(Lc5PuctTest, FillThinkingInfoNeedsVisits) {
  const auto describe = [](ValueStats stats, ThinkingInfo& info) {
    TestPolicy::FillThinkingInfo(Node{.stats = stats}, TestParams(), info);
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
    TestPolicy::FillThinkingInfo(
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
  EXPECT_EQ(PuctParams::FromOptions(parser.GetOptionsDict(), kMlh).cpuct, 3.0f);
  EXPECT_EQ(MakeVisitEngine("puct")->GetName(), "puct");
  EXPECT_EQ(MakeVisitEngine("puct-dag")->GetName(), "puct-dag");
}

Visit MakeVisit(const VisitOrigin& origin) {
  Visit visit;
  visit.origin = origin;
  visit.history = origin.history;
  visit.current_key = origin.key;
  visit.path = origin.backup_prefix;
  return visit;
}

TEST(Lc5PuctTest, PuctAveragesWhileDagReplacesEdgeValue) {
  // Edge A gets 3 visits at +0.8 (child -0.8), then a 4th visit bringing +0.2
  // (child -0.2). With N = 4, sqrt(N) = 2: A scores q + 0.5 * 2 / 5 = q + 0.2
  // and untried B scores 0.3 * 2 = 0.6.
  // "puct" averages A to 0.65 (0.85 > 0.6, root mean 0.68); "puct-dag" replaces
  // A with 0.2 (0.4 < 0.6, root mean (0.8 + 4 * 0.2) / 5 = 0.32).
  const VisitOrigin root = StartOrigin();
  const Move move_a = M(kFileA, kFileA);
  const Move move_b = M(kFileB, kFileB);
  Metrics metrics;

  const auto run = [&](std::string_view name) {
    auto engine = MakeConfiguredEngine(name);
    EXPECT_TRUE(
        engine->CompleteMaterialization(root.key, {.leaf = {.q = 0.8f},
                                                   .moves = {move_a, move_b},
                                                   .priors = {0.5f, 0.3f}}));
    for (const float child_q : {-0.8f, -0.8f, -0.8f, -0.2f}) {
      Visit visit = MakeVisit(root);
      EXPECT_EQ(engine->DescendVisit(visit, metrics),
                VisitEngine::DescendResult::kUnmaterializedLeaf);
      EXPECT_EQ(visit.path.front().selected_move, move_a);
      visit.result = {.q = child_q};
      engine->BackupVisit(visit, metrics);
    }
    Visit next = MakeVisit(root);
    EXPECT_EQ(engine->DescendVisit(next, metrics),
              VisitEngine::DescendResult::kUnmaterializedLeaf);
    const Move selected = *next.path.front().selected_move;
    engine->CancelVisit(next);
    ThinkingInfo info;
    engine->FillThinkingInfo(root, info);
    return std::pair(selected, info.score.value_or(0));
  };

  EXPECT_EQ(run("puct"), std::pair(move_a, CentipawnsFromQ(0.68f)));
  EXPECT_EQ(run("puct-dag"), std::pair(move_b, CentipawnsFromQ(0.32f)));
}

TEST(Lc5PuctTest, DagForwardsSharedChildMeanToParent) {
  const VisitOrigin root = StartOrigin();
  const Move move_a = M(kFileA, kFileA);
  VisitOrigin child = root;
  child.history.Append(move_a);
  child.key = MakeNodeKey(child.history, 7);
  const Move reply = child.history.Last().GetBoard().GenerateLegalMoves()[0];
  Metrics metrics;

  const auto run = [&](std::string_view name) {
    auto engine = MakeConfiguredEngine(name);
    EXPECT_TRUE(engine->CompleteMaterialization(
        root.key, {.leaf = {.q = 0.0f}, .moves = {move_a}, .priors = {1.0f}}));
    // Child starts at -0.8 (which is +0.8 for root) via its own evaluation,
    // representing knowledge learned without traversing root's edge.
    EXPECT_TRUE(engine->CompleteMaterialization(
        child.key, {.leaf = {.q = -0.8f}, .moves = {reply}, .priors = {1.0f}}));
    // A visit from root goes through child to an unmaterialized grandchild
    // whose leaf sample is 0.0.
    Visit visit = MakeVisit(root);
    EXPECT_EQ(engine->DescendVisit(visit, metrics),
              VisitEngine::DescendResult::kUnmaterializedLeaf);
    EXPECT_EQ(visit.path.size(), 3u);
    visit.result = {.q = 0.0f};
    engine->BackupVisit(visit, metrics);
    ThinkingInfo info;
    engine->FillThinkingInfo(root, info);
    return info.score.value_or(0);
  };

  // "puct" forwards only the 0.0 grandchild sample; "puct-dag" forwards child's
  // current mean (-0.4 -> +0.4 at root, so root mean is +0.2).
  EXPECT_EQ(run("puct"), 0);
  EXPECT_EQ(run("puct-dag"), CentipawnsFromQ(0.2f));
}

TEST(Lc5PuctTest, BothPoliciesProveWinFromCheckmateChild) {
  const VisitOrigin root = StartOrigin();
  const Move move_a = M(kFileA, kFileA);
  const Move move_b = M(kFileB, kFileB);
  VisitOrigin child = root;
  child.history.Append(move_a);
  child.key = MakeNodeKey(child.history, 7);
  Metrics metrics;

  for (const std::string_view name : {"puct", "puct-dag"}) {
    SCOPED_TRACE(name);
    auto engine = MakeConfiguredEngine(name, /*sticky_endgames=*/true);
    EXPECT_TRUE(engine->CompleteMaterialization(
        root.key, {.moves = {move_a, move_b}, .priors = {0.5f, 0.5f}}));
    EXPECT_TRUE(engine->CompleteMaterialization(
        child.key, {.terminal = TerminalKind::kCheckmate,
                    .leaf = TerminalValue(TerminalKind::kCheckmate),
                    .moves = {},
                    .priors = {}}));
    Visit visit = MakeVisit(root);
    EXPECT_EQ(engine->DescendVisit(visit, metrics),
              VisitEngine::DescendResult::kValuedLeaf);
    engine->BackupVisit(visit, metrics);
    const RootStats stats = engine->GetRootStats(root.key);
    EXPECT_TRUE(stats.decided);
    EXPECT_EQ(stats.best_move, move_a);
    ThinkingInfo info;
    engine->FillThinkingInfo(root, info);
    EXPECT_EQ(info.mate, 1);
  }
}

}  // namespace
}  // namespace lczero::lc5

int main(int argc, char** argv) {
  lczero::InitializeMagicBitboards();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
