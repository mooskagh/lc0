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
using Edge = EdgeState<PuctAveraging>;

Move M(File from, File to) {
  return Move::White(Square(from, kRank2), Square(to, kRank3));
}

Edge MakeEdge(File file, float prior, uint32_t visits = 0, float q = 0.0f) {
  return {.move = M(file, file),
          .prior = prior,
          .stats = {.visits = visits, .q = q}};
}

PuctAveraging::Params TestParams() {
  return {.cpuct = 1.0f,
          .cpuct_base = 100.0f,
          .cpuct_factor = 0.0f,
          .fpu_strategy = FpuStrategy::kAbsolute,
          .fpu_value = 0.0f,
          .restart_waiters = false};
}

TEST(Lc5PuctTest, OptionDefaultsMatchParams) {
  OptionsParser parser;
  PopulatePolicyOptions(&parser);
  EXPECT_EQ(PolicyName(parser.GetOptionsDict()), "puct");
  const auto params =
      PuctAveraging::Params::FromOptions(parser.GetOptionsDict());
  const PuctAveraging::Params defaults;
  EXPECT_EQ(params.cpuct, defaults.cpuct);
  EXPECT_EQ(params.cpuct_base, defaults.cpuct_base);
  EXPECT_EQ(params.cpuct_factor, defaults.cpuct_factor);
  EXPECT_EQ(params.fpu_strategy, FpuStrategy::kReduction);
  EXPECT_EQ(params.fpu_value, defaults.fpu_value);
  EXPECT_FALSE(params.restart_waiters);
  EXPECT_EQ(PuctAveraging::OnResume(params),
            VisitEngine::ResumeAction::kSelect);
}

TEST(Lc5PuctTest, OptionsReachParams) {
  OptionsParser parser;
  PopulatePolicyOptions(&parser);
  parser.SetUciOption("CPuct", "3");
  parser.SetUciOption("FpuStrategy", "absolute");
  parser.SetUciOption("RestartWaiters", "true");
  const auto params =
      PuctAveraging::Params::FromOptions(parser.GetOptionsDict());
  EXPECT_EQ(params.cpuct, 3.0f);
  EXPECT_EQ(params.fpu_strategy, FpuStrategy::kAbsolute);
  EXPECT_EQ(PuctAveraging::OnResume(params),
            VisitEngine::ResumeAction::kRestart);
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
                    .edges = {MakeEdge(kFileA, 0.5f, 1, q),
                              MakeEdge(kFileB, 0.3f), MakeEdge(kFileC, 0.2f)},
                    .started_end = 1};
    // The tried edge scores q + 0.25 and the first untried one 0.3.
    EXPECT_EQ(PuctAveraging::Select(node, TestParams()), q > 0.05f ? 0u : 1u);
  }
}

TEST(Lc5PuctTest, InFlightEdgeLosesUrgency) {
  Node node{.lifecycle = NodeLifecycle::kExpanded,
            .stats = {.visits = 1},
            .edges = {MakeEdge(kFileA, 0.5f), MakeEdge(kFileB, 0.3f)},
            .started_end = 1};
  // Edge A scores 0.5 / 2 with one visit in flight; edge B scores 0.3.
  node.edges[0].in_flight = 1;
  EXPECT_EQ(PuctAveraging::Select(node, TestParams()), 1u);
}

TEST(Lc5PuctTest, UpdateAveragesAndFlipsForParent) {
  ValueStats node{.visits = 1, .q = 0.5f, .d = 0.5f, .m = 2.0f};
  ValueStats edge{};
  const SearchValue message{.q = -0.5f, .d = 0.25f, .m = 4.0f};
  const SearchValue up = PuctAveraging::Update(node, &edge, message);
  EXPECT_EQ(node.visits, 2u);
  EXPECT_FLOAT_EQ(node.Q(), 0.0f);
  EXPECT_FLOAT_EQ(node.D(), 0.375f);
  EXPECT_FLOAT_EQ(node.M(), 3.0f);
  EXPECT_EQ(edge.visits, 1u);
  EXPECT_FLOAT_EQ(edge.Q(), -0.5f);
  EXPECT_FLOAT_EQ(up.q, 0.5f);
  EXPECT_FLOAT_EQ(up.d, 0.25f);
  EXPECT_FLOAT_EQ(up.m, 5.0f);

  PuctAveraging::Update(node, nullptr, message);
  EXPECT_EQ(node.visits, 3u);
  EXPECT_EQ(edge.visits, 1u);
  const SearchValue skipped = PuctAveraging::Skip(message);
  EXPECT_FLOAT_EQ(skipped.q, 0.5f);
  EXPECT_FLOAT_EQ(skipped.m, 5.0f);
}

TEST(Lc5PuctTest, ChooseEdgeRanksVisitsThenQThenPrior) {
  Node node{.lifecycle = NodeLifecycle::kExpanded,
            .edges = {MakeEdge(kFileA, 0.5f), MakeEdge(kFileB, 0.3f),
                      MakeEdge(kFileC, 0.2f)}};
  EXPECT_EQ(PuctAveraging::ChooseEdge(node), nullptr);

  node.edges[2].stats = {.visits = 1, .q = -0.9f};
  EXPECT_EQ(PuctAveraging::ChooseEdge(node), &node.edges[2]);

  node.edges[1].stats = {.visits = 2, .q = -0.5f};
  node.edges[2].stats = {.visits = 2, .q = 0.1f};
  EXPECT_EQ(PuctAveraging::ChooseEdge(node), &node.edges[2]);

  node.edges[1].stats.q = 0.1f;
  EXPECT_EQ(PuctAveraging::ChooseEdge(node), &node.edges[1]);

  node.edges[0].stats = {.visits = 3, .q = -1.0f};
  EXPECT_EQ(PuctAveraging::ChooseEdge(node), &node.edges[0]);
}

TEST(Lc5PuctTest, ChooseMoveFallsBackToTopPrior) {
  Node node{.lifecycle = NodeLifecycle::kExpanded};
  EXPECT_EQ(PuctAveraging::ChooseMove(node), nullptr);

  node.edges = {MakeEdge(kFileC, 0.5f), MakeEdge(kFileA, 0.3f),
                MakeEdge(kFileB, 0.2f)};
  EXPECT_EQ(PuctAveraging::ChooseEdge(node), nullptr);
  EXPECT_EQ(PuctAveraging::ChooseMove(node), &node.edges[0]);

  node.edges[2].stats = {.visits = 1};
  EXPECT_EQ(PuctAveraging::ChooseMove(node), &node.edges[2]);
}

TEST(Lc5PuctTest, DescribeNeedsVisits) {
  ThinkingInfo info;
  PuctAveraging::Describe(ValueStats{}, info);
  EXPECT_FALSE(info.score);
  EXPECT_FALSE(info.wdl);

  PuctAveraging::Describe({.visits = 1, .q = 0.0f, .d = 1.0f}, info);
  ASSERT_TRUE(info.score);
  EXPECT_EQ(*info.score, 0);
  ASSERT_TRUE(info.wdl);
  EXPECT_EQ(info.wdl->w, 0);
  EXPECT_EQ(info.wdl->d, 1000);
  EXPECT_EQ(info.wdl->l, 0);

  PuctAveraging::Describe({.visits = 1, .q = 0.5f}, info);
  EXPECT_GT(*info.score, 0);
  EXPECT_EQ(info.wdl->w, 750);
  EXPECT_EQ(info.wdl->l, 250);
}

}  // namespace
}  // namespace lczero::lc5

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
