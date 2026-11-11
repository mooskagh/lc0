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

#include "search/lc5/policies/node_serialization.h"

#include <gtest/gtest.h>

#include <cstddef>

#include "search/lc5/policies/puct.h"

namespace lczero::lc5 {
namespace {

Move M(File from, File to) {
  return Move::White(Square(from, kRank2), Square(to, kRank3));
}

// A node with two of its three edges tried, active reservations, visits and
// proven bounds.
Node<PuctAveraging> SampleNode() {
  Node<PuctAveraging> node{
      .version = 7,
      .in_flight = 2,
      .edges = {{.move = M(kFileA, kFileA), .prior = 0.5f},
                {.move = M(kFileB, kFileB), .prior = 0.3f},
                {.move = M(kFileC, kFileC), .prior = 0.2f}},
      .tried = {{.in_flight = 1}, {.in_flight = 1}}};
  PuctAveraging::Update(
      node, nullptr,
      PuctAveraging::Leaf({.q = 0.5f, .d = 0.25f, .m = 6.0f}, false));
  PuctAveraging::Update(
      node, &node.tried[0],
      PuctAveraging::Leaf({.q = -0.75f, .d = 0.25f, .m = 3.0f}, true));
  PuctAveraging::Update(
      node, &node.tried[1],
      PuctAveraging::Leaf({.q = -0.75f, .d = 0.25f, .m = 3.0f}, false));
  return node;
}

TEST(Lc5NodeSerializationTest, RoundTripKeepsStatisticsAndEdges) {
  const auto node = SampleNode();
  ASSERT_EQ(node.tried.size(), 2u);
  const auto deserialized = DeserializeNode<PuctAveraging>(SerializeNode(node));
  EXPECT_EQ(deserialized.version, 0u);
  EXPECT_EQ(deserialized.in_flight, 0u);
  EXPECT_EQ(deserialized.terminal, node.terminal);
  EXPECT_EQ(deserialized.stats.visits, node.stats.visits);
  EXPECT_EQ(deserialized.stats.Q(), node.stats.Q());
  EXPECT_EQ(deserialized.stats.D(), node.stats.D());
  EXPECT_EQ(deserialized.stats.M(), node.stats.M());
  EXPECT_EQ(deserialized.stats.bounds, node.stats.bounds);
  ASSERT_EQ(deserialized.edges.size(), node.edges.size());
  for (size_t i = 0; i < node.edges.size(); ++i) {
    EXPECT_EQ(deserialized.edges[i].move, node.edges[i].move);
    EXPECT_EQ(deserialized.edges[i].prior.bits, node.edges[i].prior.bits);
  }
  ASSERT_EQ(deserialized.tried.size(), node.tried.size());
  for (size_t i = 0; i < node.tried.size(); ++i) {
    EXPECT_EQ(deserialized.tried[i].in_flight, 0u);
    EXPECT_EQ(deserialized.tried[i].stats.visits, node.tried[i].stats.visits);
    EXPECT_EQ(deserialized.tried[i].stats.Q(), node.tried[i].stats.Q());
    EXPECT_EQ(deserialized.tried[i].stats.bounds, node.tried[i].stats.bounds);
  }
  // The first backup was exact, so that edge is proven.
  EXPECT_TRUE(deserialized.tried[0].stats.bounds.Proven());

  // A terminal node has no edges.
  Node<PuctAveraging> terminal_node{.terminal = TerminalKind::kCheckmate};
  PuctAveraging::Update(
      terminal_node, nullptr,
      PuctAveraging::Leaf(TerminalValue(TerminalKind::kCheckmate), true));
  const auto terminal =
      DeserializeNode<PuctAveraging>(SerializeNode(terminal_node));
  EXPECT_EQ(terminal.terminal, TerminalKind::kCheckmate);
  EXPECT_TRUE(terminal.edges.empty());
  EXPECT_TRUE(terminal.tried.empty());
  EXPECT_EQ(terminal.stats.visits, 1u);
}

}  // namespace
}  // namespace lczero::lc5

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
