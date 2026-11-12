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

#pragma once

#include <algorithm>
#include <string_view>

#include "search/lc5/graph/graph.h"
#include "search/lc5/graph/value.h"
#include "search/lc5/policies/puct.h"

namespace lczero::lc5 {

// PUCT over a graph whose transpositions share their value: an edge caches its
// child's current value (from the last visit through it) instead of averaging
// the samples that went through it, and a node's value is the visit-weighted
// mean of its own samples and its edges' cached values. Every node sends its
// current value to its parent, so a parent sees everything its child learned,
// also through other parents.
//
// A cached value refreshes only when a visit goes through that edge. A
// two-fold draw is cached by the edge that reached it and reaches ancestors
// only through the nodes' values. With a short HistoryKeyLength, more positions
// share a node and its value.
struct PuctDag : PuctPolicy<PuctDag> {
  static constexpr std::string_view kName = "puct-dag";

  // Sent to the parent during backprop: the node's current mean and bounds.
  static Message MakeCurrentMessage(const Node<PuctDag>& node) {
    const ValueStats& stats = node.stats;
    return {VisitValue{.q = stats.Q(), .d = stats.D(), .m = stats.M()}.Parent(),
            stats.bounds.Parent()};
  }

  // Folds `message` into the node and, if given, the edge the visit took, and
  // returns the node's current value and bounds. Through an edge, the node's
  // mean swaps the edge's old contribution (n visits at its cached value) for
  // the new one (n + 1 visits at the child's value); without one (the node's
  // own evaluation, or a terminal or proven value), the value is a sample of
  // its own.
  static Message ApplyMessage(Node<PuctDag>& node, TriedEdge<PuctDag>* edge,
                              const Message& message) {
    ValueStats& stats = node.stats;
    const VisitValue& value = message.value;
    if (edge) {
      ValueStats& edge_stats = edge->stats;
      const float n = static_cast<float>(edge_stats.visits++);
      const float total = static_cast<float>(++stats.visits);
      stats.q += ((n + 1.0f) * value.q - n * edge_stats.q - stats.q) / total;
      stats.d += ((n + 1.0f) * value.d - n * edge_stats.d - stats.d) / total;
      stats.m += ((n + 1.0f) * value.m - n * edge_stats.m - stats.m) / total;
      // Against float drift.
      stats.q = std::clamp(stats.q, -1.0f, 1.0f);
      stats.d = std::clamp(stats.d, 0.0f, 1.0f);
      edge_stats.q = value.q;
      edge_stats.d = value.d;
      edge_stats.m = value.m;
    } else {
      stats.Add(value);
    }
    UpdateBounds(node, edge, message.bounds);
    return MakeCurrentMessage(node);
  }
};

}  // namespace lczero::lc5
