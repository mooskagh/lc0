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

#include <string_view>

#include "search/lc5/graph/graph.h"
#include "search/lc5/policies/policy_engine.h"
#include "search/lc5/policies/puct.h"
#include "search/lc5/policies/registry.h"

namespace lczero::lc5 {
namespace {

// PUCT over running averages: an edge averages the values of the visits that
// went through it, and a node those of all visits to it.
struct PuctAveraging : PuctPolicy<PuctAveraging> {
  static constexpr std::string_view kName = "puct";

  // Folds `backup` into the node and, if given, the edge the visit took, and
  // returns what the parent receives: the value and the node's own bounds.
  static BackupValue BackupThroughNode(Node<PuctAveraging>& node,
                                       TriedEdge<PuctAveraging>* edge,
                                       const BackupValue& backup) {
    node.stats.Add(backup.value);
    if (edge) edge->stats.Add(backup.value);
    UpdateBounds(node, edge, backup.bounds);
    return {backup.value.Parent(), node.stats.bounds.Parent()};
  }
};
static_assert(sizeof(TriedEdge<PuctAveraging>) <= 24);
static_assert(sizeof(Node<PuctAveraging>) == 64);

}  // namespace

// The default policy.
REGISTER_LC5_POLICY(PuctAveraging, 100)

}  // namespace lczero::lc5
