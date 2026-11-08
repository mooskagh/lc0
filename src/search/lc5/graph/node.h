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

#include <cstdint>
#include <limits>
#include <vector>

#include "chess/types.h"
#include "search/lc5/graph/node_eval.h"
#include "search/lc5/graph/prior.h"

namespace lczero::lc5 {

using MaterializationId = uint64_t;

// Each live visit holds at most one unit of an edge's 16-bit in_flight, which
// caps the number of live visits.
inline constexpr int kMaxActiveVisitsLimit =
    std::numeric_limits<uint16_t>::max();

enum class NodeLifecycle : uint8_t { kMaterializing, kMaterialized };

// A legal move and its prior; every legal move of a materialized node has one.
struct LegalEdge {
  Move move;
  PackedPrior prior;
};
static_assert(sizeof(LegalEdge) == 4);

// The statistics of an edge that has been reserved at least once. The
// machinery owns the reservations; `stats` belongs to the policy P.
template <typename P>
struct TriedEdge {
  uint16_t in_flight = 0;  // Virtual loss; bounded by kMaxActiveVisitsLimit.
  typename P::EdgeStats stats{};
  static_assert(kMaxActiveVisitsLimit <=
                std::numeric_limits<decltype(in_flight)>::max());
};

// Deterministic tie-break between equally ranked edges: higher prior first,
// then the lower move encoding. A node's edges are sorted in this order, and
// selection relies on it.
inline bool PreferOnTie(const LegalEdge& a, const LegalEdge& b) {
  if (a.prior != b.prior) return a.prior > b.prior;
  return a.move.raw_data() < b.move.raw_data();
}

template <typename P>
struct NodeState {
  NodeLifecycle lifecycle = NodeLifecycle::kMaterializing;
  // Initialized from a counter on creation and bumped by every change to the
  // node, so an eviction can tell whether the node changed after its record
  // was taken. Wraps around.
  uint32_t version = 0;
  MaterializationId materialization_id = 0;  // Zero once materialized.
  TerminalKind terminal = TerminalKind::kNonTerminal;
  typename P::NodeStats stats{};
  std::vector<LegalEdge> edges{};
  // tried[i] belongs to edges[i]. Edges from tried.size() on have never been
  // reserved; selection appends the first of them when it picks it.
  std::vector<TriedEdge<P>> tried{};
};

}  // namespace lczero::lc5
