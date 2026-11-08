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

#include "chess/types.h"
#include "search/lc5/graph/node_eval.h"
#include "search/lc5/graph/prior.h"
#include "utils/compact_vector.h"

namespace lczero::lc5 {

// Maximum live visits, bounded by TriedEdge::in_flight.
inline constexpr int kMaxActiveVisitsLimit =
    std::numeric_limits<uint16_t>::max();

// A legal move and its prior.
struct LegalEdge {
  Move move;
  PackedPrior prior;
};
static_assert(sizeof(LegalEdge) == 4);

// Statistics and active reservations of an edge tried at least once.
template <typename P>
struct TriedEdge {
  typename P::EdgeStats stats{};
  uint16_t in_flight = 0;  // Virtual loss; bounded by kMaxActiveVisitsLimit.
  static_assert(kMaxActiveVisitsLimit <=
                std::numeric_limits<decltype(in_flight)>::max());
};

// Tie-break order for edges: descending prior, then ascending move encoding.
inline bool PreferOnTie(const LegalEdge& a, const LegalEdge& b) {
  if (a.prior != b.prior) return a.prior > b.prior;
  return a.move.raw_data() < b.move.raw_data();
}

template <typename P>
struct NodeState {
  typename P::NodeStats stats{};
  uint32_t version = 0;  // Bumped on change so eviction skips modified nodes.
  TerminalKind terminal = TerminalKind::kNonTerminal;
  CompactVector<LegalEdge> edges{};     // Sorted by PreferOnTie.
  CompactVector<TriedEdge<P>> tried{};  // Prefix of edges tried at least once.
};

}  // namespace lczero::lc5
