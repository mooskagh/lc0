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
#include <optional>
#include <span>
#include <vector>

#include "chess/types.h"
#include "search/lc5/key.h"
#include "search/lc5/value.h"

namespace lczero::lc5 {

enum class TerminalKind : uint8_t {
  kNonTerminal,
  kCheckmate,
  kStalemate,
  kRule50,
  kRepetition,
  kInsufficientMaterial,
};

// Value of a terminal position for its side to move.
inline SearchValue TerminalValue(TerminalKind terminal) {
  return terminal == TerminalKind::kCheckmate ? SearchValue{-1.0f, 0.0f, 0.0f}
                                              : SearchValue{0.0f, 1.0f, 0.0f};
}

// Immutable result of expanding a node: either a terminal kind, or legal moves
// with priors and the network's leaf value.
struct ExpansionPayload {
  TerminalKind terminal = TerminalKind::kNonTerminal;
  SearchValue leaf{};
  std::vector<Move> moves;
  std::vector<float> priors;
};

struct StoredExpansion {
  NodeKey key;
  ExpansionPayload payload;
};

// Optional batched backing store for expansion payloads.
class NodeStore {
 public:
  virtual ~NodeStore() = default;
  virtual std::vector<std::optional<ExpansionPayload>> LoadBatch(
      std::span<const NodeKey> keys) = 0;
  virtual void StoreBatch(std::span<const StoredExpansion> entries) = 0;
  virtual void Clear() = 0;
};

}  // namespace lczero::lc5
