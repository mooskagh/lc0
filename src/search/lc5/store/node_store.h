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

#include "search/lc5/graph/key.h"

namespace lczero::lc5 {

// A node's search state, serialized by the VisitEngine. The store does not
// look inside it.
struct StoredNode {
  NodeKey key;
  std::vector<uint8_t> record;
};

// The cold tier of the graph: node records that were evicted from the graph
// and can be restored into it with their statistics. The graph is always
// checked first, so a store may hold records that are out of date for nodes
// that are back in the graph. Implementations must be thread-safe.
class NodeStore {
 public:
  virtual ~NodeStore() = default;
  // One entry per key, nullopt if the store has no record for it.
  virtual std::vector<std::optional<std::vector<uint8_t>>> LoadBatch(
      std::span<const NodeKey> keys) = 0;
  // Stores or replaces the records; they are visible to LoadBatch once this
  // returns.
  virtual void StoreBatch(std::span<const StoredNode> nodes) = 0;
  virtual void EraseBatch(std::span<const NodeKey> keys) = 0;
  virtual void Clear() = 0;
};

}  // namespace lczero::lc5
