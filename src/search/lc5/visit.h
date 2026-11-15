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

#include <compare>
#include <cstdint>
#include <optional>
#include <vector>

#include "chess/position.h"
#include "search/lc5/graph/graph.h"

namespace lczero::lc5 {

struct VisitId {
  uint64_t value = 0;
  uint32_t slot() const { return static_cast<uint32_t>(value); }
  uint32_t epoch() const { return static_cast<uint32_t>(value >> 32); }
  explicit operator bool() const { return value != 0; }
  auto operator<=>(const VisitId&) const = default;
};

struct PathStep {
  NodeKey key;
  std::optional<Move> selected_move{};  // The reserved outgoing edge, if any.
  // Whether backup updates this node (false for repetition or installed leaf).
  bool update_node = false;
};

// Where a visit starts selecting. `backup_prefix` lists the steps above `key`
// that its value is also backed up through.
struct VisitOrigin {
  NodeKey key;
  PositionHistory history;
  std::vector<PathStep> backup_prefix;
};

enum class VisitState : uint8_t {
  kFree,
  kReadyDescend,
  kReadyReselect,
  kWaitingMaterialization,
};

// Leaf value stored on a Visit once valued for backup.
struct LeafValue {
  VisitValue value{};
  // `value` is exact: a terminal other than a repetition, or a proven node.
  bool exact = false;
};

// Outcome supplied to BackupLeaf for a visit paused at kUnmaterializedLeaf.
struct MaterializedLeaf {
  VisitValue value{};
  TerminalKind terminal = TerminalKind::kNonTerminal;
  // False when InsertNode already counted `value` as the leaf's first visit
  // (the materialization owner); true for additional visits (terminal waiters).
  bool count_at_leaf = false;
};

struct Visit {
  static Visit FromOrigin(const VisitOrigin& origin, VisitId id = {}) {
    return Visit{.id = id,
                 .state = VisitState::kReadyDescend,
                 .origin = origin,
                 .history = origin.history,
                 .current_key = origin.key,
                 .path = origin.backup_prefix,
                 .leaf = std::nullopt};
  }

  VisitId id;
  VisitState state = VisitState::kFree;
  VisitOrigin origin;
  PositionHistory history;
  NodeKey current_key;
  std::vector<PathStep> path;
  std::optional<LeafValue> leaf{};
};

}  // namespace lczero::lc5
