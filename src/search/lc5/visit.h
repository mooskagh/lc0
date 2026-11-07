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
  uint64_t generation = 0;
  std::optional<Move> selected_move{};  // The reserved outgoing edge, if any.
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
  kReadySelect,
  kWaitingMaterialization,
  kReadyBackup,
};

struct Visit {
  VisitId id;
  VisitState state = VisitState::kFree;
  VisitOrigin origin;
  PositionHistory history;
  NodeKey current_key;
  std::vector<PathStep> path;
  SearchValue result;
  // `result` is exact: a terminal other than a repetition, or a proven node.
  bool result_exact = false;
  // The leaf node already holds `result` (added at install).
  bool leaf_counted = false;
  // For a two-fold repetition leaf whose draw is contained: the path index of
  // the repeated position's first occurrence, the last step the draw reaches.
  std::optional<size_t> two_fold_origin{};
};

}  // namespace lczero::lc5
