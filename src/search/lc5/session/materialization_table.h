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

#include <absl/container/flat_hash_map.h>
#include <absl/synchronization/mutex.h>

#include <cassert>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

#include "search/lc5/graph/key.h"
#include "search/lc5/session/messages.h"

namespace lczero::lc5 {

// A node being materialized and the visits waiting for it.
struct MaterializationWaiters {
  NodeKey key;        // The node being materialized.
  VisitWakeup owner;  // The visit that started the materialization; does I/O.
  std::vector<VisitWakeup> waiters;  // Later visits that reached the node.
};

// The open materializations, at most one per node. Any visit worker opens or
// joins one; only its owner's worker closes it.
//
// The callbacks run under the table's mutex, and check or install the node in
// the graph: lock order is materializations -> one graph shard. So a visit
// that finds no open materialization for a node also finds the node installed.
class MaterializationTable {
 public:
  struct Joined {
    MaterializationId id;
    bool owner;          // Opened the materialization.
    size_t waiters = 0;  // Waiters on the materialization, if joined as one.
  };

  // Joins the open materialization for `key`, or opens a new one if
  // `can_open()` returns true. Returns nullopt if no materialization is open
  // and `can_open()` declines (the node was materialized since the caller last
  // checked).
  template <typename CanOpen>
  std::optional<Joined> Join(NodeKey key, VisitWakeup wakeup,
                             CanOpen&& can_open) {
    absl::MutexLock lock(mutex_);
    if (const auto existing = id_by_key_.find(key);
        existing != id_by_key_.end()) {
      auto& waiters = materializations_.at(existing->second).waiters;
      waiters.push_back(wakeup);
      return Joined{
          .id = existing->second, .owner = false, .waiters = waiters.size()};
    }
    if (!can_open()) return std::nullopt;
    const MaterializationId materialization_id = next_id_++;
    materializations_.emplace(materialization_id,
                              MaterializationWaiters{key, wakeup, {}});
    id_by_key_.emplace(key, materialization_id);
    return Joined{.id = materialization_id, .owner = true};
  }

  // `close(key)` gets open `materialization_id`'s node `key` its content and
  // returns whether it did. If so, closes the materialization and returns its
  // waiters.
  template <typename Op>
  std::optional<MaterializationWaiters> Close(
      MaterializationId materialization_id, Op&& close) {
    absl::MutexLock lock(mutex_);
    const auto it = materializations_.find(materialization_id);
    assert(it != materializations_.end());
    if (!close(it->second.key)) return std::nullopt;
    MaterializationWaiters entry = std::move(it->second);
    materializations_.erase(it);
    id_by_key_.erase(entry.key);
    return entry;
  }

  bool empty() const {
    absl::MutexLock lock(mutex_);
    assert(materializations_.size() == id_by_key_.size());
    return materializations_.empty();
  }

 private:
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<NodeKey, MaterializationId> id_by_key_
      ABSL_GUARDED_BY(mutex_);
  std::unordered_map<MaterializationId, MaterializationWaiters>
      materializations_ ABSL_GUARDED_BY(mutex_);
  MaterializationId next_id_ ABSL_GUARDED_BY(mutex_) = 1;
};

}  // namespace lczero::lc5
