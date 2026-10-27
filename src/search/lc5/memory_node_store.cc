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

#include "search/lc5/memory_node_store.h"

namespace lczero::lc5 {

std::vector<std::optional<ExpansionPayload>> MemoryNodeStore::LoadBatch(
    std::span<const NodeKey> keys) {
  std::vector<std::optional<ExpansionPayload>> result;
  result.reserve(keys.size());
  for (const NodeKey key : keys) {
    auto& shard = shards_[ShardIndex(key)];
    std::lock_guard lock(shard.mutex);
    const auto it = shard.entries.find(key);
    result.push_back(it == shard.entries.end()
                         ? std::nullopt
                         : std::optional<ExpansionPayload>(it->second));
  }
  return result;
}

void MemoryNodeStore::StoreBatch(std::span<const StoredExpansion> entries) {
  for (const auto& entry : entries) {
    auto& shard = shards_[ShardIndex(entry.key)];
    std::lock_guard lock(shard.mutex);
    shard.entries.insert_or_assign(entry.key, entry.payload);
  }
}

void MemoryNodeStore::Clear() {
  for (auto& shard : shards_) {
    std::lock_guard lock(shard.mutex);
    shard.entries.clear();
  }
}

size_t MemoryNodeStore::Size() const {
  size_t size = 0;
  for (const auto& shard : shards_) {
    std::lock_guard lock(shard.mutex);
    size += shard.entries.size();
  }
  return size;
}

}  // namespace lczero::lc5
