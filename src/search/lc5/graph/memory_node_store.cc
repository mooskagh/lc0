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

#include "search/lc5/graph/memory_node_store.h"

namespace lczero::lc5 {

std::vector<std::optional<std::vector<uint8_t>>> MemoryNodeStore::LoadBatch(
    std::span<const NodeKey> keys) {
  std::vector<std::optional<std::vector<uint8_t>>> result;
  result.reserve(keys.size());
  for (const NodeKey key : keys) {
    auto& shard = shards_[ShardIndex(key)];
    std::lock_guard lock(shard.mutex);
    const auto it = shard.records.find(key);
    if (it == shard.records.end()) {
      result.emplace_back(std::nullopt);
    } else {
      result.emplace_back(it->second);
    }
  }
  return result;
}

void MemoryNodeStore::StoreBatch(std::span<const StoredNode> nodes) {
  for (const auto& node : nodes) {
    auto& shard = shards_[ShardIndex(node.key)];
    std::lock_guard lock(shard.mutex);
    shard.records.insert_or_assign(node.key, node.record);
  }
}

void MemoryNodeStore::EraseBatch(std::span<const NodeKey> keys) {
  for (const NodeKey key : keys) {
    auto& shard = shards_[ShardIndex(key)];
    std::lock_guard lock(shard.mutex);
    shard.records.erase(key);
  }
}

void MemoryNodeStore::Clear() {
  for (auto& shard : shards_) {
    std::lock_guard lock(shard.mutex);
    shard.records.clear();
  }
}

size_t MemoryNodeStore::Size() const {
  size_t size = 0;
  for (const auto& shard : shards_) {
    std::lock_guard lock(shard.mutex);
    size += shard.records.size();
  }
  return size;
}

}  // namespace lczero::lc5
