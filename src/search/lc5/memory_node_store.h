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

#include <array>
#include <mutex>

#include "search/lc5/node_store.h"

namespace lczero::lc5 {

class MemoryNodeStore final : public NodeStore {
 public:
  std::vector<std::optional<ExpansionPayload>> LoadBatch(
      std::span<const NodeKey> keys) override;
  void StoreBatch(std::span<const StoredExpansion> entries) override;
  void Clear() override;
  size_t Size() const;

 private:
  static constexpr size_t kShardCount = 64;
  struct Shard {
    mutable std::mutex mutex;
    absl::flat_hash_map<NodeKey, ExpansionPayload> entries;
  };
  static size_t ShardIndex(NodeKey key) { return key.hash >> 58; }
  std::array<Shard, kShardCount> shards_;
};

}  // namespace lczero::lc5
