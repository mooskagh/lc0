#pragma once

#include <array>
#include <mutex>

#include <absl/container/flat_hash_map.h>

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
