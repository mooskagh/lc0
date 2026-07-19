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
