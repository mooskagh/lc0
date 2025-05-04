#include "search/lc3/storage.h"

namespace lczero {
namespace lc3 {

UpdateLock NodeStorage::GetUpdateLock() { return UpdateLock(this); }

NodeMutation::NodeMutation(UpdateLock* lock, internal::NodeData* data)
    : lock_(lock), data_(data) {
  ++lock_->ref_count_;
}
NodeMutation::~NodeMutation() { --lock_->ref_count_; }

std::optional<NodeMutation> UpdateLock::Fetch(NodeHash node) {
  auto iter = storage_->nodes_.find(node.hash);
  if (iter == storage_->nodes_.end()) return std::nullopt;
  return std::optional<NodeMutation>(std::in_place, this, &iter->second);
}

UpdateLock::~UpdateLock() { assert(ref_count_ == 0); }

CreationLock CreationLock::FromUpdateLock(UpdateLock&& lock) {
  assert(lock.ref_count_ == 0);
  return CreationLock(lock.storage_);
}

bool CreationLock::Create(NodeHash node_hash) {
  return storage_->nodes_.try_emplace(node_hash.hash).second;
}

}  // namespace lc3
}  // namespace lczero