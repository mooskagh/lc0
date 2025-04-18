#pragma once

#include <absl/container/flat_hash_map.h>

#include <cstdint>
#include <optional>
#include <span>

#include "chess/types.h"
#include "utils/exception.h"

namespace lczero {

class NodeStorage;
class UpdateLock;
struct NodeHash {
  uint64_t hash;
};

namespace internal {
struct NodeData {};
}  // namespace internal

struct NodeCreate {};

class NodeUpdate {
 public:
  ~NodeUpdate();
  bool HasVisits() const { NotImplemented(); }
  bool IsTerminal() const { NotImplemented(); }
  uint64_t IncrementN(int64_t n) { NotImplemented(); }
  size_t FetchNumMoves() const { NotImplemented(); }
  size_t FetchNumMovesWithVisits() const { NotImplemented(); }

  struct EdgeDataRequest {
    std::span<Move> moves = {};
    std::span<float> p = {};
    std::span<float> q = {};
    std::span<uint64_t> n = {};
  };

  void FetchEdgeData(EdgeDataRequest request) const { NotImplemented(); }
  void UpdateEdgeN(std::span<const uint64_t> n) const { NotImplemented(); }

 private:
  NodeUpdate(UpdateLock* lock, internal::NodeData* data);
  UpdateLock* const lock_;
  internal::NodeData* const data_;
  friend class UpdateLock;
};

// While this lock is held, no hashmap rehashing will occur.
class UpdateLock {
 public:
  // Returns nullopt if the node is not found.
  std::optional<NodeUpdate> Fetch(NodeHash node);
  ~UpdateLock();

 private:
  UpdateLock(NodeStorage* storage) : storage_(storage) {}

  NodeStorage* const storage_;
#ifndef NDEBUG
  uint32_t ref_count_ = 0;  // For debugging only.
#endif
  friend class NodeUpdate;
  friend class NodeStorage;
  friend class CreationLock;
};

class CreationLock {
 public:
  static CreationLock FromUpdateLock(UpdateLock&& lock);

 private:
  CreationLock(NodeStorage* storage) : storage_(storage) {}
  NodeStorage* const storage_;
#ifndef NDEBUG
  uint32_t ref_count_ = 0;  // For debugging only.
#endif
};

class NodeStorage {
 public:
  UpdateLock GetUpdateLock();

 private:
  absl::flat_hash_map<uint64_t, internal::NodeData> nodes_;
  friend class UpdateLock;
  friend class NodeUpdate;
};

}  // namespace lczero