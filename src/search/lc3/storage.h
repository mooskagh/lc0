#pragma once

#include <absl/container/flat_hash_map.h>

#include <cstdint>
#include <optional>
#include <span>

#include "chess/types.h"
#include "utils/exception.h"

namespace lczero {
namespace lc3 {

class NodeStorage;
class UpdateLock;
struct NodeHash {
  uint64_t hash;
  bool operator==(const NodeHash& other) const = default;
};

namespace internal {
struct NodeData {
  bool is_terminal = false;
};
}  // namespace internal

// NOtes:
// - Node N doesn't include n_in_flight
// - Edge N does ingluce n_in_flight
// - Node Q is updated
// - Edge Q is copied from node Q

// TODO proper name and location
struct EdgeUpdate {
  size_t edge_idx;
  int num_visits_to_decrement;
  double q;
};

class NodeMutation {
 public:
  ~NodeMutation();
  bool HasVisits() const { NotImplemented(); }
  bool IsTerminal() const { return data_->is_terminal; }
  uint64_t GetN() const { NotImplemented(); }
  // uint64_t IncrementN(int64_t) { NotImplemented(); }
  size_t FetchNumMoves() const { NotImplemented(); }
  size_t FetchNumMovesWithVisits() const { NotImplemented(); }

  struct EdgeDataRequest {
    std::span<Move> moves = {};
    std::span<float> p = {};
    std::span<float> q = {};
    std::span<uint64_t> n = {};
  };

  void FetchEdgeData(EdgeDataRequest) const { NotImplemented(); }
  void IncrementEdgeN(std::span<const uint64_t>) const { NotImplemented(); }
  void SetEdgeData(std::span<const Move>, std::span<const float> p) {
    NotImplemented();
  }
  void UpdateNodeData(int num_visits, float q, float d, float m) {
    NotImplemented();
  }
  void UpdateEdgeData(std::span<const EdgeUpdate>) { NotImplemented(); }

 private:
  NodeMutation(UpdateLock* lock, internal::NodeData* data);
  UpdateLock* const lock_;
  internal::NodeData* const data_;
  friend class UpdateLock;
};

// While this lock is held, no hashmap rehashing will occur.
class UpdateLock {
 public:
  // Returns nullopt if the node is not found.
  std::optional<NodeMutation> Fetch(NodeHash node);
  ~UpdateLock();

 private:
  UpdateLock(NodeStorage* storage) : storage_(storage) {}

  NodeStorage* const storage_;
#ifndef NDEBUG
  uint32_t ref_count_ = 0;  // For debugging only.
#endif
  friend class NodeMutation;
  friend class NodeStorage;
  friend class CreationLock;
};

class CreationLock {
 public:
  static CreationLock FromUpdateLock(UpdateLock&& lock);
  // Creates "empty" node.
  bool Create(NodeHash node_hash);

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
  friend class NodeMutation;
  friend class CreationLock;
};

}  // namespace lc3
}  // namespace lczero