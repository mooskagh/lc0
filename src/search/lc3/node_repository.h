#pragma once

#include <absl/container/flat_hash_map.h>

#include <cstdint>
#include <optional>
#include <span>

#include "chess/types.h"
#include "utils/exception.h"

namespace lczero {
namespace lc3 {

class NodeRepository;
class AccessLock;
struct NodeHash {
  uint64_t hash;
  bool operator==(const NodeHash& other) const = default;
};

struct StorageNodeData {
  size_t n;
  double agg_v;
  float agg_d;
  float agg_m;
};

namespace internal {
struct EdgeData {
  float q = 0.0f;
  uint64_t n = 0;
};
struct NodeData {
  bool is_terminal = false;
  std::vector<Move> moves;
  std::vector<float> p;
  StorageNodeData value;
  std::vector<EdgeData> edges;  // Edges to children, indexed by move
};
}  // namespace internal

// NOtes:
// - Node N doesn't include n_in_flight
// - Edge N does ingluce n_in_flight
// - Node Q is updated
// - Edge Q is copied from node Q

// TODO proper name and location
struct StorageEdgePatch {
  size_t edge_idx;
  size_t visits_to_undo;
  float agg_q;
};

// All fields are out parameters. FetchEdgeData fills these spans with data
// from the node's internal arrays. The moves, p, q, and n arrays are parallel
// with the same index representing the same move across all arrays.
// Spans should be pre-allocated with sufficient size.
struct EdgeDataRequest {
  std::span<Move> moves = {};
  std::span<float> p = {};
  std::span<float> q = {};
  std::span<uint64_t> n = {};
};

class NodeMutation {
 public:
  ~NodeMutation();
  // Should be private but std::optional needs it.
  NodeMutation(AccessLock* lock, internal::NodeData* data);
  NodeMutation(const NodeMutation&) = delete;
  NodeMutation& operator=(const NodeMutation&) = delete;
  // If/when we implement move semantics, make the constructor above private.
  NodeMutation(NodeMutation&&) = delete;
  NodeMutation& operator=(NodeMutation&&) = delete;

  bool HasVisits() const { return data_->value.n > 0; }
  bool IsTerminal() const { return data_->is_terminal; }
  uint64_t GetN() const { return data_->value.n; }
  size_t FetchNumMoves() const { return data_->moves.size(); }
  size_t FetchNumMovesWithVisits() const { return data_->edges.size(); }

  void FetchEdgeData(EdgeDataRequest request) const;
  // TODO potentially combine IncrementEdgeN and UpdateEdgeData
  void IncrementEdgeN(std::span<const uint64_t>) const;
  void UpdateEdgeData(std::span<const StorageEdgePatch>);
  void SetEdgeData(std::span<const Move> moves, std::span<const float> p);
  void SetIsTerminal() { NotImplemented(); }
  StorageNodeData AccumulateNodeData(StorageNodeData);

 private:
  AccessLock* const lock_;
  internal::NodeData* const data_;
  friend class AccessLock;
  friend class std::optional<NodeMutation>;
};

class NodeView {
 public:
  ~NodeView();
  // Should be private but std::optional needs it.
  NodeView(AccessLock* lock, internal::NodeData* data);

  uint64_t GetN() const { return data_->value.n; }
  size_t FetchNumMoves() const { return data_->moves.size(); }
  size_t FetchNumMovesWithVisits() const { return data_->edges.size(); }

  void FetchEdgeData(EdgeDataRequest request) const;

 private:
  AccessLock* const lock_;
  internal::NodeData* const data_;
};

// While this lock is held, no hashmap rehashing will occur.
class AccessLock {
 public:
  // Returns nullopt if the node is not found.
  std::optional<NodeMutation> FetchMutable(NodeHash node);
  std::optional<NodeView> FetchReadOnly(NodeHash node);
  ~AccessLock();

 private:
  AccessLock(NodeRepository* node_repository)
      : node_repository_(node_repository) {}

  NodeRepository* const node_repository_;
#ifndef NDEBUG
  uint32_t ref_count_ = 0;  // For debugging only.
#endif
  friend class NodeMutation;
  friend class NodeView;
  friend class NodeRepository;
  friend class CreationLock;
};

class CreationLock {
 public:
  static CreationLock FromAccessLock(AccessLock&& lock);
  // Creates "empty" node.
  bool Create(NodeHash node_hash);

 private:
  CreationLock(NodeRepository* node_repository)
      : node_repository_(node_repository) {}
  NodeRepository* const node_repository_;
#ifndef NDEBUG
  uint32_t ref_count_ = 0;  // For debugging only.
#endif
};

class NodeRepository {
 public:
  AccessLock GetAccessLock();

 private:
  absl::flat_hash_map<uint64_t, internal::NodeData> nodes_;
  friend class AccessLock;
  friend class NodeMutation;
  friend class CreationLock;
};

}  // namespace lc3
}  // namespace lczero