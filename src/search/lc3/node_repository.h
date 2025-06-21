#pragma once

#include <absl/container/flat_hash_map.h>

#include <cstdint>
#include <optional>
#include <span>

#include "chess/types.h"
#include "utils/exception.h"

namespace lczero {
namespace lc3 {

struct NodeKey {
  uint64_t hash;
  auto operator<=>(const NodeKey& other) const = default;
};

class NodeHandle {
 public:
  // All fields are out parameters. FetchEdgeData fills these spans with data
  // from the node's internal arrays. The moves, p, q, and n arrays are parallel
  // with the same index representing the same move across all arrays.
  // Spans should be pre-allocated with sufficient size.
  struct EdgeDataDestination {
    std::span<Move> moves = {};
    std::span<float> p = {};
    std::span<float> q = {};
    std::span<uint64_t> n = {};
  };

  struct EdgePatch {
    size_t edge_idx;
    size_t visits_to_undo;
    float agg_q;
  };

  enum class CertaintyState {
    kNonTerminal,  // The node is non-terminal.
    kTerminal,     // The node is terminal.
  };

  struct NodeAggregates {
    size_t n;
    double agg_v;
    float agg_d;
    float agg_m;
    CertaintyState state = CertaintyState::kNonTerminal;

    bool IsTerminal() const { return state != CertaintyState::kNonTerminal; }
  };

  struct MoveCounts {
    size_t total = 0;
    size_t with_visits = 0;
  };

  // Return true if the handle is valid (i.e., it points to a node).
  operator bool() const { return data_ != nullptr; }
  // Returns true if the node has just been created.
  bool IsNew() const { return is_new_; }
  // Releases the lock and invalidates the handle.
  void Release();

  // Node aggregates.
  void ApplyNodeUpdate(NodeAggregates);
  NodeAggregates GetNodeAggregates() const;

  // Edges.
  MoveCounts FetchMoveCounts() const;
  void InitializeEdges(std::span<const Move> moves, std::span<const float> p);
  void FetchEdges(EdgeDataDestination request) const;
  void AddEdgeVisits(std::span<const uint64_t>) const;
  void UpdateEdges(std::span<const EdgePatch>);
  friend class NodeRepository;

 private:
  struct NodeData;
  NodeHandle() = default;
  NodeHandle(NodeData* data, std::unique_lock<std::mutex> lock, bool is_new)
      : data_(data), lock_(std::move(lock)), is_new_(is_new) {}

  NodeData* data_ = nullptr;
  std::unique_lock<std::mutex> lock_;
  bool is_new_ = false;
};

class NodeRepository {
 public:
  NodeRepository();
  ~NodeRepository();
  NodeHandle GetNodeForUpdate(const NodeKey& key, bool create_if_missing);

 private:
  // Disable copy and move semantics for NodeRepository.
  NodeRepository(const NodeRepository&) = delete;
  NodeRepository& operator=(const NodeRepository&) = delete;
  NodeRepository(NodeRepository&&) = delete;
  NodeRepository& operator=(NodeRepository&&) = delete;

  struct StorageImpl;
  struct Shard;
  std::unique_ptr<StorageImpl> storage_impl_;
  friend class NodeHandle;
};

}  // namespace lc3
}  // namespace lczero