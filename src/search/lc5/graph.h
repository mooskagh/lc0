#pragma once

#include <absl/container/flat_hash_map.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

#include "search/lc5/key.h"
#include "search/lc5/node_store.h"
#include "search/lc5/settings.h"
#include "search/lc5/value.h"

namespace lczero::lc5 {

// Sharded map from NodeKey (typically a position history hash, but can be any
// unique key) to node search state and outgoing edges. Edges don't store child
// pointers; child keys are computed by the caller.
using MaterializationTicketId = uint64_t;

enum class NodeLifecycle : uint8_t { kMaterializing, kExpanded };

struct EdgeState {
  Move move;
  float prior = 0.0f;
  uint64_t visits = 0;
  uint32_t in_flight = 0;  // Virtual loss.
  double q_sum = 0.0;
  double d_sum = 0.0;
  double m_sum = 0.0;
  float Q() const { return visits ? static_cast<float>(q_sum / visits) : 0.0f; }
};

struct NodeState {
  uint64_t generation =
      0;  // Incremented on recreation to reject stale visit updates.
  NodeLifecycle lifecycle = NodeLifecycle::kMaterializing;
  MaterializationTicketId ticket = 0;
  TerminalKind terminal = TerminalKind::kNonTerminal;
  ValueStats value;
  std::vector<EdgeState> edges;
  uint64_t last_access_epoch = 0;
};

using NodeSnapshot = NodeState;

// Lightweight snapshot without edges or value stats, used during selection.
struct NodeMetadataSnapshot {
  uint64_t generation;
  NodeLifecycle lifecycle;
  TerminalKind terminal;
};

struct FindOrCreateResult {
  uint64_t generation;
  bool created;
  NodeLifecycle lifecycle;
  MaterializationTicketId ticket;
};

enum class SelectStatus {
  kSelected,
  kMissing,
  kStale,
  kMaterializing,
  kTerminal
};

struct SelectResult {
  SelectStatus status = SelectStatus::kMissing;
  Move move{};
  uint64_t generation = 0;
};

enum class UpdateResult {
  kApplied,
  kMissing,
  kStale,
  kEdgeMissing,
  kUnderflow
};

struct BackupResult {
  UpdateResult node;
  UpdateResult edge;
};

class GameGraph {
 public:
  static constexpr size_t kShardCount = 1024;

  // Finds existing node or creates one in kMaterializing state with ticket.
  FindOrCreateResult FindOrCreateMaterializing(NodeKey key,
                                               MaterializationTicketId ticket);

  // Transitions node to kExpanded and installs edges. Rejected if ticket
  // doesn't match.
  uint64_t InstallPayload(NodeKey key, MaterializationTicketId ticket,
                          const ExpansionPayload& payload,
                          bool* accepted = nullptr);

  // Selects best edge via PUCT and increments in_flight (virtual loss).
  SelectResult SelectAndReserve(NodeKey key, uint64_t expected_generation,
                                const Settings::Resolved& settings);

  // Commits node value stats and optionally completes an edge.
  BackupResult BackupNode(NodeKey key, uint64_t expected_generation,
                          SearchValue value, std::optional<Move> move);

  UpdateResult UpdateNodeValue(NodeKey key, uint64_t expected_generation,
                               SearchValue value);

  UpdateResult CompleteEdge(NodeKey key, uint64_t expected_generation,
                            Move move, SearchValue value);

  // Decrements in_flight without committing visits or values.
  UpdateResult CancelEdge(NodeKey key, uint64_t expected_generation, Move move);

  std::optional<NodeMetadataSnapshot> SnapshotNodeMetadata(NodeKey key) const;
  std::optional<NodeSnapshot> SnapshotNode(NodeKey key) const;

  bool Erase(NodeKey key);
  void Clear();
  size_t Size() const;
  std::vector<std::pair<NodeKey, NodeSnapshot>> SnapshotAllForTesting() const;

 private:
  struct GraphShard {
    mutable std::mutex mutex;
    absl::flat_hash_map<NodeKey, NodeState> nodes;
  };

  static size_t ShardIndex(NodeKey key) {
    return (key.hash * 11400714819323198485ul) >> 54;
  }

  uint64_t NextGeneration();

  std::array<GraphShard, kShardCount> shards_;
  std::atomic<size_t> size_{0};
  std::atomic<uint64_t> next_generation_{1};
  std::atomic<uint64_t> access_epoch_{1};
};

}  // namespace lczero::lc5
