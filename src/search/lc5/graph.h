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

// clang-format off
// ============================================================================
// GameGraph Overview
// ============================================================================
//
// GameGraph is the concurrent, game-scoped search graph for lc5. Unlike classic
// tree-based MCTS implementations where nodes hold explicit pointers or indices
// to child nodes, GameGraph is designed as a pointerless Directed Acyclic Graph
// (DAG) stored in a sharded associative hash map keyed by NodeKey.
//
// Key Architectural Principles:
//
// 1. Pointerless DAG Representation:
//    - Nodes in the graph represent game states keyed by recent position history
//      hashes (NodeKey).
//    - Outgoing edges (EdgeState) record moves, policy priors, completed visit
//      counts, value sums (Q, D, M), and in-flight reservations.
//    - Edges do NOT contain child pointers. Instead, child node keys are computed
//      dynamically on the fly from the position history and the selected move:
//        child_key = MakeNodeKey(history, history_key_length);
//    - This allows chess transpositions (positions reached via different move
//      orders) to automatically converge on the same node without requiring
//      graph cycle tracking, DAG link pointer fixups, or tree duplication.
//    - Visits own their traversal path history (std::vector<PathStep>). Visit
//      state is released upon completion or cancellation, while graph state
//      persists across searches until explicitly erased or cleared.
//
// 2. High-Concurrency Sharding:
//    - Nodes are partitioned across kShardCount (1024) independent shards.
//    - Each shard possesses its own std::mutex and absl::flat_hash_map.
//    - Shard selection uses 64-bit multiplicative Fibonacci hashing to achieve
//      uniform key distribution and eliminate false-sharing / lock contention.
//    - Search workers hold locks only for the duration of a single node lookup,
//      selection, or update, allowing massive parallelism across search threads.
//
// 3. Two-Phase Node Lifecycle (Materialization):
//    - When a visit reaches an unexpanded position, it registers the node in
//      the kMaterializing state with a MaterializationTicketId via
//      FindOrCreateMaterializing().
//    - The first visit (owner) initiates asynchronous expansion (neural network
//      inference, Syzygy tablebase lookup, or NodeStore disk retrieval).
//    - Subsequent visits encountering the same key suspend and register as
//      waiters on the ticket in SearchRun.
//    - When evaluation completes, InstallPayload() atomically transitions the
//      node to kExpanded, publishing its legal moves, policy priors, and
//      terminal status. Waiters are then awakened to resume selection.
//
// 4. Stale-Visit Protection via Generations:
//    - Every node carries a 64-bit incarnation ID (generation), incremented
//      each time a node is created.
//    - In-flight visits record the generation of each node they visit.
//    - Mutating methods (SelectAndReserve, BackupNode, CompleteEdge, CancelEdge)
//      verify that the expected_generation matches the node's current generation.
//    - If a node was erased and recreated while a visit was in flight, the visit
//      detects generation mismatch (kStale) and safely aborts or discards its
//      update without corrupting the new node.
//
// 5. Virtual Loss via In-Flight Reservations:
//    - Parallel search avoids redundant work by discouraging multiple workers
//      from simultaneously exploring the same path.
//    - SelectAndReserve() atomically selects the best edge via PUCT and
//      increments EdgeState::in_flight (virtual loss).
//    - Completed visits convert in_flight reservations into completed visits
//      and value accumulations via BackupNode() or CompleteEdge().
//    - Aborted or cancelled visits release their reservations via CancelEdge()
//      without altering visit counts or value statistics.
//
// 6. Memory Footprint:
//    - On 64-bit Linux platforms, NodeState occupies 96 bytes plus 48 bytes per
//      allocated edge vector element.
//    - Additional overhead includes the 8-byte key, hash table bucket metadata,
//      and heap allocator overhead.
//
// ============================================================================
// GameGraph API Promises and Guarantees
// ============================================================================
//
// What GameGraph GUARANTEES:
// - Value Semantics & Reference Safety:
//     GameGraph never returns raw pointers or references to internal NodeState
//     or EdgeState. All data returned from public methods (snapshots, result
//     structs, move values) is copied by value while holding the shard mutex.
//     Once a method returns, internal node data may be mutated, reallocated,
//     or erased concurrently without invalidating the caller's snapshot.
// - Bounded Single-Shard Locking (Deadlock-Free):
//     All point operations (lookup, selection, installation, backup, erasure)
//     acquire at most one shard mutex for a bounded, O(1) duration.
//     GameGraph never acquires a second shard lock during point operations and
//     NEVER calls external subsystems (no neural backend, no NodeStore, no I/O,
//     no user callbacks) while holding a lock.
// - Strict Invariant & Underflow Protection:
//     In-flight reservation counters (EdgeState::in_flight) are never allowed
//     to underflow. If CancelEdge or CompleteEdge is invoked when in_flight == 0,
//     GameGraph leaves the counter at 0 and returns UpdateResult::kUnderflow.
// - Generational Safety:
//     Monotonically increasing generation IDs guarantee that search visits
//     cannot commit statistics to a recreated node from an earlier search run
//     or eviction generation.
// - Idempotent One-Shot Expansion:
//     InstallPayload() verifies node.ticket == ticket. Once expanded, subsequent
//     or duplicate completions are rejected without corrupting live edges.
//
// What GameGraph DOES NOT Do:
// - It does NOT track visit paths, repetition history, or tree structure. Visits
//   track their own history and compute child keys independently.
// - It does NOT manage worker threads, queues, or suspension for materialization.
//   The ticket registry and worker continuations are managed by SearchRun.
// - It does NOT evaluate chess positions or perform legal move generation.
// - It does NOT handle tiered disk/database persistence (see NodeStore below).
//
// ============================================================================
// Relationship between GameGraph and NodeStore
// ============================================================================
//
// In lc5, GameGraph and NodeStore represent two distinct layers in the search
// architecture with clear separation of responsibilities:
//
//   +-----------------------------------------------------------------------+
//   |                              SearchRun                                |
//   |              (Coordinates Visits, Workers & Schedulers)               |
//   +-----------------------------------+-----------------------------------+
//                                       |
//                 +---------------------+---------------------+
//                 |                                           |
//                 v                                           v
//      +----------------------+                   +----------------------+
//      |      GameGraph       |                   |      NodeStore       |
//      +----------------------+                   +----------------------+
//      | - Mutable MCTS State |                   | - Immutable Payloads |
//      | - Visits & Q/D/M sums|                   | - Legal moves &      |
//      | - In-flight reserves |                   |   policy priors      |
//      | - Node generations   |                   | - Leaf evaluations   |
//      | - RAM-only (sharded) |                   | - Reusable / Cache   |
//      | - Transient (evict)  |                   | - RAM / Disk / DB    |
//      +----------------------+                   +----------------------+
//
// 1. Separation of State:
//    - GameGraph holds dynamic, search-specific statistics: visit counts,
//      Q/D/M values, virtual loss reservations, and generation counters.
//      These change continuously as visits flow through the search tree.
//    - NodeStore holds static, reusable expansion payloads (ExpansionPayload):
//      the position's legal moves, policy priors from the neural network,
//      terminal kind, and evaluation estimates. This data is immutable once
//      evaluated.
//
// 2. Zero Direct Coupling:
//    - GameGraph and NodeStore never reference or call each other.
//    - Neither holds locks belonging to the other.
//
// 3. Coordination via SearchRun:
//    - Cache-First Expansion: When a visit reaches an unexpanded node in
//      GameGraph (FindOrCreateMaterializing), SearchRun first queries
//      NodeStore::LoadBatch() to see if an ExpansionPayload was already
//      cached (e.g. from an earlier search or persistent disk cache).
//    - Hit: If NodeStore has the payload, SearchRun installs it directly into
//      GameGraph via InstallPayload(), skipping neural network evaluation.
//    - Miss: If NodeStore does not have the payload, SearchRun evaluates the
//      position via Backend::ComputeBlocking(), installs the payload into
//      GameGraph, and asynchronously persists it into NodeStore via StoreBatch().
//
// 4. Eviction and Rehydration:
//    - If memory pressure forces GameGraph to evict nodes via Erase(), only
//      the transient MCTS statistics (visits and reservations) are discarded.
//    - The expensive neural network computation remains stored in NodeStore.
//    - If the search explores that position again, GameGraph creates a fresh
//      node generation, and SearchRun rehydrates it from NodeStore without
//      re-running neural network inference.
//
// ============================================================================
// Sample Usage
// ============================================================================
//
// Below is an illustrative example demonstrating how a search pipeline interacts
// with GameGraph during node expansion, selection, backup, and cancellation:
//
//   GameGraph graph;
//   Settings::Resolved settings = ...;
//   PositionHistory history;  // Root chess position history
//   NodeKey root_key = MakeNodeKey(history, settings.history_key_length);
//
//   // -----------------------------------------------------------------------
//   // Step 1: Root Node Materialization (Creation & Payload Installation)
//   // -----------------------------------------------------------------------
//   MaterializationTicketId ticket = 1;
//   FindOrCreateResult create_res =
//       graph.FindOrCreateMaterializing(root_key, ticket);
//   if (create_res.created) {
//     // Run NN evaluation or terminal detection to obtain legal moves and priors:
//     ExpansionPayload payload;
//     payload.terminal = TerminalKind::kNonTerminal;
//     payload.moves = {move1, move2, move3};
//     payload.priors = {0.6f, 0.3f, 0.1f};
//
//     bool accepted = false;
//     uint64_t gen = graph.InstallPayload(root_key, ticket, payload, &accepted);
//     assert(accepted);
//   }
//
//   // -----------------------------------------------------------------------
//   // Step 2: Visit Selection (Traversing the graph via PUCT)
//   // -----------------------------------------------------------------------
//   struct PathStep {
//     NodeKey key;
//     uint64_t generation;
//     std::optional<Move> selected_move;
//   };
//   std::vector<PathStep> path;
//   NodeKey current_key = root_key;
//
//   while (true) {
//     // Inspect node metadata without copying full edge vectors:
//     auto meta = graph.SnapshotNodeMetadata(current_key);
//     if (!meta || meta->lifecycle == NodeLifecycle::kMaterializing) {
//       // Node needs expansion: suspend visit, schedule eval job, and wait.
//       break;
//     }
//     if (meta->terminal != TerminalKind::kNonTerminal) {
//       // Reached a terminal leaf (e.g. checkmate or stalemate).
//       break;
//     }
//
//     // Select edge using PUCT and reserve an in-flight visit (virtual loss):
//     SelectResult sel =
//         graph.SelectAndReserve(current_key, meta->generation, settings);
//     if (sel.status != SelectStatus::kSelected) {
//       // Stale generation, node missing, or concurrent change: handle retry.
//       break;
//     }
//
//     path.push_back({current_key, sel.generation, sel.move});
//     history.Append(sel.move);
//     current_key = MakeNodeKey(history, settings.history_key_length);
//   }
//
//   // -----------------------------------------------------------------------
//   // Step 3: Backpropagation (Committing values and clearing virtual loss)
//   // -----------------------------------------------------------------------
//   // Assume evaluation of the leaf yielded a SearchValue {q, d, m}:
//   SearchValue value = {0.25f, 0.5f, 12.0f};  // From leaf perspective
//
//   for (size_t i = path.size(); i-- > 0;) {
//     const PathStep& step = path[i];
//     // Backup node value and complete edge reservation for the move taken:
//     BackupResult res =
//         graph.BackupNode(step.key, step.generation, value, step.selected_move);
//     if (res.node != UpdateResult::kApplied) {
//       // Node was stale or missing; record metric / ignore.
//     }
//     value = value.Parent();  // Invert Q (-Q) and advance distance for opponent
//   }
//
//   // -----------------------------------------------------------------------
//   // Step 4: Cancellation (Rollback on search stop / interrupt)
//   // -----------------------------------------------------------------------
//   // If a visit is cancelled before backup completes, rollback all in_flight
//   // reservations so virtual loss does not leak:
//   for (const PathStep& step : path) {
//     if (step.selected_move) {
//       graph.CancelEdge(step.key, step.generation, *step.selected_move);
//     }
//   }
// clang-format on

// Unique identifier assigned to an in-flight node materialization job
// (evaluation / expansion request). Coordinated by SearchRun.
using MaterializationTicketId = uint64_t;

// Lifecycle state of a search graph node.
enum class NodeLifecycle : uint8_t {
  // Node has been inserted into the graph and assigned a
  // MaterializationTicketId, but its legal moves and policy priors have not yet
  // been published.
  kMaterializing,

  // Node expansion payload (legal moves, priors, terminal state) has been
  // published via InstallPayload(). The node is ready for PUCT selection.
  kExpanded,
};

// Represents an outgoing move edge from a node, tracking policy prior and MCTS
// statistics for that move.
//
// Edges do NOT contain child node pointers. The child NodeKey is computed
// dynamically from the position history plus the move.
struct EdgeState {
  Move move;            // The chess move represented by this edge.
  float prior = 0.0f;   // Policy network prior probability P(s, a).
  uint64_t visits = 0;  // Number of completed visits N(s, a).
  uint32_t in_flight =
      0;               // In-flight visits traversing this edge (virtual loss).
  double q_sum = 0.0;  // Accumulated win/loss value sum from completed visits.
  double d_sum = 0.0;  // Accumulated draw value sum from completed visits.
  double m_sum =
      0.0;  // Accumulated moves-to-conversion sum from completed visits.

  // Mean action value Q(s, a) in [-1.0, 1.0]. Returns 0.0f if visits == 0.
  float Q() const { return visits ? static_cast<float>(q_sum / visits) : 0.0f; }
};

// Internal representation of a search graph node.
//
// Stored within a GraphShard hash map. Holds aggregate node value statistics,
// outgoing edges, and concurrency control metadata.
struct NodeState {
  // Graph-wide incarnation ID, monotonically incremented on node creation.
  // Passed as expected_generation to operations (SelectAndReserve, BackupNode,
  // CompleteEdge, CancelEdge) to detect and reject stale operations if a node
  // was erased and recreated while a visit was in flight.
  uint64_t generation = 0;

  // Current lifecycle state (kMaterializing or kExpanded).
  NodeLifecycle lifecycle = NodeLifecycle::kMaterializing;

  // Ticket ID for pending materialization (0 once expanded).
  MaterializationTicketId ticket = 0;

  // Terminal status (checkmate, stalemate, repetition, etc.) or kNonTerminal.
  TerminalKind terminal = TerminalKind::kNonTerminal;

  // Aggregate value statistics (visit count and Q/D/M sums) accumulated across
  // all visits passing through this node.
  ValueStats value;

  // Outgoing edges corresponding to legal moves from this position.
  // Populated when the node transitions to kExpanded via InstallPayload().
  std::vector<EdgeState> edges;

  // Logical recency counter stamped on find/create, publication, and selection.
  // Not wall-clock time or a reclamation epoch. Reserved for future cache
  // eviction policies.
  uint64_t last_access_epoch = 0;
};

// Full snapshot of a node's state, copying edge vectors and value stats.
// Returned by SnapshotNode() for inspection, UCI reporting, or testing.
using NodeSnapshot = NodeState;

// Lightweight snapshot containing only node metadata, without copying the
// edge vector or value statistics.
//
// Used during visit selection (e.g. in SearchRun::AdvanceVisit) to quickly
// check whether a node is expanded or terminal before taking further action.
struct NodeMetadataSnapshot {
  uint64_t generation;
  NodeLifecycle lifecycle;
  TerminalKind terminal;
};

// Result returned by GameGraph::FindOrCreateMaterializing().
struct FindOrCreateResult {
  uint64_t generation;             // Incarnation ID of found or created node.
  bool created;                    // True if newly inserted, false if existed.
  NodeLifecycle lifecycle;         // Current lifecycle of the node.
  MaterializationTicketId ticket;  // Ticket ID for node's materialization.
};

// Status outcome of SelectAndReserve().
enum class SelectStatus {
  kSelected,       // Move successfully chosen via PUCT and in_flight reserved.
  kMissing,        // Node does not exist in the graph.
  kStale,          // Node generation differs from expected_generation.
  kMaterializing,  // Node is not yet expanded (awaiting NN/eval payload).
  kTerminal,       // Node is a terminal position or has no legal moves.
};

// Result returned by GameGraph::SelectAndReserve().
struct SelectResult {
  SelectStatus status = SelectStatus::kMissing;
  Move move{};  // Selected move (valid only when status == kSelected).
  uint64_t generation = 0;  // Node generation observed during selection.
};

// Return code for node value and edge updates (BackupNode, CompleteEdge,
// CancelEdge).
enum class UpdateResult {
  kApplied,      // Update successfully applied.
  kMissing,      // Node does not exist in the graph.
  kStale,        // Node generation differs from expected_generation.
  kEdgeMissing,  // Specified move was not found in the node's edge list.
  kUnderflow,    // Attempted to decrement in_flight when it was already zero.
};

// Decomposed result of a BackupNode() call, reporting status of the node
// value update and the edge completion separately.
struct BackupResult {
  UpdateResult node;  // Outcome of updating the node's aggregate ValueStats.
  UpdateResult edge;  // Outcome of completing the edge (kApplied if no move).
};

// GameGraph represents the game-scoped search graph for lc5.
//
// Manages nodes across 1024 shards, handling concurrent lookups, PUCT edge
// selection with in-flight reservations (virtual loss), two-phase
// materialization, and generational stale-visit protection.
class GameGraph {
 public:
  // Number of independent shards. 1024 shards provide high parallelism across
  // worker threads while keeping memory overhead minimal.
  static constexpr size_t kShardCount = 1024;

  // Finds an existing node or creates a new node in the kMaterializing state.
  //
  // If the node already exists:
  //   Returns the existing node's generation, lifecycle, and ticket ID with
  //   created = false. Updates the node's last_access_epoch.
  // If the node does not exist:
  //   Allocates a new node with a fresh generation, sets its lifecycle to
  //   kMaterializing with the provided ticket ID, and returns created = true.
  //
  // Thread-safe: locks only the shard corresponding to key.
  FindOrCreateResult FindOrCreateMaterializing(NodeKey key,
                                               MaterializationTicketId ticket);

  // Publishes the expansion payload (legal moves, priors, and terminal state)
  // for a materializing node, transitioning it to kExpanded.
  //
  // Idempotency and Race Safety:
  // - Publication is one-shot per node generation.
  // - If the node is already kExpanded or if node.ticket != ticket, the payload
  //   is rejected to prevent duplicate completions from overwriting live stats
  //   or in-flight reservations.
  // - If accepted != nullptr, *accepted reports whether the payload was used.
  //   The return value is always the current node generation.
  // - If the node did not previously exist, it is created directly as
  // kExpanded.
  //
  // Thread-safe: locks only the shard corresponding to key.
  uint64_t InstallPayload(NodeKey key, MaterializationTicketId ticket,
                          const ExpansionPayload& payload,
                          bool* accepted = nullptr);

  // Evaluates the PUCT selection formula across all outgoing edges of a node,
  // selects the best edge, and atomically increments in_flight (virtual loss).
  //
  // Selection Policy:
  // - Computes PUCT score for each edge:
  //     score = Q(s, a) + cpuct * P(s, a) * sqrt(sum_N) / (1 + N(s, a) +
  //     in_flight)
  // - Unvisited edges use First Play Urgency (FPU) from settings.fpu_strategy.
  // - Breaks ties by higher prior, then by lowest raw move representation.
  // - On success, increments edge.in_flight and returns status kSelected with
  //   the chosen Move.
  //
  // Returns:
  // - kSelected: Move chosen and in_flight reserved.
  // - kMissing: Node not found in graph.
  // - kStale: Node generation != expected_generation.
  // - kMaterializing: Node is still awaiting expansion payload.
  // - kTerminal: Node is terminal or has no legal moves.
  //
  // Thread-safe: locks only the shard corresponding to key.
  SelectResult SelectAndReserve(NodeKey key, uint64_t expected_generation,
                                const Settings::Resolved& settings);

  // Atomically updates node value statistics and optionally completes an edge.
  //
  // - Updates node.value by adding value.q, value.d, and value.m.
  // - If move is provided: locates the matching edge, decrements in_flight,
  //   increments visits, and accumulates value into the edge's Q/D/M sums.
  // - If move is std::nullopt (e.g. at the leaf node of a visit), only the node
  //   value statistics are updated.
  // - Note: The node value is committed even if edge completion fails (e.g.
  //   edge missing or in_flight underflow).
  //
  // Thread-safe: locks only the shard corresponding to key.
  BackupResult BackupNode(NodeKey key, uint64_t expected_generation,
                          SearchValue value, std::optional<Move> move);

  // Updates only the node-level ValueStats (visits and Q/D/M sums) without
  // modifying any edge.
  //
  // Typically used for root node value updates or when edge completion is
  // handled separately. Returns kMissing or kStale if generation check fails.
  //
  // Thread-safe: locks only the shard corresponding to key.
  UpdateResult UpdateNodeValue(NodeKey key, uint64_t expected_generation,
                               SearchValue value);

  // Completes an in-flight edge traversal: decrements in_flight, increments
  // visits, and adds value sums to the edge.
  //
  // Returns kUnderflow if in_flight was already 0, kEdgeMissing if the move
  // was not found, or kMissing / kStale on node/generation mismatch.
  //
  // Thread-safe: locks only the shard corresponding to key.
  UpdateResult CompleteEdge(NodeKey key, uint64_t expected_generation,
                            Move move, SearchValue value);

  // Reverts an in-flight reservation on an edge without recording a visit or
  // value sample.
  //
  // Decrements edge.in_flight. Used when a search visit is aborted or cancelled
  // before completing backup. Returns kUnderflow if in_flight was 0.
  //
  // Thread-safe: locks only the shard corresponding to key.
  UpdateResult CancelEdge(NodeKey key, uint64_t expected_generation, Move move);

  // Returns a lightweight snapshot containing only generation, lifecycle, and
  // terminal kind. Avoids copying edge vectors or value statistics.
  //
  // Returns std::nullopt if the node does not exist.
  //
  // Thread-safe: locks only the shard corresponding to key.
  std::optional<NodeMetadataSnapshot> SnapshotNodeMetadata(NodeKey key) const;

  // Returns a full copy of the node's state, including all outgoing edges and
  // value statistics.
  //
  // Returns std::nullopt if the node does not exist.
  //
  // Thread-safe: locks only the shard corresponding to key.
  std::optional<NodeSnapshot> SnapshotNode(NodeKey key) const;

  // Erases a node from the graph.
  //
  // Returns true if the node was present and erased, false otherwise.
  // Decrements graph Size() by 1.
  //
  // Thread-safe: locks only the shard corresponding to key.
  bool Erase(NodeKey key);

  // Removes all nodes across all shards and resets Size() to 0.
  //
  // Shards are locked and cleared individually to avoid holding a global lock.
  //
  // Thread-safe across all shards.
  void Clear();

  // Returns the total number of nodes currently stored across all shards.
  // Maintained atomically with relaxed memory ordering.
  size_t Size() const;

  // Captures a snapshot of all nodes across all shards.
  // Intended for unit testing, invariant validation, and debugging.
  //
  // Thread-safe: locks each shard sequentially.
  std::vector<std::pair<NodeKey, NodeSnapshot>> SnapshotAllForTesting() const;

 private:
  struct GraphShard {
    mutable std::mutex mutex;
    absl::flat_hash_map<NodeKey, NodeState> nodes;
  };

  // Maps a 64-bit NodeKey hash to a shard index in [0, kShardCount - 1]
  // using multiplicative Fibonacci hashing (golden ratio 2^64 / phi).
  static size_t ShardIndex(NodeKey key) {
    return (key.hash * 11400714819323198485ul) >> 54;
  }

  // Generates a globally unique, monotonically increasing node generation.
  uint64_t NextGeneration();

  std::array<GraphShard, kShardCount> shards_;

  // Total node count across all shards. Updated while holding affected shard
  // lock.
  std::atomic<size_t> size_{0};

  // Generator for monotonic node incarnation IDs.
  std::atomic<uint64_t> next_generation_{1};

  // Logical recency timestamp for access tracking.
  std::atomic<uint64_t> access_epoch_{1};
};

}  // namespace lczero::lc5
