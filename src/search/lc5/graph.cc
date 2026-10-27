#include "search/lc5/graph.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace lczero::lc5 {
namespace {

template <typename Map>
auto* FindNode(Map& nodes, NodeKey key) {
  const auto it = nodes.find(key);
  return it == nodes.end() ? nullptr : &it->second;
}

UpdateResult CheckGeneration(const NodeState* node, uint64_t generation) {
  if (!node) return UpdateResult::kMissing;
  if (node->generation != generation) return UpdateResult::kStale;
  return UpdateResult::kApplied;
}

EdgeState* FindEdge(NodeState& node, Move move) {
  const auto it = std::ranges::find(node.edges, move, &EdgeState::move);
  return it == node.edges.end() ? nullptr : &*it;
}

// Releases one reservation of the edge.
UpdateResult ReleaseEdge(EdgeState* edge) {
  if (!edge) return UpdateResult::kEdgeMissing;
  if (edge->in_flight == 0) return UpdateResult::kUnderflow;
  --edge->in_flight;
  return UpdateResult::kApplied;
}

}  // namespace

NodeState& GameGraph::InsertLocked(Shard& shard, NodeKey key,
                                   MaterializationTicketId ticket) {
  const uint64_t generation =
      next_generation_.fetch_add(1, std::memory_order_relaxed);
  assert(generation != 0);
  size_.fetch_add(1, std::memory_order_relaxed);
  return shard.nodes
      .emplace(key, NodeState{.generation = generation, .ticket = ticket})
      .first->second;
}

FindOrCreateResult GameGraph::FindOrCreateMaterializing(
    NodeKey key, MaterializationTicketId ticket) {
  auto& shard = ShardFor(key);
  std::lock_guard lock(shard.mutex);
  if (const NodeState* node = FindNode(shard.nodes, key)) {
    return {node->generation, false, node->lifecycle, node->ticket};
  }
  const NodeState& node = InsertLocked(shard, key, ticket);
  return {node.generation, true, node.lifecycle, ticket};
}

bool GameGraph::CancelMaterialization(NodeKey key,
                                      MaterializationTicketId ticket) {
  auto& shard = ShardFor(key);
  std::lock_guard lock(shard.mutex);
  const NodeState* node = FindNode(shard.nodes, key);
  if (!node || node->lifecycle != NodeLifecycle::kMaterializing ||
      node->ticket != ticket) {
    return false;
  }
  shard.nodes.erase(key);
  size_.fetch_sub(1, std::memory_order_relaxed);
  return true;
}

InstallResult GameGraph::InstallPayload(NodeKey key,
                                        MaterializationTicketId ticket,
                                        const ExpansionPayload& payload) {
  assert(payload.moves.size() == payload.priors.size());
  assert(payload.terminal == TerminalKind::kNonTerminal ||
         payload.moves.empty());
  auto& shard = ShardFor(key);
  std::lock_guard lock(shard.mutex);
  NodeState* existing = FindNode(shard.nodes, key);
  NodeState& node = existing ? *existing : InsertLocked(shard, key, ticket);
  // Late or duplicate completions must not reset expanded edges, including
  // their reservations and statistics.
  if (node.lifecycle == NodeLifecycle::kExpanded || node.ticket != ticket) {
    return {node.generation, false};
  }
  node.lifecycle = NodeLifecycle::kExpanded;
  node.ticket = 0;
  node.terminal = payload.terminal;
  node.edges.clear();
  node.edges.reserve(payload.moves.size());
  for (size_t i = 0; i < payload.moves.size(); ++i) {
    node.edges.push_back({.move = payload.moves[i], .prior = payload.priors[i]});
  }
  return {node.generation, true};
}

SelectResult GameGraph::SelectAndReserve(NodeKey key,
                                         uint64_t expected_generation,
                                         const Settings::Resolved& settings) {
  auto& shard = ShardFor(key);
  std::lock_guard lock(shard.mutex);
  NodeState* node = FindNode(shard.nodes, key);
  if (!node) return {.status = SelectStatus::kMissing};
  const auto reject = [&](SelectStatus status) {
    return SelectResult{.status = status, .generation = node->generation};
  };
  if (node->generation != expected_generation) {
    return reject(SelectStatus::kStale);
  }
  if (node->lifecycle == NodeLifecycle::kMaterializing) {
    return reject(SelectStatus::kMaterializing);
  }
  if (node->terminal != TerminalKind::kNonTerminal || node->edges.empty()) {
    return reject(SelectStatus::kTerminal);
  }

  const auto started = [](const EdgeState& edge) {
    return edge.value.visits + edge.in_flight;
  };
  uint64_t children_started = 0;
  float visited_policy = 0.0f;
  for (const auto& edge : node->edges) {
    children_started += started(edge);
    if (started(edge) > 0) visited_policy += edge.prior;
  }
  const float cpuct =
      settings.cpuct +
      settings.cpuct_factor *
          std::log((node->value.visits + settings.cpuct_base) /
                   settings.cpuct_base);
  const float scale =
      std::sqrt(static_cast<float>(std::max<uint64_t>(1, children_started)));
  const float fpu =
      settings.fpu_strategy == FpuStrategy::kAbsolute
          ? settings.fpu_value
          : node->value.Q() - settings.fpu_value * std::sqrt(visited_policy);

  EdgeState* best = nullptr;
  float best_score = -std::numeric_limits<float>::infinity();
  for (auto& edge : node->edges) {
    const float q = edge.value.visits ? edge.Q() : fpu;
    const float score = q + cpuct * edge.prior * scale / (1.0f + started(edge));
    if (!best || score > best_score ||
        (score == best_score && PreferOnTie(edge, *best))) {
      best = &edge;
      best_score = score;
    }
  }
  ++best->in_flight;
  return {.status = SelectStatus::kSelected,
          .move = best->move,
          .generation = node->generation};
}

BackupResult GameGraph::BackupNode(NodeKey key, uint64_t expected_generation,
                                   SearchValue value,
                                   std::optional<Move> move) {
  auto& shard = ShardFor(key);
  std::lock_guard lock(shard.mutex);
  NodeState* node = FindNode(shard.nodes, key);
  if (const auto check = CheckGeneration(node, expected_generation);
      check != UpdateResult::kApplied) {
    return {check, move ? check : UpdateResult::kApplied};
  }
  node->value.Add(value);
  if (!move) return {UpdateResult::kApplied, UpdateResult::kApplied};
  EdgeState* edge = FindEdge(*node, *move);
  const auto result = ReleaseEdge(edge);
  if (result == UpdateResult::kApplied) edge->value.Add(value);
  return {UpdateResult::kApplied, result};
}

UpdateResult GameGraph::CancelEdge(NodeKey key, uint64_t expected_generation,
                                   Move move) {
  auto& shard = ShardFor(key);
  std::lock_guard lock(shard.mutex);
  NodeState* node = FindNode(shard.nodes, key);
  if (const auto check = CheckGeneration(node, expected_generation);
      check != UpdateResult::kApplied) {
    return check;
  }
  return ReleaseEdge(FindEdge(*node, move));
}

std::optional<NodeMetadata> GameGraph::SnapshotNodeMetadata(
    NodeKey key) const {
  const auto& shard = ShardFor(key);
  std::lock_guard lock(shard.mutex);
  const NodeState* node = FindNode(shard.nodes, key);
  if (!node) return std::nullopt;
  return NodeMetadata{node->generation, node->lifecycle, node->terminal};
}

std::optional<NodeState> GameGraph::SnapshotNode(NodeKey key) const {
  const auto& shard = ShardFor(key);
  std::lock_guard lock(shard.mutex);
  const NodeState* node = FindNode(shard.nodes, key);
  if (!node) return std::nullopt;
  return *node;
}

bool GameGraph::Erase(NodeKey key) {
  auto& shard = ShardFor(key);
  std::lock_guard lock(shard.mutex);
  if (shard.nodes.erase(key) == 0) return false;
  size_.fetch_sub(1, std::memory_order_relaxed);
  return true;
}

void GameGraph::Clear() {
  for (auto& shard : shards_) {
    std::lock_guard lock(shard.mutex);
    // Subtract per shard to preserve concurrent insertions into shards that
    // have already been cleared.
    size_.fetch_sub(shard.nodes.size(), std::memory_order_relaxed);
    shard.nodes.clear();
  }
}

std::vector<std::pair<NodeKey, NodeState>> GameGraph::SnapshotAllForTesting()
    const {
  std::vector<std::pair<NodeKey, NodeState>> result;
  for (const auto& shard : shards_) {
    std::lock_guard lock(shard.mutex);
    result.insert(result.end(), shard.nodes.begin(), shard.nodes.end());
  }
  return result;
}

}  // namespace lczero::lc5
