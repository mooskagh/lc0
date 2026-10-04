#include "search/lc5/graph.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace lczero::lc5 {

uint64_t GameGraph::NextGeneration() {
  const uint64_t generation =
      next_generation_.fetch_add(1, std::memory_order_relaxed);
  assert(generation != 0);
  return generation;
}

FindOrCreateResult GameGraph::FindOrCreateMaterializing(
    NodeKey key, MaterializationTicketId ticket) {
  auto& shard = shards_[ShardIndex(key)];
  std::lock_guard lock(shard.mutex);
  auto it = shard.nodes.find(key);
  if (it != shard.nodes.end()) {
    it->second.last_access_epoch = access_epoch_.fetch_add(1);
    return {it->second.generation, false, it->second.lifecycle,
            it->second.ticket};
  }
  NodeState state;
  state.generation = NextGeneration();
  state.ticket = ticket;
  state.last_access_epoch = access_epoch_.fetch_add(1);
  const uint64_t generation = state.generation;
  shard.nodes.emplace(key, std::move(state));
  size_.fetch_add(1, std::memory_order_relaxed);
  return {generation, true, NodeLifecycle::kMaterializing, ticket};
}

uint64_t GameGraph::InstallPayload(NodeKey key, MaterializationTicketId ticket,
                                   const ExpansionPayload& payload) {
  assert(payload.moves.size() == payload.priors.size());
  assert(payload.terminal == TerminalKind::kNonTerminal ||
         payload.moves.empty());
  auto& shard = shards_[ShardIndex(key)];
  std::lock_guard lock(shard.mutex);
  auto it = shard.nodes.find(key);
  if (it == shard.nodes.end()) {
    NodeState state;
    state.generation = NextGeneration();
    state.ticket = ticket;
    it = shard.nodes.emplace(key, std::move(state)).first;
    size_.fetch_add(1, std::memory_order_relaxed);
  }
  NodeState& node = it->second;
  // Publication is one-shot for each generation. Late or duplicate completions
  // must not reset expanded edges, including their reservations and statistics.
  if (node.lifecycle == NodeLifecycle::kExpanded || node.ticket != ticket) {
    return node.generation;
  }
  node.lifecycle = NodeLifecycle::kExpanded;
  node.ticket = 0;
  node.terminal = payload.terminal;
  node.edges.clear();
  node.edges.reserve(payload.moves.size());
  for (size_t i = 0; i < payload.moves.size(); ++i) {
    node.edges.push_back(
        EdgeState{.move = payload.moves[i], .prior = payload.priors[i]});
  }
  node.last_access_epoch = access_epoch_.fetch_add(1);
  return node.generation;
}

SelectResult GameGraph::SelectAndReserve(NodeKey key,
                                         uint64_t expected_generation,
                                         const Settings::Resolved& settings) {
  auto& shard = shards_[ShardIndex(key)];
  std::lock_guard lock(shard.mutex);
  auto it = shard.nodes.find(key);
  if (it == shard.nodes.end()) return {.status = SelectStatus::kMissing};
  NodeState& node = it->second;
  if (node.generation != expected_generation) {
    return {.status = SelectStatus::kStale, .generation = node.generation};
  }
  if (node.lifecycle == NodeLifecycle::kMaterializing) {
    return {.status = SelectStatus::kMaterializing,
            .generation = node.generation};
  }
  if (node.terminal != TerminalKind::kNonTerminal || node.edges.empty()) {
    return {.status = SelectStatus::kTerminal, .generation = node.generation};
  }
  uint64_t children_started = 0;
  float visited_policy = 0.0f;
  for (const auto& edge : node.edges) {
    const uint64_t started = edge.visits + edge.in_flight;
    children_started += started;
    if (started > 0) visited_policy += edge.prior;
  }
  const float cpuct =
      settings.cpuct +
      settings.cpuct_factor *
          std::log((node.value.visits + settings.cpuct_base) /
                   settings.cpuct_base);
  const float scale = std::sqrt(static_cast<float>(std::max<uint64_t>(1, children_started)));
  size_t best = 0;
  float best_score = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < node.edges.size(); ++i) {
    const auto& edge = node.edges[i];
    const uint64_t started = edge.visits + edge.in_flight;
    const float q = edge.visits
                        ? edge.Q()
                        : settings.fpu_strategy == FpuStrategy::kAbsolute
                              ? settings.fpu_value
                              : node.value.Q() - settings.fpu_value *
                                                     std::sqrt(visited_policy);
    const float score = q + cpuct * edge.prior * scale / (1.0f + started);
    const auto& current = node.edges[best];
    if (score > best_score ||
        (score == best_score &&
         (edge.prior > current.prior ||
          (edge.prior == current.prior &&
           edge.move.raw_data() < current.move.raw_data())))) {
      best = i;
      best_score = score;
    }
  }
  ++node.edges[best].in_flight;
  node.last_access_epoch = access_epoch_.fetch_add(1);
  return {.status = SelectStatus::kSelected,
          .move = node.edges[best].move,
          .generation = node.generation};
}

UpdateResult GameGraph::UpdateNodeValue(NodeKey key,
                                        uint64_t expected_generation,
                                        SearchValue value) {
  auto& shard = shards_[ShardIndex(key)];
  std::lock_guard lock(shard.mutex);
  auto it = shard.nodes.find(key);
  if (it == shard.nodes.end()) return UpdateResult::kMissing;
  if (it->second.generation != expected_generation) return UpdateResult::kStale;
  it->second.value.Add(value.q, value.d, value.m);
  return UpdateResult::kApplied;
}

UpdateResult GameGraph::CompleteEdge(NodeKey key,
                                     uint64_t expected_generation, Move move,
                                     SearchValue value) {
  auto& shard = shards_[ShardIndex(key)];
  std::lock_guard lock(shard.mutex);
  auto it = shard.nodes.find(key);
  if (it == shard.nodes.end()) return UpdateResult::kMissing;
  if (it->second.generation != expected_generation) return UpdateResult::kStale;
  auto edge = std::find_if(it->second.edges.begin(), it->second.edges.end(),
                           [move](const EdgeState& e) { return e.move == move; });
  if (edge == it->second.edges.end()) return UpdateResult::kEdgeMissing;
  if (edge->in_flight == 0) return UpdateResult::kUnderflow;
  --edge->in_flight;
  ++edge->visits;
  edge->q_sum += value.q;
  edge->d_sum += value.d;
  edge->m_sum += value.m;
  return UpdateResult::kApplied;
}

UpdateResult GameGraph::CancelEdge(NodeKey key,
                                   uint64_t expected_generation, Move move) {
  auto& shard = shards_[ShardIndex(key)];
  std::lock_guard lock(shard.mutex);
  auto it = shard.nodes.find(key);
  if (it == shard.nodes.end()) return UpdateResult::kMissing;
  if (it->second.generation != expected_generation) return UpdateResult::kStale;
  auto edge = std::find_if(it->second.edges.begin(), it->second.edges.end(),
                           [move](const EdgeState& e) { return e.move == move; });
  if (edge == it->second.edges.end()) return UpdateResult::kEdgeMissing;
  if (edge->in_flight == 0) return UpdateResult::kUnderflow;
  --edge->in_flight;
  return UpdateResult::kApplied;
}

std::optional<NodeMetadataSnapshot> GameGraph::SnapshotNodeMetadata(
    NodeKey key) const {
  auto& shard = shards_[ShardIndex(key)];
  std::lock_guard lock(shard.mutex);
  const auto it = shard.nodes.find(key);
  if (it == shard.nodes.end()) return std::nullopt;
  const NodeState& node = it->second;
  return NodeMetadataSnapshot{node.generation, node.lifecycle, node.terminal};
}

std::optional<NodeSnapshot> GameGraph::SnapshotNode(NodeKey key) const {
  auto& shard = shards_[ShardIndex(key)];
  std::lock_guard lock(shard.mutex);
  const auto it = shard.nodes.find(key);
  return it == shard.nodes.end() ? std::nullopt
                                 : std::optional<NodeSnapshot>(it->second);
}

bool GameGraph::Erase(NodeKey key) {
  auto& shard = shards_[ShardIndex(key)];
  std::lock_guard lock(shard.mutex);
  if (shard.nodes.erase(key) == 0) return false;
  size_.fetch_sub(1, std::memory_order_relaxed);
  return true;
}

void GameGraph::Clear() {
  for (auto& shard : shards_) {
    std::lock_guard lock(shard.mutex);
    const size_t removed = shard.nodes.size();
    shard.nodes.clear();
    // Preserve concurrent insertions into shards that have already been cleared.
    size_.fetch_sub(removed, std::memory_order_relaxed);
  }
}

size_t GameGraph::Size() const {
  return size_.load(std::memory_order_relaxed);
}

std::vector<std::pair<NodeKey, NodeSnapshot>>
GameGraph::SnapshotAllForTesting() const {
  std::vector<std::pair<NodeKey, NodeSnapshot>> result;
  for (const auto& shard : shards_) {
    std::lock_guard lock(shard.mutex);
    result.insert(result.end(), shard.nodes.begin(), shard.nodes.end());
  }
  return result;
}

}  // namespace lczero::lc5
