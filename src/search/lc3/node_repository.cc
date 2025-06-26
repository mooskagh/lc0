// #define LCZERO_DEBUG_LOGGING
#include "search/lc3/node_repository.h"

#include <signal.h>

namespace lczero {
namespace lc3 {

namespace {
constexpr size_t kShardBits = 7;
constexpr size_t kNumShards = 1 << kShardBits;

size_t GetShardIndex(NodeKey key) {
  return (key.hash * 11400714819323198485ull) >> (64 - kShardBits);
}

struct EdgeData {
  float q = 0.0f;
  uint64_t n = 0;
};

}  // namespace

struct NodeHandle::NodeData {
  NodeAggregates value;
  std::vector<Move> moves;
  std::vector<float> p;
  std::vector<EdgeData> edges;
};

struct NodeRepository::Shard {
  absl::flat_hash_map<uint64_t, NodeHandle::NodeData> nodes;
  std::mutex mutex;
};

struct NodeRepository::StorageImpl {
  std::array<Shard, kNumShards> shards;
};

NodeRepository::NodeRepository()
    : storage_impl_(std::make_unique<StorageImpl>()) {}
NodeRepository::~NodeRepository() {}

NodeHandle NodeRepository::GetNodeForUpdate(const NodeKey& key,
                                            bool create_if_missing) {
  Shard& shard = storage_impl_->shards[GetShardIndex(key)];
  std::unique_lock lock(shard.mutex);
  auto iter = shard.nodes.find(key.hash);
  if (iter != shard.nodes.end()) {
    return NodeHandle(&iter->second, std::move(lock), false);
  }
  if (!create_if_missing) return NodeHandle();
  auto [new_iter, success] =
      shard.nodes.emplace(key.hash, NodeHandle::NodeData{});
  assert(success);
  return NodeHandle(&new_iter->second, std::move(lock), true);
}

void NodeHandle::Release() {
  if (data_ == nullptr) return;
  lock_.unlock();
  data_ = nullptr;
}

void NodeHandle::ApplyNodeUpdate(NodeAggregates new_data) {
  if (new_data.n <= 0) return;

  // Calculate the weight for the new data
  float weight = static_cast<float>(new_data.n) / (data_->value.n + new_data.n);

  data_->value.n += new_data.n;
  data_->value.agg_v += weight * (new_data.agg_v - data_->value.agg_v);
  data_->value.agg_d += weight * (new_data.agg_d - data_->value.agg_d);
  data_->value.agg_m += weight * (new_data.agg_m - data_->value.agg_m);
  // TODO probably with certainty propagation we'll need something smarter here.
  data_->value.state = new_data.state;
}

NodeHandle::NodeAggregates NodeHandle::GetNodeAggregates() const {
  return data_->value;
}

NodeHandle::MoveCounts NodeHandle::FetchMoveCounts() const {
  return {.total = data_->moves.size(), .with_visits = data_->edges.size()};
}

void NodeHandle::InitializeEdges(std::span<const Move> moves,
                                 std::span<const float> p) {
  if (moves.size() != p.size()) {
    throw Exception("Moves and probabilities arrays must have the same size");
  }
  data_->moves.assign(moves.begin(), moves.end());
  data_->p.assign(p.begin(), p.end());
}

void NodeHandle::FetchEdges(EdgeDataDestination request) const {
  const size_t num_moves = request.moves.size();
  const size_t num_edges = std::min(num_moves, data_->edges.size());

  assert(request.p.empty() || request.p.size() >= num_moves);
  assert(request.q.empty() || request.q.size() >= num_moves);
  assert(request.n.empty() || request.n.size() >= num_moves);

  std::memcpy(request.moves.data(), data_->moves.data(),
              num_moves * sizeof(Move));
  if (!request.p.empty()) {
    std::memcpy(request.p.data(), data_->p.data(), num_moves * sizeof(float));
  }

  auto fill_edge_data = [&](auto& dst, const auto& src_accessor) {
    if (dst.empty()) return;
    for (size_t i = 0; i < num_edges; ++i)
      dst[i] = src_accessor(data_->edges[i]);
    if (num_edges < num_moves) {
      std::memset(dst.data() + num_edges, 0,
                  (num_moves - num_edges) * sizeof(dst[0]));
    }
  };

  fill_edge_data(request.q, [](const auto& edge) { return edge.q; });
  fill_edge_data(request.n, [](const auto& edge) { return edge.n; });
}

void NodeHandle::AddEdgeVisits(std::span<const uint64_t> n_delta) const {
  auto& edges = data_->edges;
  if (n_delta.size() > edges.size()) edges.resize(n_delta.size());
  for (size_t i = 0; i < n_delta.size(); ++i) edges[i].n += n_delta[i];
}

void NodeHandle::UpdateEdges(std::span<const EdgePatch> updates) {
  assert(!updates.empty());
  size_t max_idx = std::max_element(updates.begin(), updates.end(),
                                    [](const EdgePatch& a, const EdgePatch& b) {
                                      return a.edge_idx < b.edge_idx;
                                    })
                       ->edge_idx;
  if (max_idx >= data_->edges.size()) {
    data_->edges.resize(max_idx + 1);
  }
  for (const EdgePatch& update : updates) {
    EdgeData& edge = data_->edges[update.edge_idx];
    edge.q = update.agg_q;
    edge.n -= update.visits_to_undo;
  }
}

}  // namespace lc3
}  // namespace lczero