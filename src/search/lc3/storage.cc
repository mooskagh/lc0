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

void NodeMutation::SetEdgeData(std::span<const Move> moves,
                               std::span<const float> p) {
  if (moves.size() != p.size()) {
    throw Exception("Moves and probabilities arrays must have the same size");
  }
  data_->moves.assign(moves.begin(), moves.end());
  data_->p.assign(p.begin(), p.end());
}

void NodeMutation::AccumulateNodeData(int new_visits, float q, float d,
                                      float m) {
  if (new_visits <= 0) return;

  // Calculate the weight for the new data
  float weight =
      static_cast<float>(new_visits) / (data_->num_visits + new_visits);

  // Use incremental update formulas (q_new = q_old + weight * (q_incoming -
  // q_old))
  data_->q += weight * (q - data_->q);
  data_->d += weight * (d - data_->d);
  data_->m += weight * (m - data_->m);

  // Update visit count
  data_->num_visits += new_visits;
}

void NodeMutation::FetchEdgeData(EdgeDataRequest request) const {
  const size_t num_moves = request.moves.size();
  assert(request.p.size() >= num_moves);
  assert(request.q.size() >= num_moves);
  assert(request.n.size() >= num_moves);

  std::memcpy(request.moves.data(), data_->moves.data(),
              num_moves * sizeof(Move));
  std::memcpy(request.p.data(), data_->p.data(), num_moves * sizeof(float));
  const size_t num_edges = std::min(num_moves, data_->edges.size());
  for (size_t i = 0; i < num_edges; ++i) {
    request.q[i] = data_->edges[i].q;
    request.n[i] = data_->edges[i].n;
  }
  if (num_edges < num_moves) {
    std::memset(request.q.data() + num_edges, 0,
                (num_moves - num_edges) * sizeof(request.q[0]));
    std::memset(request.n.data() + num_edges, 0,
                (num_moves - num_edges) * sizeof(request.n[0]));
  }
}

void NodeMutation::IncrementEdgeN(std::span<const uint64_t> n_delta) const {
  auto& edges = data_->edges;
  if (n_delta.size() > edges.size()) edges.resize(n_delta.size());
  for (size_t i = 0; i < n_delta.size(); ++i) edges[i].n += n_delta[i];
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