// #define LCZERO_DEBUG_LOGGING

#include "search/lc3/workers/gather_worker.h"

#include <absl/algorithm/container.h>

#include <queue>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "chess/gamestate.h"
#include "search/lc3/node_repository/node_repository.h"
#include "search/lc3/workers/node_event_queue.h"
#include "utils/freelist.h"

namespace lczero {
namespace lc3 {

struct GatherWorker::NodeAndBatch {
  Variation node;
  size_t batch_size;
};

void GatherWorker::Stop() {
  stop_requested_.store(true, std::memory_order_relaxed);
}

void GatherWorker::Run() {
  while (true) {
    env_.rate_limiter->Wait();
    if (stop_requested_.load(std::memory_order_relaxed)) break;
    GatherDescent(2560);
  }
}

void GatherWorker::GatherDescent(size_t target_batch_size) {
  work_queue_.clear();

  // Seed with the head node.
  work_queue_.emplace_back(NodeAndBatch{
      .node = *env_.head,
      .batch_size = target_batch_size,
  });

  // Propagate the work wave through the depth of the tree.
  for (size_t depth = 0; !work_queue_.empty(); ++depth) {
    next_depth_work_queue_.clear();
    for (NodeAndBatch& item : work_queue_) ProcessNode(depth, item);
    next_depth_work_queue_.swap(work_queue_);
  }
}

void GatherWorker::ProcessNode(size_t depth, NodeAndBatch& item) {
  Variation& node = item.node;
  // Fetch the node (and hold the lock until the end of the function).
  NodeHandle node_handle =
      env_.node_repository->GetNodeForUpdate(node->key,
                                             /*create_if_missing=*/true);
  assert(node_handle);  // create_if_missing=true always returns a valid handle.
  if (node_handle.IsNew()) {
    // Freshly created, send for evaluation.
    EnqueueNodeForEval(std::move(node), item.batch_size);
    return;
  }
  NodeHandle::NodeAggregates aggregates = node_handle.GetNodeAggregates();
  if (aggregates.IsTerminal()) {
    // Known terminal node, send for backpropagation.
    EnqueueNodeForBackprop(std::move(node), aggregates, item.batch_size);
    return;
  }
  if (aggregates.n == 0) {
    // Node exists but has no visits means it's a collision (another thread just
    // created it but evaluate didn't finish). Send it for collision
    // rollback.
    // TODO Currently N=0 is a signal for that, however in future we might have
    // legit nodes with N=0, so we'll have to have explicit flag.
    EnqueueNodeForCollisionRollback(std::move(node), item.batch_size);
    return;
  }

  // It's just a regular node in the middle of a tree, forwarding visits to
  // children.
  ForwardToChildren(node_handle, depth, item.batch_size, aggregates.n, node);
}

namespace {

// Struct to fetch the edge data from the node repository.
// It's parallel vectors because the node repository has such API and also
// there's more hope to vectorization.
struct EdgeInfos {
  EdgeInfos(size_t num_edges)
      : moves(num_edges),
        edge_P(num_edges),
        edge_Q(num_edges),
        edge_N(num_edges) {}

  absl::InlinedVector<Move, 8> moves;
  absl::InlinedVector<float, 8> edge_P;
  absl::InlinedVector<float, 8> edge_Q;
  absl::InlinedVector<uint64_t, 8> edge_N;
};
}  // namespace

void GatherWorker::ForwardToChildren(NodeHandle& node_handle, size_t depth,
                                     size_t batch_size, size_t parent_n,
                                     Variation& node) {
  const NodeHandle::MoveCounts move_counts = node_handle.FetchMoveCounts();
  const size_t num_moves_to_fetch = Policy::GetNumEdgesToFetch(
      move_counts.total, move_counts.with_visits, depth, batch_size);

  EdgeInfos edge_infos(num_moves_to_fetch);
  NodeHandle::EdgeDataDestination request{
      .moves = edge_infos.moves,
      .p = edge_infos.edge_P,
      .q = edge_infos.edge_Q,
      .n = edge_infos.edge_N,
  };
  node_handle.FetchEdges(request);
  absl::FixedArray<size_t> edge_visits =
      Policy::DistributeVisits(depth, batch_size, parent_n, edge_infos.edge_P,
                               edge_infos.edge_Q, edge_infos.edge_N);
  node_handle.AddEdgeVisits(edge_visits);
  node_handle.Release();

  // Spawn new work items for the children.
  for (size_t i = 0; i < num_moves_to_fetch; ++i) {
    if (edge_visits[i] == 0) continue;  // TODO factor out into variable.
    const Move& move = edge_infos.moves[i];
    const Position next_position = Position(node->position, move);
    next_depth_work_queue_.push_back(NodeAndBatch{
        .node = node.make_child(
            /*key=*/Policy::MakeNodeKey(node->key, move, next_position),
            /*position=*/next_position,
            /*depth=*/node->depth + 1,
            /*idx_in_parent=*/i),
        .batch_size = edge_visits[i],
    });
  }
}

template <typename... Args>
NodeEvent* GatherWorker::MakeNodeEvent(Args&&... args) {
  NodeEvent* event = env_.node_event_pool->allocate(1);
  ::new (event) NodeEvent(std::forward<Args>(args)...);
  return event;
}

void GatherWorker::EnqueueNodeForEval(Variation&& node, size_t batch_size) {
  NodeEvent* event = MakeNodeEvent(
      /*variation=*/std::move(node),
      /*num_visits=*/batch_size);
  env_.eval_sender.Enqueue(event);
}

void GatherWorker::EnqueueNodeForBackprop(
    Variation&& node, const NodeHandle::NodeAggregates& aggregates,
    size_t batch_size) {
  // For now we only do that for terminal nodes, but if needed, we can change
  // result_type below.
  assert(aggregates.IsTerminal());
  NodeEvent* event = MakeNodeEvent(
      /*variation=*/std::move(node),
      /*num_visits=*/batch_size,
      /*result_type=*/NodeEvent::ResultType::kTerminal,
      /*v=*/aggregates.agg_v,
      /*d=*/aggregates.agg_d,
      /*m=*/aggregates.agg_m);
  env_.backprop_sender.Enqueue(event);
}

void GatherWorker::EnqueueNodeForCollisionRollback(Variation&& node,
                                                   size_t batch_size) {
  NodeEvent* event = MakeNodeEvent(
      /*variation=*/std::move(node),
      /*num_visits=*/batch_size,
      /*result_type=*/NodeEvent::ResultType::kCollisionRollback);
  env_.backprop_sender.Enqueue(event);
}
GatherWorker::GatherWorker(GatherWorkerEnvironment env)
    : env_(std::move(env)) {}

GatherWorker::~GatherWorker() = default;

}  // namespace lc3
}  // namespace lczero
