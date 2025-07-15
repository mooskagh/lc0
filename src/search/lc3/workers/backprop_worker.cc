// #define LCZERO_DEBUG_LOGGING

#include "search/lc3/workers/backprop_worker.h"

#include <absl/algorithm/container.h>
#include <absl/cleanup/cleanup.h>
#include <absl/container/fixed_array.h>
#include <signal.h>

#include <array>
#include <optional>
#include <vector>

#include "search/lc3/workers/node_event_queue.h"

namespace lczero {
namespace lc3 {

struct BackpropWorker::NodeUpdate {
  Variation variation;
  Policy::ValueDelta value_delta;
  absl::InlinedVector<NodeHandle::EdgeMutation, 8> edge_updates;

  bool operator<(const NodeUpdate& other) const {
    if (variation->depth != other.variation->depth) {
      return variation->depth < other.variation->depth;
    }
    return variation->key < other.variation->key;
  }
};

void BackpropWorker::Run() { while (OneStep()); }

// Fetch eval results from the queue, update the nodes they reference, and
// forward the updates to the parent nodes.
// Returns an empty vector after the queue is drained.
std::vector<BackpropWorker::NodeUpdate> BackpropWorker::FetchBackpropTasks() {
  std::vector<NodeUpdate> node_updates;
  absl::MutexLock queue_lock(env_.backprop_receiver->GetConsumerMutex());
  std::array<NodeEvent*, 1024> buffer;
  // Fetch the first batch blockingly (note we are under mutex).
  size_t num_events = env_.backprop_receiver->Collect(buffer, /*block=*/true);
  // If no events despite being blocking, we are in draining mode.
  if (num_events == 0) return {};
  // If all events we received so far are collisions, do not backprop them until
  // we get any non-collision items.
  bool all_events_collisions = true;
  do {
    for (size_t i = 0; i < num_events; ++i) {
      NodeEvent* event = buffer[i];
      if (!event) continue;  // Skip sentinel item used for draining the queue.
      // NodeEvent ends its lifetime here.
      absl::Cleanup dispose_node_event = [&]() { DisposeNodeEvent(event); };
      const bool is_collision =
          event->result_type == NodeEvent::ResultType::kCollisionRollback;
      all_events_collisions &= is_collision;

      // Update the node in the repository (set value, populate edges).
      Policy::ValueDelta delta = Policy::NodeEventToValueDelta(event);
      NodeHandle::NodeAggregates node_value =
          Policy::ValueDeltaToNodeAggregates(delta);
      if (!is_collision) UpdateLeafNode(event, node_value);

      // If the node is already root, we do not backprop it.
      if (event->variation->idx_in_parent == kNoIdxInParent) continue;

      Policy::MoveNodeUpdateToParent(&delta);
      node_updates.push_back({
          .variation = event->variation,
          .value_delta = delta,
          .edge_updates = {Policy::MakeEdgeDelta(
              event->variation->idx_in_parent, delta, node_value)},
      });
    }
    // Fetch more items if they are available. If all items were collisions, do
    // not process them until we get some non-collision items.
    num_events = env_.backprop_receiver->Collect(
        buffer, /*block=*/all_events_collisions);
  } while (num_events > 0);
  return node_updates;
}

namespace {
// Permutes `moves` and `p` in-place so that `p` is sorted in descending order.
void SortMovesByPolicy(std::span<Move> moves, std::span<float> p) {
  assert(moves.size() == p.size());
  absl::FixedArray<std::pair<float, Move>> p_and_move(p.size());
  for (size_t i = 0; i < p.size(); ++i) p_and_move[i] = {p[i], moves[i]};

  absl::c_sort(p_and_move,
               [](const auto& a, const auto& b) { return a.first > b.first; });
  for (size_t i = 0; i < p.size(); ++i) {
    p[i] = p_and_move[i].first;
    moves[i] = p_and_move[i].second;
  }
}
}  // namespace

void BackpropWorker::UpdateLeafNode(
    NodeEvent* event, const NodeHandle::NodeAggregates& node_value) {
  // For the "leaf" node, in addition to initializing values like for the rest
  // of backprop, we'll need to initialize edges.
  NodeHandle node_to_update =
      env_.node_repository->GetNodeForUpdate(event->variation->key,
                                             /*create_if_missing=*/false);
  assert(node_to_update);  // Gather thread already created the node.

  // Terminal nodes have no edges, collisions already have edges initialized.
  if (event->result_type == NodeEvent::ResultType::kNormal) {
    SortMovesByPolicy(event->moves, event->p);
    node_to_update.InitializeEdges(event->moves, event->p);
  }
  node_to_update.SetNodeAggregates(node_value);
}

BackpropWorker::NodeUpdate BackpropWorker::CollectSameVariationUpdates(
    std::vector<NodeUpdate>& backprop_heap) {
  absl::c_pop_heap(backprop_heap);
  // Extract the first item from the heap.
  NodeUpdate combined_item = std::move(backprop_heap.back());
  backprop_heap.pop_back();

  // Now combine the updates for the same variation coming from different edges.
  NodeKey cur_hash = combined_item.variation->key;
  while (!backprop_heap.empty() &&
         backprop_heap.front().variation->key == cur_hash) {
    NodeUpdate& update = backprop_heap.front();

    // Weighted (by num_visits) average of v, d, m.
    Policy::MergeNodeUpdates(&combined_item.value_delta, update.value_delta);
    combined_item.edge_updates.insert(combined_item.edge_updates.end(),
                                      update.edge_updates.begin(),
                                      update.edge_updates.end());
    absl::c_pop_heap(backprop_heap);
    backprop_heap.pop_back();
  }
  return combined_item;
}

NodeHandle::NodeAggregates BackpropWorker::UpdateNodeInRepository(
    const NodeUpdate& update) {
  NodeHandle node_handle =
      env_.node_repository->GetNodeForUpdate(update.variation->key,
                                             /*create_if_missing=*/false);
  assert(node_handle);
  // Undo per-edge number of visits, and cache child node value.
  node_handle.UpdateEdges(update.edge_updates);
  NodeHandle::NodeAggregates node_aggregates = node_handle.GetNodeAggregates();
  if (Policy::UpdateNodeAggregate(&node_aggregates, update.value_delta)) {
    node_handle.SetNodeAggregates(node_aggregates);
  }
  return node_handle.GetNodeAggregates();
}

bool BackpropWorker::OneStep() {
  std::vector<NodeUpdate> backprop_heap = FetchBackpropTasks();
  if (backprop_heap.empty()) return false;  // Drained.

  // `backprop_heap` is a queue of nodes to backpropagate, sorted by depth and
  // position. That means that the same position will be fetched sequentially.
  // The updates for the same position come through different edges, so the
  // `edge_update` will be different.
  absl::c_make_heap(backprop_heap);
  while (!backprop_heap.empty()) {
    // Collect all updates for the same variation (position).
    NodeUpdate update = CollectSameVariationUpdates(backprop_heap);
    const NodeHandle::NodeAggregates updated_node_value =
        UpdateNodeInRepository(update);
    // If the node is root, we do not backprop it.
    const size_t idx_in_parent = update.variation->idx_in_parent;
    if (idx_in_parent == kNoIdxInParent) continue;

    Policy::MoveNodeUpdateToParent(&update.value_delta);
    assert(update.variation.has_parent());
    update.variation = update.variation.parent();  // Move to parent variation.
    // Modify the value that we backpropagate for the parent node (i.e. flip
    // WDL, add 1 to moves left, etc.).
    update.edge_updates = {Policy::MakeEdgeDelta(
        idx_in_parent, update.value_delta, updated_node_value)};
    backprop_heap.push_back(std::move(update));
    absl::c_push_heap(backprop_heap);
  }
  return true;
}

void BackpropWorker::DisposeNodeEvent(NodeEvent* event) {
  event->~NodeEvent();
  env_.eval_item_pool->deallocate(event, 1);
}

}  // namespace lc3
}  // namespace lczero
