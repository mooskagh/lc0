// #define LCZERO_DEBUG_LOGGING

#include "search/lc3/workers/backprop_worker.h"

#include <absl/algorithm/container.h>
#include <absl/container/fixed_array.h>
#include <signal.h>

#include <array>
#include <optional>
#include <vector>

#include "search/lc3/workers/node_event_queue.h"

namespace lczero {
namespace lc3 {

namespace {
struct NodeUpdate {
  Variation variation;
  size_t num_visits;
  double v;
  float d;
  float m;

  std::string ToString() const {
    return "NodeUpdate{variation=" + variation->position.DebugString() +
           ", num_visits=" + std::to_string(num_visits) +
           ", v=" + std::to_string(v) + ", d=" + std::to_string(d) +
           ", m=" + std::to_string(m) + "}";
  }
};
}  // namespace

struct BackpropWorker::BackPropItem {
  NodeUpdate node_update;
  NodeHandle::EdgePatch edge_update;

  bool operator<(const BackPropItem& other) const {
    if (node_update.variation->depth != other.node_update.variation->depth) {
      return node_update.variation->depth < other.node_update.variation->depth;
    }
    return node_update.variation->key < other.node_update.variation->key;
  }

  std::string ToString() const {
    return "BackPropItem{variation=" +
           node_update.variation->position.DebugString() +
           ", num_visits=" + std::to_string(node_update.num_visits) +
           ", v=" + std::to_string(node_update.v) +
           ", d=" + std::to_string(node_update.d) +
           ", m=" + std::to_string(node_update.m) +
           ", edge_update.edge_idx=" + std::to_string(edge_update.edge_idx) +
           ", edge_update.num_visits_to_decrement=" +
           std::to_string(edge_update.visits_to_undo) +
           ", edge_update.q=" + std::to_string(edge_update.agg_q) + "}";
  }
};

void BackpropWorker::Run() { while (OneStep()); }

namespace {
size_t GetNumVisitsToApply(const NodeEvent& event) {
  switch (event.result_type) {
    case NodeEvent::ResultType::kNormal:
      // If the node is normal, we apply one visit to it and roll back the rest.
      return 1;
    case NodeEvent::ResultType::kTerminal:
      // If the node is terminal, we apply all visits to it
      return event.num_visits;
    case NodeEvent::ResultType::kCollisionRollback:
      // If the node is a collision, we roll back all visits.
      return 0;
  }
  assert(false);
  return 0;  // Unreachable, but avoids compiler warning.
}
}  // namespace

// Fetch eval results from the queue, update the nodes they reference, and
// forward the updates to the parent nodes.
// Returns an empty vector after the queue is drained.
std::vector<BackpropWorker::BackPropItem> BackpropWorker::FetchBackpropTasks() {
  std::vector<BackPropItem> backprop_items;
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

      // Determine how many of total visits we will apply to this node. The rest
      // are rolled back. If it's a terminal node, we apply all visits, if it's
      // a normal node, we apply one, if it's a collision, we apply none.
      const size_t num_visits_to_apply = GetNumVisitsToApply(*event);
      all_events_collisions &= (num_visits_to_apply == 0);

      // Update the node in the repository (set value, populate edges).
      UpdateLeafNode(event, num_visits_to_apply);
      if (event->variation->idx_in_parent != kNoIdxInParent) {
        // If the node is already root, we do not backprop it.
        backprop_items.push_back(
            NodeEventToBackpropItem(event, num_visits_to_apply));
      }
      DisposeNodeEvent(event);
    }
    // Fetch more items if they are available. If all items were collisions, do
    // not process them until we get some non-collision items.
    num_events = env_.backprop_receiver->Collect(
        buffer, /*block=*/all_events_collisions);
  } while (num_events > 0);
  return backprop_items;
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

void BackpropWorker::UpdateLeafNode(NodeEvent* event,
                                    size_t num_visits_to_apply) {
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

  // num_visits_to_apply means collision, leaf nodes have nothing to undo.
  if (num_visits_to_apply == 0) return;
  node_to_update.ApplyNodeUpdate({
      .n = num_visits_to_apply,
      .agg_v = event->v,
      .agg_d = event->d,
      .agg_m = event->m,
      .state = event->result_type == NodeEvent::ResultType::kTerminal
                   ? NodeHandle::CertaintyState::kTerminal
                   : NodeHandle::CertaintyState::kNonTerminal,
  });
}

namespace {
// TODO move to logic.h
void MergeNodeUpdates(NodeUpdate* dst, const NodeUpdate& src) {
  assert(dst->variation->key == src.variation->key);
  if (src.num_visits == 0) return;

  // dst v, d and q are weighted averages of v, d, q, weighted by
  // num_visits_to_apply.
  const double total_visits = dst->num_visits + src.num_visits;
  const double weight = static_cast<double>(src.num_visits) / total_visits;
  dst->v += (src.v - dst->v) * weight;
  dst->d += (src.d - dst->d) * weight;
  dst->m += (src.m - dst->m) * weight;
  dst->num_visits += src.num_visits;
}

void MoveNodeUpdateToParent(NodeUpdate* node_update) {
  assert(node_update->variation.has_parent());
  node_update->variation =
      node_update->variation.parent();  // Move to parent variation.
  node_update->v = -node_update->v;     // Negate v for backprop as it's a
                                        // opponent's perspective.
  node_update->m += 1;  // Increment "moves left" for a parent node.
};

// TODO move to logic.h
float ComputeQ(float v, float /* d */, float /* m */) { return v; }
}  // namespace

struct BackpropWorker::CombinedBackPropItem {
  NodeUpdate node_update;
  absl::InlinedVector<NodeHandle::EdgePatch, 8> edge_updates;
  size_t visits_to_undo;
};

BackpropWorker::CombinedBackPropItem
BackpropWorker::CollectSameVariationUpdates(
    std::vector<BackPropItem>& backprop_heap) {
  absl::c_pop_heap(backprop_heap);
  BackPropItem& backprop_item = backprop_heap.back();

  // Extract the first item from the heap.
  CombinedBackPropItem combined_item{
      .node_update = backprop_item.node_update,
      .edge_updates = {backprop_item.edge_update},
      .visits_to_undo = backprop_item.edge_update.visits_to_undo,
  };

  backprop_heap.pop_back();

  // Now combine the updates for the same variation coming from different edges.
  NodeKey cur_hash = combined_item.node_update.variation->key;
  while (!backprop_heap.empty() &&
         backprop_heap.front().node_update.variation->key == cur_hash) {
    BackPropItem& backprop_item = backprop_heap.front();

    combined_item.visits_to_undo += backprop_item.edge_update.visits_to_undo;

    // Weighted (by num_visits) average of v, d, m.
    MergeNodeUpdates(&combined_item.node_update, backprop_item.node_update);
    combined_item.edge_updates.push_back(backprop_item.edge_update);

    absl::c_pop_heap(backprop_heap);
    backprop_heap.pop_back();
  }
  return combined_item;
}

bool BackpropWorker::OneStep() {
  std::vector<BackPropItem> backprop_heap = FetchBackpropTasks();
  if (backprop_heap.empty()) return false;  // Drained.

  // `backprop_heap` is a queue of nodes to backpropagate, sorted by depth and
  // position. That means that the same position will be fetched sequentially.
  // The updates for the same position come through different edges, so the
  // `edge_update` will be different.
  absl::c_make_heap(backprop_heap);
  while (!backprop_heap.empty()) {
    // Collect all updates for the same variation.
    auto update = CollectSameVariationUpdates(backprop_heap);

    NodeHandle node_handle = env_.node_repository->GetNodeForUpdate(
        update.node_update.variation->key,
        /*create_if_missing=*/false);
    assert(node_handle);
    // When we rollback a collision, we don't need to apply the node update.
    if (update.node_update.num_visits != 0) {
      node_handle.ApplyNodeUpdate({
          .n = update.node_update.num_visits,
          .agg_v = update.node_update.v,
          .agg_d = update.node_update.d,
          .agg_m = update.node_update.m,
          .state = NodeHandle::CertaintyState::kNonTerminal,
      });
    }
    // Undo per-edge number of visits, and cache child node value.
    node_handle.UpdateEdges(update.edge_updates);

    // If the node is root, we do not backprop it.
    const size_t idx_in_parent = update.node_update.variation->idx_in_parent;
    if (idx_in_parent == kNoIdxInParent) continue;

    // Modify the value that we backpropagate for the parent node (i.e. flip
    // WDL, add 1 to moves left, etc.).
    MoveNodeUpdateToParent(&update.node_update);
    NodeHandle::NodeAggregates node_value = node_handle.GetNodeAggregates();
    backprop_heap.push_back(
        {.node_update = update.node_update,
         .edge_update = {.edge_idx = idx_in_parent,
                         .visits_to_undo = update.visits_to_undo,
                         .agg_q = -ComputeQ(node_value.agg_v, node_value.agg_d,
                                            node_value.agg_m)}});
    absl::c_push_heap(backprop_heap);
  }
  return true;
}

BackpropWorker::BackPropItem BackpropWorker::NodeEventToBackpropItem(
    NodeEvent* event, size_t num_visits) {
  assert(event->variation->idx_in_parent != kNoIdxInParent);
  assert(event->variation.has_parent());
  return BackPropItem{
      .node_update =
          {
              .variation = event->variation.parent(),
              .num_visits = num_visits,
              .v = -event->v,
              .d = event->d,
              .m = event->m + 1,
          },
      .edge_update =
          {
              .edge_idx = event->variation->idx_in_parent,
              .visits_to_undo = event->num_visits - num_visits,
              .agg_q = -ComputeQ(event->v, event->d, event->m),
          },
  };
}

void BackpropWorker::DisposeNodeEvent(NodeEvent* event) {
  event->~NodeEvent();
  env_.eval_item_pool->deallocate(event, 1);
}

}  // namespace lc3
}  // namespace lczero