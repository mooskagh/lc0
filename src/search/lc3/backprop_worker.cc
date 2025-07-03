// #define LCZERO_DEBUG_LOGGING

#include "search/lc3/backprop_worker.h"

#include <signal.h>

#include <array>
#include <optional>
#include <vector>

#include "search/lc3/channels.h"

namespace lczero {
namespace lc3 {

namespace {
void SortMovesByPolicy(std::span<Move> moves, std::span<float> p) {
  assert(moves.size() == p.size());

  std::vector<std::pair<float, Move>> p_and_move;
  p_and_move.reserve(p.size());
  for (size_t i = 0; i < p.size(); ++i) {
    p_and_move.emplace_back(p[i], moves[i]);
  }
  std::sort(p_and_move.begin(), p_and_move.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
  for (size_t i = 0; i < p.size(); ++i) {
    p[i] = p_and_move[i].first;
    moves[i] = p_and_move[i].second;
  }
}

}  // namespace

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

// TODO move to logic.h
float ComputeQ(float v, float /* d */, float /* m */) { return v; }

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

size_t MoveNodeUpdateToParent(NodeUpdate* node_update) {
  assert(node_update->variation.has_parent());
  size_t idx_in_parent =
      node_update->variation->idx_in_parent;  // Save idx_in_parent for later.
  node_update->variation =
      node_update->variation.parent();  // Move to parent variation.
  node_update->v = -node_update->v;     // Negate v for backprop as it's a
                                        // opponent's perspective.
  node_update->m -= 1;  // Decrement "moves left" for a parent node.
  return idx_in_parent;
};

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

BackpropWorker::BackPropItem BackpropWorker::EvalItemToBackpropItem(
    EvalItem* item, size_t num_visits) {
  assert(item->variation->idx_in_parent != kNoIdxInParent);
  assert(item->variation.has_parent());
  return BackPropItem{
      .node_update =
          {
              .variation = item->variation.parent(),
              .num_visits = num_visits,
              .v = -item->v,
              .d = item->d,
              .m = item->m - 1,
          },
      .edge_update =
          {
              .edge_idx = item->variation->idx_in_parent,
              .visits_to_undo = item->num_visits - num_visits,
              .agg_q = -ComputeQ(item->v, item->d, item->m),
          },
  };
}

std::pair<std::optional<BackpropWorker::BackPropItem>, bool>
BackpropWorker::ProcessSingleBackpropTask(EvalItem* item) {
  // If the node is terminal, we allow all visits to it, otherwise we
  // only apply a single NN eval.
  size_t num_visits_to_apply;
  switch (item->result_type) {
    case EvalItem::ResultType::kNormal:
      num_visits_to_apply = 1;
      break;
    case EvalItem::ResultType::kTerminal:
      num_visits_to_apply = item->num_visits;
      break;
    case EvalItem::ResultType::kCollisionRollback:
      num_visits_to_apply = 0;
  }

  NodeHandle node_to_update =
      env_.node_repository->GetNodeForUpdate(item->variation->key,
                                             /*create_if_missing=*/false);
  // The node was already created by the gather thread.
  assert(node_to_update);

  if (item->result_type == EvalItem::ResultType::kNormal) {
    SortMovesByPolicy(item->moves, item->p);
    node_to_update.InitializeEdges(item->moves, item->p);
  }

  const bool is_collision_rollback =
      item->result_type == EvalItem::ResultType::kCollisionRollback;

  if (!is_collision_rollback) {
    node_to_update.ApplyNodeUpdate({
        .n = num_visits_to_apply,
        .agg_v = item->v,
        .agg_d = item->d,
        .agg_m = item->m,
        .state = item->result_type == EvalItem::ResultType::kTerminal
                     ? NodeHandle::CertaintyState::kTerminal
                     : NodeHandle::CertaintyState::kNonTerminal,
    });
  }

  if (item->variation->idx_in_parent == kNoIdxInParent) {
    return {std::nullopt, is_collision_rollback};
  }
  return {EvalItemToBackpropItem(item, num_visits_to_apply),
          is_collision_rollback};
}

void BackpropWorker::DisposeEvalItem(EvalItem* item) {
  item->~EvalItem();
  env_.eval_item_pool->deallocate(item, 1);
}

std::optional<std::vector<BackpropWorker::BackPropItem>>
BackpropWorker ::FetchBackpropTasks() {
  std::vector<BackPropItem> backprop_items;
  // Fetch eval results from the queue, update the nodes they reference, and
  // forward the updates to the parent nodes.
  absl::MutexLock queue_lock(env_.backprop_receiver->GetConsumerMutex());
  std::array<EvalItem*, 1024> buffer;
  // Fetch the first batch blockingly, then try to fetch more non-blockingly.
  size_t num_items = env_.backprop_receiver->Collect(buffer, /*block=*/true);
  if (num_items == 0) return std::nullopt;  // Drained.
  bool all_items_collisions = true;
  do {
    for (size_t i = 0; i < num_items; ++i) {
      EvalItem* item = buffer[i];
      if (!item) continue;  // Sentinel item used for draining the queue.
      auto [backprop_item, is_collision_rollback] =
          ProcessSingleBackpropTask(item);
      if (backprop_item) backprop_items.push_back(*backprop_item);
      all_items_collisions &= is_collision_rollback;
      DisposeEvalItem(item);
    }
    // Fetch more items if they are available. If all items were collisions, do
    // not process them until we get some non-collision items.
    num_items =
        env_.backprop_receiver->Collect(buffer, /*block=*/all_items_collisions);
  } while (num_items > 0);
  return backprop_items;
}

void BackpropWorker::Run() { while (OneStep()); }

bool BackpropWorker::OneStep() {
  std::optional<std::vector<BackPropItem>> maybe_backprop_heap =
      FetchBackpropTasks();
  if (!maybe_backprop_heap) return false;
  std::vector<BackPropItem>& backprop_heap = *maybe_backprop_heap;

  std::make_heap(backprop_heap.begin(), backprop_heap.end());
  while (!backprop_heap.empty()) {
    std::pop_heap(backprop_heap.begin(), backprop_heap.end());
    BackPropItem& backprop_item = backprop_heap.back();

    // Extract the first item from the heap.
    size_t visits_to_undo = backprop_item.edge_update.visits_to_undo;
    std::vector<NodeHandle::EdgePatch> edge_updates{backprop_item.edge_update};
    NodeUpdate node_update = backprop_item.node_update;
    backprop_heap.pop_back();

    // Now accumulate all updates for the same variation.
    NodeKey cur_hash = node_update.variation->key;
    while (!backprop_heap.empty() &&
           backprop_heap.front().node_update.variation->key == cur_hash) {
      BackPropItem& backprop_item = backprop_heap.front();

      visits_to_undo += backprop_item.edge_update.visits_to_undo;

      MergeNodeUpdates(&node_update, backprop_item.node_update);
      edge_updates.push_back(backprop_item.edge_update);

      std::pop_heap(backprop_heap.begin(), backprop_heap.end());
      backprop_heap.pop_back();
    }

    // TODO buffer this and apply in batches.
    NodeHandle update =
        env_.node_repository->GetNodeForUpdate(node_update.variation->key,
                                               /*create_if_missing=*/false);
    assert(update);
    // When we rollback a collision, we don't need to apply the node update.
    if (node_update.num_visits != 0) {
      update.ApplyNodeUpdate({
          .n = node_update.num_visits,
          .agg_v = node_update.v,
          .agg_d = node_update.d,
          .agg_m = node_update.m,
          .state = NodeHandle::CertaintyState::kNonTerminal,
      });
    }
    update.UpdateEdges(edge_updates);
    if (node_update.variation->idx_in_parent != kNoIdxInParent) {
      const size_t idx_in_parent = MoveNodeUpdateToParent(&node_update);
      NodeHandle::NodeAggregates node_value = update.GetNodeAggregates();
      backprop_heap.push_back(
          {.node_update = node_update,
           .edge_update = {
               .edge_idx = idx_in_parent,
               .visits_to_undo = visits_to_undo,
               .agg_q = -ComputeQ(node_value.agg_v, node_value.agg_d,
                                  node_value.agg_m)}});
      std::push_heap(backprop_heap.begin(), backprop_heap.end());
    }
  }
  return true;
}

}  // namespace lc3
}  // namespace lczero