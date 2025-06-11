// #define LCZERO_DEBUG_LOGGING

#include "search/lc3/backprop_worker.h"

#include <array>
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

struct NodeUpdate {
  Variation variation;
  size_t num_visits;
  double v;
  float d;
  float m;
};

// TODO move to logic.h
float ComputeQ(float v, float /* d */, float /* m */) { return v; }

// TODO move to logic.h
void MergeNodeUpdates(NodeUpdate* dst, const NodeUpdate& src) {
  assert(dst->variation->hash == src.variation->hash);

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

struct BackPropItem {
  NodeUpdate node_update;
  StorageEdgePatch edge_update;

  bool operator<(const BackPropItem& other) const {
    if (node_update.variation->depth != other.node_update.variation->depth) {
      return node_update.variation->depth < other.node_update.variation->depth;
    }
    return node_update.variation->hash.hash <
           other.node_update.variation->hash.hash;
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

BackPropItem EvalItemToBackpropItem(EvalItem* item, size_t num_visits) {
  assert(item->variation->idx_in_parent != kNoIdxInParent);
  assert(item->variation.has_parent());
  return BackPropItem{
      .node_update =
          {
              .variation = item->variation.parent(),
              .num_visits = num_visits,
              .v = -item->v,
              .d = -item->d,
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

}  // namespace

void BackpropWorker::OneStep() {
  std::vector<BackPropItem> backprop_heap;

  {
    DPRINT_SCOPE("Fetching eval results");
    // Fetch eval results from the queue, update the nodes they reference, and
    // forward the updates to the parent nodes.
    absl::MutexLock queue_lock(&ctx_.search_channels->request_consumer_mutex_);
    std::array<EvalItem*, 1024> buffer;
    // Fetch the first batch blockingly, then try to fetch more non-blockingly.
    size_t num_items =
        ctx_.search_channels->FetchEvalResults(buffer, /*block=*/true);
    AccessLock update_lock = ctx_.node_repository->GetAccessLock();
    do {
      DPRINT << "fetched num_items=" << num_items;
      for (size_t i = 0; i < num_items; ++i) {
        // Process each EvalItem one by one.
        EvalItem* item = buffer[i];
        DPRINT_SCOPE("Processing eval item " +
                     item->variation->position.DebugString() +
                     ", num_visits=" + std::to_string(item->num_visits));
        SortMovesByPolicy(item->moves, item->p);
        std::optional<NodeMutation> node_to_update =
            update_lock.FetchMutable(item->variation->hash);
        // The node was already created by the gather thread.
        assert(node_to_update);
        // {
        //   DPRINT_SCOPE("Moves:");
        //   for (size_t j = 0; j < item->moves.size(); ++j) {
        //     DPRINT << "  move=" << item->moves[j].ToString(true)
        //            << ", p=" << item->p[j];
        //   }
        // }
        node_to_update->SetEdgeData(item->moves, item->p);
        if (item->terminal_type != EvalItem::TerminalType::kNonTerminal) {
          DPRINT << "Node is terminal, type=" << int(item->terminal_type);
          node_to_update->SetIsTerminal();
        }
        // If the node is terminal, we allow all visits to it, otherwise we
        // only apply a single NN eval.
        const size_t num_visits_to_apply =
            item->terminal_type == EvalItem::TerminalType::kNonTerminal
                ? 1
                : item->num_visits;
        node_to_update->AccumulateNodeData({
            .n = num_visits_to_apply,
            .agg_v = item->v,
            .agg_d = item->d,
            .agg_m = item->m,
        });
        if (item->variation->idx_in_parent != kNoIdxInParent) {
          backprop_heap.push_back(
              EvalItemToBackpropItem(item, num_visits_to_apply));
          DPRINT << "Forwarde to parent " << backprop_heap.back().ToString();
        }
      }
      // Fetch more items if they are available.
      num_items = ctx_.search_channels->FetchEvalResults(buffer,
                                                         /*block=*/false);
    } while (num_items > 0);
  }

  std::make_heap(backprop_heap.begin(), backprop_heap.end());
  while (!backprop_heap.empty()) {
    DPRINT_SCOPE("Processing backprop heap, size=" +
                 std::to_string(backprop_heap.size()));
    std::pop_heap(backprop_heap.begin(), backprop_heap.end());
    BackPropItem& backprop_item = backprop_heap.back();
    DPRINT << "Processing backprop item: " << backprop_item.ToString();

    // Extract the first item from the heap.
    size_t visits_to_undo = backprop_item.edge_update.visits_to_undo;
    DPRINT << "visits_to_undo_so_far=" << visits_to_undo;
    std::vector<StorageEdgePatch> edge_updates{backprop_item.edge_update};
    NodeUpdate node_update = backprop_item.node_update;
    backprop_heap.pop_back();

    // Now accumulate all updates for the same variation.
    NodeHash cur_hash = node_update.variation->hash;
    while (!backprop_heap.empty() &&
           backprop_heap.front().node_update.variation->hash == cur_hash) {
      DPRINT_SCOPE("Merging backprop");
      BackPropItem& backprop_item = backprop_heap.front();
      DPRINT << backprop_item.ToString();

      visits_to_undo += backprop_item.edge_update.visits_to_undo;
      DPRINT << "visits_to_undo += " << backprop_item.edge_update.visits_to_undo
             << " = " << visits_to_undo;

      MergeNodeUpdates(&node_update, backprop_item.node_update);
      edge_updates.push_back(backprop_item.edge_update);

      std::pop_heap(backprop_heap.begin(), backprop_heap.end());
      backprop_heap.pop_back();
    }
    DPRINT << "Merged " << edge_updates.size() << " items";

    // TODO buffer this and apply in batches.
    AccessLock update_lock = ctx_.node_repository->GetAccessLock();
    std::optional<NodeMutation> update =
        update_lock.FetchMutable(node_update.variation->hash);
    assert(update);
    DPRINT << "About to call accumulate: num_visits=" << node_update.num_visits
           << ", visits_to_undo=" << visits_to_undo;
    StorageNodeData node_value = update->AccumulateNodeData({
        .n = node_update.num_visits,
        .agg_v = node_update.v,
        .agg_d = node_update.d,
        .agg_m = node_update.m,
    });
    DPRINT << "Accumulated " << edge_updates.size() << " edge updates.";
    update->UpdateEdgeData(edge_updates);
    if (node_update.variation->idx_in_parent != kNoIdxInParent) {
      const size_t idx_in_parent = MoveNodeUpdateToParent(&node_update);
      backprop_heap.push_back(
          {.node_update = node_update,
           .edge_update = {.edge_idx = idx_in_parent,
                           .visits_to_undo = visits_to_undo,
                           .agg_q = ComputeQ(node_value.agg_v, node_value.agg_d,
                                             node_value.agg_m)}});
      DPRINT << "Forwarded to parent " << backprop_heap.back().ToString();
      std::push_heap(backprop_heap.begin(), backprop_heap.end());
    }
  }
}

}  // namespace lc3
}  // namespace lczero