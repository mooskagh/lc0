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
  size_t backprop_edge_idx;
  size_t num_visits_to_apply;
  size_t num_visits_to_undo;
  double v;
  float d;
  float m;
  double q;

  bool operator<(const NodeUpdate& other) const {
    if (variation->depth != other.variation->depth) {
      return variation->depth < other.variation->depth;
    }
    return variation->hash.hash < other.variation->hash.hash;
  }
};

NodeUpdate EvalItemToParentNodeUpdate(EvalItem* item, size_t num_visits) {
  assert(item->variation->idx_in_parent != kNoIdxInParent);
  return NodeUpdate{
      .variation = item->variation.parent(),
      .backprop_edge_idx = item->variation->idx_in_parent,
      .num_visits_to_apply = num_visits,
      .num_visits_to_undo = item->num_visits - num_visits,
      .v = -item->v,
      .d = -item->d,
      .m = item->m - 1,
      .q = -item->v,
  };
}

void MergeNodeUpdates(NodeUpdate* /* dst */, const NodeUpdate& /* src */) {
  NotImplemented();
}

void MoveNodeUpdateToParent(NodeUpdate* /* node_update */) { NotImplemented(); }

}  // namespace

void BackpropWorker::OneStep() {
  std::vector<NodeUpdate> backprop_heap;

  {
    // Fetch eval results from the queue, update the nodes they reference, and
    // forward the updates to the parent nodes.
    absl::MutexLock queue_lock(&ctx_.search_channels->request_consumer_mutex_);
    std::array<EvalItem*, 1024> buffer;
    // Fetch the first batch blockingly, then try to fetch more non-blockingly.
    size_t num_items =
        ctx_.search_channels->FetchEvalResults(buffer, /*block=*/true);
    UpdateLock update_lock = ctx_.storage->GetUpdateLock();
    do {
      CERR << "BackpropWorker::OneStep: fetched num_items=" << num_items;
      for (size_t i = 0; i < num_items; ++i) {
        // Process each EvalItem one by one.
        EvalItem* item = buffer[i];
        SortMovesByPolicy(item->moves, item->p);
        std::optional<NodeMutation> node_to_update =
            update_lock.Fetch(item->variation->hash);
        // The node was already created by the gather thread.
        assert(node_to_update);
        node_to_update->SetEdgeData(item->moves, item->p);
        if (item->terminal_type != EvalItem::TerminalType::kNonTerminal) {
          node_to_update->SetIsTerminal();
        }
        // If the node is terminal, we allow all visits to it, otherwise we
        // only apply a single NN eval.
        const size_t num_visits_to_apply =
            item->terminal_type == EvalItem::TerminalType::kNonTerminal
                ? 1
                : item->num_visits;
        node_to_update->UpdateNodeData(num_visits_to_apply, item->v, item->d,
                                       item->m);
        if (item->variation->idx_in_parent != kNoIdxInParent) {
          backprop_heap.push_back(
              EvalItemToParentNodeUpdate(item, num_visits_to_apply));
        }
      }
      // Fetch more items if they are available.
      num_items = ctx_.search_channels->FetchEvalResults(buffer,
                                                         /*block=*/false);
    } while (num_items > 0);
  }

  std::vector<EdgeUpdate> edge_updates;

  std::make_heap(backprop_heap.begin(), backprop_heap.end());
  while (!backprop_heap.empty()) {
    edge_updates.clear();
    std::pop_heap(backprop_heap.begin(), backprop_heap.end());
    NodeUpdate node_update = backprop_heap.back();
    edge_updates.push_back(EdgeUpdate{
        .edge_idx = node_update.backprop_edge_idx,
        .num_visits_to_decrement =
            static_cast<int>(node_update.num_visits_to_undo) -
            static_cast<int>(node_update.num_visits_to_apply),
        .q = node_update.q,
    });
    NodeHash cur_hash = node_update.variation->hash;
    backprop_heap.pop_back();
    while (!backprop_heap.empty() &&
           backprop_heap.front().variation->hash == cur_hash) {
      const NodeUpdate& upd = backprop_heap.front();
      edge_updates.push_back(EdgeUpdate{
          .edge_idx = upd.backprop_edge_idx,
          .num_visits_to_decrement =
              static_cast<int>(node_update.num_visits_to_undo) -
              static_cast<int>(node_update.num_visits_to_apply),
          .q = upd.q,
      });
      MergeNodeUpdates(&node_update, upd);
      std::pop_heap(backprop_heap.begin(), backprop_heap.end());
      backprop_heap.pop_back();
    }
    // TODO buffer this and apply in batches.
    UpdateLock update_lock = ctx_.storage->GetUpdateLock();
    std::optional<NodeMutation> update =
        update_lock.Fetch(node_update.variation->hash);
    assert(update);
    update->UpdateNodeData(node_update.num_visits_to_apply, node_update.v,
                           node_update.d, node_update.m);
    update->UpdateEdgeData(edge_updates);
    if (node_update.variation->idx_in_parent != kNoIdxInParent) {
      MoveNodeUpdateToParent(&node_update);
      backprop_heap.push_back(node_update);
      std::push_heap(backprop_heap.begin(), backprop_heap.end());
    }
  }
}

}  // namespace lc3
}  // namespace lczero