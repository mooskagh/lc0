// #define LCZERO_DEBUG_LOGGING

#include "search/lc3/gather_worker.h"

#include <queue>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "chess/gamestate.h"
#include "search/lc3/channels.h"
#include "search/lc3/node_repository.h"
#include "search/lc3/positions.h"
#include "utils/freelist.h"

// TODO Make it a function and move to logic.h
constexpr int kExtraFetch = 2;
constexpr float kCpuctConst = 1.745f;            // TODO: Make this configurable
constexpr float kBatchIterationFraction = 0.4f;  // TODO: Make this configurable

namespace {
// TODO Implement proper InlineVector
template <typename T>
using InlineVector = std::vector<T>;
}  // namespace

namespace lczero {
namespace lc3 {
namespace {

// TODO move to logic.h
std::vector<size_t> DistributeVisits(size_t depth, size_t visits_to_distribute,
                                     size_t node_n,
                                     std::span<const float> edge_P,
                                     std::span<const float> edge_Q,
                                     std::span<const uint64_t> edge_N) {
  DPRINT_SCOPE("DistributeVisits");
  DPRINT << "depth=" << depth
         << " visits_to_distribute=" << visits_to_distribute
         << " node_n=" << node_n << " edge_P.size()=" << edge_P.size()
         << " edge_Q.size()=" << edge_Q.size()
         << " edge_N.size()=" << edge_N.size();

  assert(edge_P.size() > 0);
  assert(edge_P.size() == edge_Q.size());
  assert(edge_P.size() == edge_N.size());
  // If there is only one edge, we just return all visits to it.
  if (edge_P.size() == 1) return {visits_to_distribute};

  std::vector<size_t> result(edge_P.size(), 0);

  // parent_n_sqrt × kCpuctConst
  const float factor = std::sqrt(static_cast<float>(node_n)) * kCpuctConst;
  auto q_plus_u = [&](size_t idx) {
    return edge_Q[idx] +
           factor * edge_P[idx] / (1.0f + edge_N[idx] + result[idx]);
  };

  while (visits_to_distribute > 0) {
    const size_t visits_this_step = static_cast<size_t>(
        std::ceil(visits_to_distribute * kBatchIterationFraction));
    DPRINT_SCOPE("Distributing visits, remaining=" +
                 std::to_string(visits_to_distribute) +
                 " this_step=" + std::to_string(visits_this_step));

    float best_score = q_plus_u(0);
    size_t best_idx = 0;
    for (size_t i = 1; i < edge_P.size(); ++i) {
      const float cur_score = q_plus_u(i);
      if (cur_score > best_score) {
        best_score = cur_score;
        best_idx = i;
      }
    }

    DPRINT << "Routing to idx=" << best_idx << " (num=" << visits_this_step
           << ")";
    result[best_idx] += visits_this_step;
    visits_to_distribute -= visits_this_step;
  }
  return result;
}

struct EdgeInfos {
  EdgeInfos(size_t num_edges)
      : moves(num_edges),
        edge_P(num_edges),
        edge_Q(num_edges),
        edge_N(num_edges) {}

  InlineVector<Move> moves;
  InlineVector<float> edge_P;
  InlineVector<float> edge_Q;
  InlineVector<uint64_t> edge_N;
};

}  // namespace

void HandleCollision() { NotImplemented(); }
void HandleTerminal() { NotImplemented(); }

MctsGatherWorker::MctsGatherWorker(const Context& context,
                                   size_t gather_task_idx)
    : gather_task_idx_(gather_task_idx), ctx_(context) {}

void MctsGatherWorker::GatherDescent(size_t target_batch_size) {
  DPRINT_SCOPE("GatherDescent");
  DPRINT << "target_batch_size=" << target_batch_size;
  struct NodeAndBatch {
    Variation node;
    size_t batch_size;
  };
  std::vector<NodeAndBatch> work_queue(1, NodeAndBatch{
                                              .node = *ctx_.head,
                                              .batch_size = target_batch_size,
                                          });
  std::vector<NodeAndBatch> next_iter_work_queue;

  for (size_t depth = 0; !work_queue.empty();
       ++depth, next_iter_work_queue.swap(work_queue)) {
    DPRINT_SCOPE("depth=" + std::to_string(depth));
    // Work queue has variations that have to be owned or deleted.
    next_iter_work_queue.clear();

    std::vector<NodeAndBatch> nodes_to_create;
    // Fetch nodes from the node_repository.
    {
      AccessLock lock = ctx_.node_repository->GetAccessLock();
      for (NodeAndBatch& item : work_queue) {
        DPRINT_SCOPE("Item " + item.node->position.DebugString());
        DPRINT << "batch_size=" << item.batch_size;
        Variation& node = item.node;
        std::optional<NodeMutation> update = lock.FetchMutable(node->hash);
        if (!update) {
          DPRINT << "Node not found in node_repository, creating new node";
          nodes_to_create.push_back(std::move(item));
          continue;
        }
        if (update->IsTerminal()) {
          HandleTerminal();
          continue;
        }
        if (!update->HasVisits()) {
          HandleCollision();
          continue;
        }
        const uint64_t node_n = update->GetN();
        const size_t num_moves = update->FetchNumMoves();
        const size_t num_moves_with_visits = update->FetchNumMovesWithVisits();
        const size_t num_moves_to_fetch =
            std::min(num_moves, kExtraFetch + num_moves_with_visits);

        EdgeInfos edge_infos(num_moves_to_fetch);
        NodeMutation::EdgeDataRequest request{
            .moves = edge_infos.moves,
            .p = edge_infos.edge_P,
            .q = edge_infos.edge_Q,
            .n = edge_infos.edge_N,
        };
        update->FetchEdgeData(request);
        {
          DPRINT_SCOPE("Fetched moves:");
          for (size_t i = 0; i < num_moves_to_fetch; ++i) {
            DPRINT << "idx=" << i
                   << " move=" << edge_infos.moves[i].ToString(true)
                   << " p=" << edge_infos.edge_P[i]
                   << " q=" << edge_infos.edge_Q[i]
                   << " n=" << edge_infos.edge_N[i];
          }
        }
        std::vector<size_t> edge_visits =
            DistributeVisits(depth, item.batch_size, node_n, edge_infos.edge_P,
                             edge_infos.edge_Q, edge_infos.edge_N);
        update->IncrementEdgeN(edge_visits);
        // Spawn new work items for the children.
        DPRINT_SCOPE("Spawning children");
        // TODO no need to hold an `update` lock.
        for (size_t i = 0; i < num_moves_to_fetch; ++i) {
          if (edge_visits[i] == 0) continue;  // TODO factor out into variable.
          const Move& move = edge_infos.moves[i];
          DPRINT << "pos=" << node->position.DebugString()
                 << ", move=" << move.ToString(true) << ", resulting="
                 << Position(node->position, move).DebugString();
          next_iter_work_queue.push_back(NodeAndBatch{
              .node = node.make_child(
                  /*hash=*/NodeHash{HashCat(node->hash.hash, move.raw_data())},
                  /*position=*/Position(node->position, move),
                  /*depth=*/node->depth + 1,
                  /*idx_in_parent=*/i),
              .batch_size = edge_visits[i],
          });
        }
      }

      // Create new nodes for the work items that were not found in the
      // node_repository.
      if (!nodes_to_create.empty()) {
        DPRINT_SCOPE("Creating new nodes. count=" +
                     std::to_string(nodes_to_create.size()));
        CreationLock create_lock =
            CreationLock::FromAccessLock(std::move(lock));
        std::vector<EvalItem*> eval_items;
        eval_items.reserve(nodes_to_create.size());
        for (NodeAndBatch& item : nodes_to_create) {
          DPRINT_SCOPE("Creating node " + item.node->position.DebugString());
          if (create_lock.Create(item.node->hash)) {
            EvalItem* task = ctx_.eval_item_pool->allocate(1);
            ::new (task) EvalItem(
                /*variation=*/std::move(item.node),
                /*num_visits=*/item.batch_size);
            DPRINT << "created eval_item node="
                   << task->variation->position.DebugString()
                   << ", num_visits=" << item.batch_size;
            eval_items.push_back(task);
          } else {
            DPRINT << "Collision";
            // Two moves result in the same position.
            HandleCollision();
          }
        }
        DPRINT << "sending eval_items.size()=" << eval_items.size();
        ctx_.search_channels->SendEvalRequests(gather_task_idx_, eval_items);
      }
    }
  }
}

}  // namespace lc3
}  // namespace lczero
