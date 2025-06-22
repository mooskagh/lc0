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
                                   GatherWorkerChannels channels)
    : ctx_(context), channels_(std::move(channels)) {}

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

    // Fetch nodes from the node_repository.
    {
      for (NodeAndBatch& item : work_queue) {
        DPRINT_SCOPE("Item " + item.node->position.DebugString());
        DPRINT << "batch_size=" << item.batch_size;
        Variation& node = item.node;
        NodeHandle update =
            ctx_.node_repository->GetNodeForUpdate(node->key,
                                                   /*create_if_missing=*/true);
        assert(update);
        if (update.IsNew()) {
          DPRINT << "Node not found in node_repository, creating new node";
          EnqueueNodeForEval(std::move(node), item.batch_size);
          continue;
        }
        NodeHandle::NodeAggregates aggregates = update.GetNodeAggregates();
        if (aggregates.IsTerminal()) {
          HandleTerminal();
          continue;
        }
        if (aggregates.n == 0) {
          HandleCollision();
          continue;
        }

        const NodeHandle::MoveCounts move_counts = update.FetchMoveCounts();
        const size_t num_moves_to_fetch =
            std::min(move_counts.total, kExtraFetch + move_counts.with_visits);

        EdgeInfos edge_infos(num_moves_to_fetch);
        NodeHandle::EdgeDataDestination request{
            .moves = edge_infos.moves,
            .p = edge_infos.edge_P,
            .q = edge_infos.edge_Q,
            .n = edge_infos.edge_N,
        };
        update.FetchEdges(request);
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
        std::vector<size_t> edge_visits = DistributeVisits(
            depth, item.batch_size, aggregates.n, edge_infos.edge_P,
            edge_infos.edge_Q, edge_infos.edge_N);
        update.AddEdgeVisits(edge_visits);
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
                  /*hash=*/NodeKey{HashCat(node->key.hash, move.raw_data())},
                  /*position=*/Position(node->position, move),
                  /*depth=*/node->depth + 1,
                  /*idx_in_parent=*/i),
              .batch_size = edge_visits[i],
          });
        }
      }
    }
  }
}

void MctsGatherWorker::EnqueueNodeForEval(Variation&& node, size_t batch_size) {
  DPRINT_SCOPE("EnqueueNodeForEval");
  EvalItem* task = ctx_.eval_item_pool->allocate(1);
  ::new (task) EvalItem(
      /*variation=*/std::move(node),
      /*num_visits=*/batch_size);
  DPRINT << "created eval_item node=" << task->variation->position.DebugString()
         << ", num_visits=" << batch_size;
  channels_.SendForEval(task);
}

}  // namespace lc3
}  // namespace lczero
