#include "search/lc3/gather_worker.h"

#include <queue>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "chess/gamestate.h"
#include "search/lc3/channels.h"
#include "search/lc3/positions.h"
#include "search/lc3/storage.h"
#include "utils/freelist.h"

constexpr int kExtraFetch = 1;

template <typename T>
using InlineVector = std::vector<T>;

namespace lczero {
namespace lc3 {
namespace {

std::vector<size_t> DistributeVisits(size_t /* depth */,
                                     size_t /* num_visits */,
                                     std::span<const float> /* edge_P */,
                                     std::span<const float> /* edge_Q */,
                                     std::span<const uint64_t> /* edge_N */) {
  NotImplemented();
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
    // Work queue has variations that have to be owned or deleted.
    next_iter_work_queue.clear();

    std::vector<NodeAndBatch> nodes_to_create;
    // Fetch nodes from the storage.
    {
      UpdateLock lock = ctx_.storage->GetUpdateLock();
      for (NodeAndBatch& item : work_queue) {
        Variation& node = item.node;
        std::optional<NodeMutation> update = lock.Fetch(node->hash);
        if (!update) {
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
        std::vector<size_t> edge_visits =
            DistributeVisits(depth, node_n, edge_infos.edge_P,
                             edge_infos.edge_Q, edge_infos.edge_N);
        // Spawn new work items for the children.
        for (size_t i = 0; i < num_moves_to_fetch; ++i) {
          edge_infos.edge_N[i] += edge_visits[i];
          const Move& move = edge_infos.moves[i];
          next_iter_work_queue.push_back(NodeAndBatch{
              .node = node.make_child(
                  /*hash=*/NodeHash{HashCat(node->hash.hash, move.raw_data())},
                  /*position=*/Position(node->position, move),
                  /*depth=*/node->depth + 1,
                  /*idx_in_parent=*/i),
              .batch_size = edge_visits[i],
          });
        }
        update->IncrementEdgeN(edge_infos.edge_N);
      }

      // Create new nodes for the work items that were not found in the storage.
      if (!nodes_to_create.empty()) {
        CreationLock create_lock =
            CreationLock::FromUpdateLock(std::move(lock));
        std::vector<EvalItem*> eval_tasks;
        eval_tasks.reserve(nodes_to_create.size());
        for (NodeAndBatch& item : nodes_to_create) {
          if (create_lock.Create(item.node->hash)) {
            EvalItem* task = ctx_.eval_item_pool->AllocateRaw(
                /*variation=*/std::move(item.node),
                /*num_visits=*/item.batch_size);
            eval_tasks.push_back(task);
          } else {
            // Two moves result in the same position.
            HandleCollision();
          }
        }
        ctx_.search_channels->SendEvalRequests(gather_task_idx_, eval_tasks);
      }
    }
  }
}

}  // namespace lc3
}  // namespace lczero

/*

 Q+U = edge_Q + Cpuct × sqrt((parent_N) / (1 + edge_N))



*/