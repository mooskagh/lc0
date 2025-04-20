#include "search/lc3/mcts_worker.h"

#include <queue>
#include <utility>

#include "chess/gamestate.h"

constexpr int kExtraFetch = 1;

template <typename T>
using InlineVector = std::vector<T>;

namespace lczero {
namespace lc3 {
namespace {

std::vector<size_t> DistributeVisits(size_t depth, size_t num_visits,
                                     std::span<const float> edge_P,
                                     std::span<const float> edge_Q,
                                     std::span<const uint64_t> edge_N) {
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

MctsWorker::MctsWorker(NodeStorage* storage, PositionChain head,
                       EvalQueue* eval_queue)
    : storage_(storage),
      root_(std::make_unique<WorkTreeNode>(
          /*parent=*/nullptr,
          /*position=*/head,
          /*index_in_parent=*/-1)),
      eval_queue_(eval_queue) {}

void MctsWorker::GatherDescent(size_t target_batch_size) {
  struct NodeAndBatch {
    WorkTreeNode* node_id;
    size_t batch_size;
  };
  std::vector<NodeAndBatch> work_queue(1, NodeAndBatch{
                                              .node_id = root_.get(),
                                              .batch_size = target_batch_size,
                                          });
  std::vector<NodeAndBatch> next_iter_work_queue;

  for (size_t depth = 0;; ++depth) {
    if (work_queue.empty()) break;
    next_iter_work_queue.clear();

    std::vector<NodeAndBatch> nodes_to_create;
    // Fetch nodes from the storage.
    {
      UpdateLock lock = storage_->GetUpdateLock();
      for (NodeAndBatch& item : work_queue) {
        WorkTreeNode& node = *item.node_id;
        const NodeHash node_hash = node.position.hash;
        std::optional<NodeUpdate> update = lock.Fetch(node_hash);
        if (!update) {
          nodes_to_create.push_back(item);
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
        const uint64_t new_n = update->IncrementN(item.batch_size);
        const size_t num_moves = update->FetchNumMoves();
        const size_t num_moves_with_visits = update->FetchNumMovesWithVisits();
        const size_t num_moves_to_fetch =
            std::min(num_moves, kExtraFetch + num_moves_with_visits);

        EdgeInfos edge_infos(num_moves_to_fetch);
        NodeUpdate::EdgeDataRequest request{
            .moves = edge_infos.moves,
            .p = edge_infos.edge_P,
            .q = edge_infos.edge_Q,
            .n = edge_infos.edge_N,
        };
        std::vector<size_t> edge_visits =
            DistributeVisits(depth, new_n, edge_infos.edge_P, edge_infos.edge_Q,
                             edge_infos.edge_N);
        for (size_t i = 0; i < num_moves_to_fetch; ++i) {
          edge_infos.edge_N[i] += edge_visits[i];
        }
        update->UpdateEdgeN(edge_infos.edge_N);

        // Spawn new work items for the children.
        for (size_t i = 0; i < num_moves_to_fetch; ++i) {
          if (edge_visits[i] == 0) continue;
          if (!node.children[i]) {
            node.children[i] = std::make_unique<WorkTreeNode>(
                /*parent=*/&node,
                /*position=*/
                PositionChain::FromMove(&node.position, edge_infos.moves[i]),
                /*index_in_parent=*/i);
          }
          next_iter_work_queue.push_back(NodeAndBatch{
              .node_id = node.children[i].get(),
              .batch_size = edge_visits[i],
          });
        }
      }

      // Create new nodes for the work items that were not found in the storage.
      if (!nodes_to_create.empty()) {
        CreationLock create_lock =
            CreationLock::FromUpdateLock(std::move(lock));
        std::vector<WorkTreeNode*> nodes_to_create_ptrs;
        nodes_to_create_ptrs.reserve(nodes_to_create.size());
        for (NodeAndBatch& item : nodes_to_create) {
          if (create_lock.Create(item.node_id->position.hash,
                                 item.batch_size)) {
            nodes_to_create_ptrs.push_back(item.node_id);
          } else {
            // Two moves result in the same position.
            HandleCollision();
          }
        }
        eval_queue_->enqueue_bulk(ptok_, nodes_to_create_ptrs.data(),
                                  nodes_to_create.size());
      }
    }
  }
}

}  // namespace lc3
}  // namespace lczero

/*

 Q+U = edge_Q + Cpuct × sqrt((parent_N) / (1 + edge_N))



*/