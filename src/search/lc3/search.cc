
#include "search/lc3/search.h"

#include <queue>
#include <utility>

#include "chess/gamestate.h"

constexpr int kExtraFetch = 1;

template <typename T>
using InlineVector = std::vector<T>;

namespace lczero {
namespace lc3 {

namespace {
std::vector<size_t> DistributeVisits(size_t num_visits,
                                     std::span<const float> edge_P,
                                     std::span<const float> edge_Q,
                                     std::span<const uint64_t> edge_N);
}

NodeHash ComputePositionHash(const Position& pos);

void HandleCollision();
void HandleTerminal();

void Search::GatherDescent(const Position& head, size_t target_batch_size) {
  const NodeHash head_hash = ComputePositionHash(head);
  struct WorkItem {
    Position pos;
    size_t batch_size;
    // TreeNodeId tree_node;
  };
  std::vector<WorkItem> work_items;
  std::vector<WorkItem> new_work_items;
  work_items.push_back({
      .pos = head, .batch_size = target_batch_size,
      // .tree_node = work_tree_.MakeRootNode(),
  });
  work_tree_nodes_.push_back({
      .parent = TreeNodeId(-1),
      .node_hash = head_hash,
  });

  for (size_t depth = 0;; ++depth) {
    if (work_items.empty()) break;
    new_work_items.clear();

    std::vector<WorkItem*> nodes_to_create;
    // Fetch nodes from the storage.
    {
      UpdateLock lock = storage_.GetUpdateLock();
      for (WorkItem& item : work_items) {
        const NodeHash node_hash = ComputePositionHash(item.pos);
        std::optional<NodeUpdate> update = lock.Fetch(node_hash);
        if (!update) {
          nodes_to_create.push_back(&item);
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
        InlineVector<Move> moves;
        InlineVector<float> edge_P;
        InlineVector<float> edge_Q;
        InlineVector<uint64_t> edge_N;

        moves.resize(num_moves_to_fetch);
        edge_P.resize(num_moves_to_fetch);
        edge_Q.resize(num_moves_to_fetch);
        edge_N.resize(num_moves_to_fetch);
        NodeUpdate::EdgeDataRequest request{
            .moves = moves,
            .p = edge_P,
            .q = edge_Q,
            .n = edge_N,
        };
        std::vector<size_t> edge_visits =
            DistributeVisits(new_n, edge_P, edge_Q, edge_N);
        for (size_t i = 0; i < num_moves_to_fetch; ++i) {
          edge_N[i] += edge_visits[i];
        }
        update->UpdateEdgeN(edge_N);

        // Spawn new work items for the children.
        for (size_t i = 0; i < num_moves_to_fetch; ++i) {
          if (edge_visits[i] == 0) continue;
          new_work_items.push_back({
              .pos = Position(item.pos, moves[i]), .batch_size = edge_visits[i],
              // .tree_node = work_tree_.MakeChildNode(item.tree_node),
          });
        }
      }

      // Create new nodes for the work items that were not found in the storage.
      if (!nodes_to_create.empty()) {
        CreateLock create_lock = CreateLock::FromUpdateLock(std::move(lock));
        // Handle terminal
      }
    }
    std::swap(work_items, new_work_items);
  }
}

}  // namespace lc3
}  // namespace lczero

/*

 Q+U = edge_Q + Cpuct × sqrt((parent_N) / (1 + edge_N))



*/