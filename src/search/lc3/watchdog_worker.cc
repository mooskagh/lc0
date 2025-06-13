#include "search/lc3/watchdog_worker.h"

#include <algorithm>
#include <string>
#include <vector>

#include "chess/position.h"
#include "utils/hashcat.h"

namespace lczero {
namespace lc3 {

void WatchdogWorker::CheckOnce() {
  AccessLock lock = ctx_.node_repository->GetAccessLock();
  std::optional<NodeView> root_view = lock.FetchReadOnly((*ctx_.head)->hash);
  if (!root_view) return;
  
  std::vector<Move> pv = BuildPV();
  std::string pv_str = "N=" + std::to_string(root_view->GetN()) + " PV=";
  for (const Move& move : pv) {
    pv_str += move.ToString(true) + " ";
  }
  CERR << pv_str;
}

Functions:
WatchdogWorker::BuildPV()
NodeView::FetchEdgeData
NodeMutation::FetchEdgeData
are written by AI. Review and fix the style.

std::vector<Move> WatchdogWorker::BuildPV() const {
  std::vector<Move> pv;
  AccessLock lock = ctx_.node_repository->GetAccessLock();
  
  // Start from the head position
  NodeHash current_hash = (*ctx_.head)->hash;
  Position current_position = (*ctx_.head)->position;
  
  // Follow the path of highest N edges
  while (true) {
    std::optional<NodeView> node_view = lock.FetchReadOnly(current_hash);
    if (!node_view) break;
    
    const size_t num_moves_with_visits = node_view->FetchNumMovesWithVisits();
    if (num_moves_with_visits == 0) break;
    
    // Allocate vectors for edge data
    std::vector<Move> moves(num_moves_with_visits);
    std::vector<uint64_t> n(num_moves_with_visits);
    
    // Fetch edge data
    EdgeDataRequest request{
        .moves = moves,
        .n = n,
    };
    node_view->FetchEdgeData(request);
    
    // Find the edge with the highest N
    size_t best_idx = 0;
    uint64_t best_n = n[0];
    for (size_t i = 1; i < num_moves_with_visits; ++i) {
      if (n[i] > best_n) {
        best_n = n[i];
        best_idx = i;
      }
    }
    
    // If the best edge has no visits, we're done
    if (best_n == 0) break;
    
    // Add the move to PV and advance to the next position
    const Move& best_move = moves[best_idx];
    pv.push_back(best_move);
    current_position = Position(current_position, best_move);
    current_hash = NodeHash{HashCat(current_hash.hash, best_move.raw_data())};
  }
  
  return pv;
}
}  // namespace lc3
}  // namespace lczero
