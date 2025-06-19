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

std::vector<Move> WatchdogWorker::BuildPV() const {
  std::vector<Move> pv;
  AccessLock lock = ctx_.node_repository->GetAccessLock();
  NodeHash current_hash = (*ctx_.head)->hash;
  Position current_position = (*ctx_.head)->position;

  while (auto node_view = lock.FetchReadOnly(current_hash)) {
    const size_t num_moves = node_view->FetchNumMovesWithVisits();
    if (num_moves == 0) break;
    std::vector<Move> moves(num_moves);
    std::vector<uint64_t> n(num_moves);
    node_view->FetchEdgeData({.moves = moves, .n = n});
    const size_t best_idx = std::max_element(n.begin(), n.end()) - n.begin();
    const Move best_move = moves[best_idx];
    pv.push_back(best_move);
    current_position = Position(current_position, best_move);
    current_hash = NodeHash{HashCat(current_hash.hash, best_move.raw_data())};
  }

  return pv;
}
}  // namespace lc3
}  // namespace lczero
