#include "search/lc3/watchdog_worker.h"

#include <algorithm>
#include <string>
#include <vector>

#include "chess/callbacks.h"
#include "chess/position.h"
#include "search/lc3/debug.h"
#include "utils/hashcat.h"

namespace lczero {
namespace lc3 {

void WatchdogWorker::CheckOnce() {
  auto lock = ctx_.node_repository->GetAccessLock();
  auto root_view = lock.FetchReadOnly((*ctx_.head)->hash);
  if (!root_view) return;

  for (const auto& debug_line :
       DebugNodeDataFromStorage(*root_view).ToStrings()) {
    CERR << debug_line;
  }

  auto pv = BuildPV();
  const bool head_is_black = (*ctx_.head)->position.IsBlackToMove();
  for (size_t i = 0; i < pv.size(); ++i) {
    if (head_is_black == (i % 2 == 0)) pv[i].Flip();
  }

  std::vector<ThinkingInfo> infos = {
      {.nodes = static_cast<int64_t>(root_view->GetN()), .pv = std::move(pv)}};
  ctx_.uci_responder->OutputThinkingInfo(&infos);
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
