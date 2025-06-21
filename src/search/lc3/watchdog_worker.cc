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
  // PrintNodeTree(&lock, std::cerr, (*ctx_.head)->position,
  // (*ctx_.head)->hash);
  NodeHandle node_handle =
      ctx_.node_repository->GetNodeForUpdate((*ctx_.head)->key,
                                             /*create_if_missing=*/false);
  if (!node_handle) return;

  // for (const auto& debug_line :
  //      DebugNodeDataFromStorage(*root_view).ToStrings()) {
  //   CERR << debug_line;
  // }

  auto pv = BuildPV();
  const bool head_is_black = (*ctx_.head)->position.IsBlackToMove();
  for (size_t i = 0; i < pv.size(); ++i) {
    if (head_is_black == (i % 2 == 0)) pv[i].Flip();
  }

  std::vector<ThinkingInfo> infos = {
      {.nodes = static_cast<int64_t>(node_handle.GetNodeAggregates().n),
       .pv = std::move(pv)}};
  ctx_.uci_responder->OutputThinkingInfo(&infos);
}

std::vector<Move> WatchdogWorker::BuildPV() const {
  std::vector<Move> pv;
  NodeKey current_hash = (*ctx_.head)->key;
  Position current_position = (*ctx_.head)->position;

  while (NodeHandle node_handle = ctx_.node_repository->GetNodeForUpdate(
             current_hash,
             /*create_if_missing=*/false)) {
    const size_t num_moves = node_handle.FetchMoveCounts().with_visits;
    if (num_moves == 0) break;
    std::vector<Move> moves(num_moves);
    std::vector<uint64_t> n(num_moves);
    node_handle.FetchEdges({.moves = moves, .n = n});
    const size_t best_idx = std::max_element(n.begin(), n.end()) - n.begin();
    const Move best_move = moves[best_idx];
    pv.push_back(best_move);
    current_position = Position(current_position, best_move);
    current_hash = NodeKey{HashCat(current_hash.hash, best_move.raw_data())};
  }

  return pv;
}
}  // namespace lc3
}  // namespace lczero
