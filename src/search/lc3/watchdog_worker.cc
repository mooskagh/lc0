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

void WatchdogWorker::Run() {
  while (true) {
    CheckOnce();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void WatchdogWorker::CheckOnce() {
  NodeHandle node_handle =
      ctx_.node_repository->GetNodeForUpdate((*ctx_.head)->key,
                                             /*create_if_missing=*/false);
  if (!node_handle) return;
  const int64_t nodes = node_handle.GetNodeAggregates().n;
  node_handle.Release();

  auto pv = BuildPV();
  const bool head_is_black = (*ctx_.head)->position.IsBlackToMove();
  for (size_t i = 0; i < pv.size(); ++i) {
    if (head_is_black == (i % 2 == 0)) pv[i].Flip();
  }

  const auto now = std::chrono::steady_clock::now();
  if (pv != previous_pv_ || last_check_time_ + std::chrono::seconds(5) < now) {
    if (current_nps_check_time_ + std::chrono::seconds(1) < now) {
      prev_nps_check_time_ = current_nps_check_time_;
      prev_nps_check_nodes_ = current_nps_check_nodes_;
      current_nps_check_time_ = now;
      current_nps_check_nodes_ = nodes;
    }

    const auto num_seconds_since_prev_check =
        std::chrono::duration_cast<std::chrono::microseconds>(
            now - prev_nps_check_time_)
            .count() /
        1000000.0;
    const int nps = num_seconds_since_prev_check > 0
                        ? static_cast<int>((nodes - prev_nps_check_nodes_) /
                                           num_seconds_since_prev_check)
                        : 0;
    std::vector<ThinkingInfo> infos = {
        {.nodes = static_cast<int64_t>(nodes), .nps = nps, .pv = pv}};
    ctx_.uci_responder->OutputThinkingInfo(&infos);
    previous_pv_ = std::move(pv);
    last_check_time_ = std::chrono::steady_clock::now();
  }
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
    node_handle.Release();
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
