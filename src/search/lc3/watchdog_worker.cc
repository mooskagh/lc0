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
    if (env_.can_exit->WaitForNotificationWithTimeout(absl::Milliseconds(10))) {
      break;
    }
  }
}

void WatchdogWorker::CheckOnce() {
  NodeHandle node_handle =
      env_.node_repository->GetNodeForUpdate((*env_.head)->key,
                                             /*create_if_missing=*/false);
  if (!node_handle) return;
  const int64_t nodes = node_handle.GetNodeAggregates().n;
  node_handle.Release();

  auto pv = BuildPV();
  const bool head_is_black = (*env_.head)->position.IsBlackToMove();
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
    env_.uci_responder->OutputThinkingInfo(&infos);
    previous_pv_ = std::move(pv);
    last_check_time_ = std::chrono::steady_clock::now();
  }
}

std::vector<Move> WatchdogWorker::BuildPV() const {
  std::vector<Move> pv;
  struct HashAndPosition {
    NodeKey hash;
    std::vector<Move> moves = {};
    size_t n = 0;
  };

  auto fetch_position = [&](const NodeKey& key) -> HashAndPosition {
    NodeHandle node_handle =
        env_.node_repository->GetNodeForUpdate(key,
                                               /*create_if_missing=*/false);
    if (!node_handle) return HashAndPosition{};
    size_t num_moves = node_handle.FetchMoveCounts().with_visits;
    HashAndPosition current{
        .hash = key,
        .moves = std::vector<Move>(num_moves),
        .n = node_handle.GetNodeAggregates().n,
    };
    node_handle.FetchEdges(
        NodeHandle::EdgeDataDestination{.moves = current.moves});
    return current;
  };

  std::optional<HashAndPosition> current_position =
      fetch_position((*env_.head)->key);

  while (current_position && !current_position->moves.empty()) {
    std::vector<HashAndPosition> candidates;
    for (const Move& move : current_position->moves) {
      NodeKey next_hash{HashCat(current_position->hash.hash, move.raw_data())};
      candidates.push_back(fetch_position(next_hash));
    }
    size_t best_idx =
        std::max_element(candidates.begin(), candidates.end(),
                         [](const HashAndPosition& a,
                            const HashAndPosition& b) { return a.n < b.n; }) -
        candidates.begin();
    pv.push_back(current_position->moves[best_idx]);
    current_position = std::move(candidates[best_idx]);
  }
  return pv;
}
}  // namespace lc3
}  // namespace lczero
