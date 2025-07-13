#include "search/lc3/workers/watchdog_worker.h"

#include <absl/algorithm/container.h>

#include <algorithm>
#include <string>
#include <vector>

#include "chess/callbacks.h"
#include "chess/position.h"

namespace lczero {
namespace lc3 {

// Node, it's number of visits, and legal moves from this node.
struct WatchdogWorker::HashAndPosition {
  NodeKey key;
  Position position;  // Unused, but in future may be needed for NodeKey.
  size_t n = 0;
  std::vector<Move> moves = {};
};

void WatchdogWorker::Stop(bool must_respond_bestmove) {
  must_respond_bestmove_.store(must_respond_bestmove,
                               std::memory_order_relaxed);
  must_exit_.Notify();
}

void WatchdogWorker::Run() {
  while (true) {
    if (CheckOnce()) return;  // Return if responded bestmove.
    if (must_exit_.WaitForNotificationWithTimeout(absl::Milliseconds(10)) &&
        !must_respond_bestmove_.load()) {
      return;
    }
  }
}

bool WatchdogWorker::CheckOnce() {
  const Variation& head = *env_.head;
  // Fetch the head position.
  std::optional<HashAndPosition> position =
      FetchPosition(head->key, head->position);
  if (!position) return false;
  const int64_t nodes = position->n;

  auto pv = BuildPV(position);
  if (pv.empty()) return false;

  // TODO remove that flipping logic once we have Move from white perspective.
  const bool head_is_black = (*env_.head)->position.IsBlackToMove();
  for (size_t i = 0; i < pv.size(); ++i) {
    if (head_is_black == (i % 2 == 0)) pv[i].Flip();
  }

  const bool will_respond_bestmove =
      must_respond_bestmove_.load(std::memory_order_relaxed);

  const auto now = Clock::now();
  using Duration = std::chrono::duration<double>;
  const Duration time_since_last_check = now - last_info_print_;

  if (will_respond_bestmove ||
      (pv != previous_pv_ || time_since_last_check > std::chrono::seconds(5))) {
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
    last_info_print_ = std::chrono::steady_clock::now();
  }
  if (will_respond_bestmove) {
    BestMoveInfo best_move(previous_pv_[0],
                           previous_pv_.size() > 1 ? previous_pv_[1] : Move{});
    env_.uci_responder->OutputBestMove(&best_move);
    return true;
  }
  return false;
}

// The function builds the PV from the given position by picking the most
// visited child node (not edge). As it's only allowed to hold one node of the
// NodeRepository at a time, it all the children nodes one by one, then picks
// the one with the most visits and continues until it reaches a node with no
// children.
std::vector<Move> WatchdogWorker::BuildPV(
    std::optional<HashAndPosition> position) const {
  std::vector<Move> pv;

  while (position && !position->moves.empty()) {
    // Fetch all children nodes into `candidates`.
    std::vector<std::optional<HashAndPosition>> candidates;
    for (const Move& move : position->moves) {
      Position next_position = Position(position->position, move);
      NodeKey next_hash =
          Policy::MakeNodeKey(position->key, move, next_position);
      candidates.push_back(FetchPosition(next_hash, next_position));
    }
    // Pick the one with the most svisits.
    size_t best_idx =
        absl::c_max_element(candidates,
                            [](const std::optional<HashAndPosition>& a,
                               const std::optional<HashAndPosition>& b) {
                              if (!a) return true;
                              if (!b) return false;
                              return a->n < b->n;
                            }) -
        candidates.begin();
    pv.push_back(position->moves[best_idx]);

    // And make it the current position.
    position = std::move(candidates[best_idx]);
  }
  return pv;
}

// Fetches the position for the given key, temporarily locking the node.
// Only fetches moves that have any visits.
auto WatchdogWorker::FetchPosition(const NodeKey& key,
                                   const Position& position) const
    -> std::optional<HashAndPosition> {
  NodeHandle node_handle =
      env_.node_repository->GetNodeForUpdate(key,
                                             /*create_if_missing=*/false);
  if (!node_handle) return std::nullopt;
  size_t num_moves = node_handle.FetchMoveCounts().with_visits;
  HashAndPosition current{
      .key = key,
      .position = position,
      .n = node_handle.GetNodeAggregates().n,
      .moves = std::vector<Move>(num_moves),
  };
  node_handle.FetchEdges(
      NodeHandle::EdgeDataDestination{.moves = current.moves});
  return current;
}

}  // namespace lc3
}  // namespace lczero
