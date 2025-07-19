#pragma once

#include <chrono>
#include <vector>

#include "absl/synchronization/notification.h"
#include "chess/callbacks.h"
#include "chess/types.h"
#include "search/lc3/node_repository/node_repository.h"
#include "search/lc3/search_policy/search_policy.h"
#include "search/lc3/workers/variation.h"

namespace lczero {
namespace lc3 {

struct WatchdogWorkerEnvironment {
  NodeRepository* node_repository;
  Variation* head;
  UciResponder* uci_responder;
};

class WatchdogWorker {
 public:
  WatchdogWorker(WatchdogWorkerEnvironment env) : env_(std::move(env)) {}

  // Runs in a separate thread.
  void Run();

  void Stop(bool must_respond_bestmove);

 private:
  using Policy = SearchPolicy;
  struct HashAndPosition;

  bool CheckOnce();
  std::vector<Move> BuildPV(std::optional<HashAndPosition>) const;
  std::optional<HashAndPosition> FetchPosition(const NodeKey& key,
                                               const Position& position) const;
  bool MustExit() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(must_exit_mutex_) {
    return must_exit_;
  }

  WatchdogWorkerEnvironment env_;
  absl::Mutex must_exit_mutex_;
  bool must_exit_{false} ABSL_GUARDED_BY(must_exit_mutex_);
  std::atomic<bool> must_respond_bestmove_{false};

  std::vector<Move> previous_pv_;
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;
  TimePoint last_info_print_;

  TimePoint prev_nps_check_time_ = Clock::now();
  int64_t prev_nps_check_nodes_ = 0;
  TimePoint current_nps_check_time_ = Clock::now();
  int64_t current_nps_check_nodes_ = 0;
};

}  // namespace lc3
}  // namespace lczero