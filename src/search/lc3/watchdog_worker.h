#pragma once

#include <chrono>
#include <vector>

#include "absl/synchronization/notification.h"
#include "chess/callbacks.h"
#include "chess/types.h"
#include "search/lc3/positions.h"

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
  bool CheckOnce();
  std::vector<Move> BuildPV() const;

  WatchdogWorkerEnvironment env_;
  absl::Notification must_exit_;
  std::atomic<bool> must_respond_bestmove_{false};

  std::vector<Move> previous_pv_;
  std::chrono::steady_clock::time_point last_check_time_;

  std::chrono::steady_clock::time_point prev_nps_check_time_ =
      std::chrono::steady_clock::now();
  int64_t prev_nps_check_nodes_ = 0;
  std::chrono::steady_clock::time_point current_nps_check_time_ =
      std::chrono::steady_clock::now();
  int64_t current_nps_check_nodes_ = 0;
};

}  // namespace lc3
}  // namespace lczero