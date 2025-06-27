#pragma once

#include <chrono>
#include <vector>

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

  void Run();

 private:
  void CheckOnce();
  std::vector<Move> BuildPV() const;

  WatchdogWorkerEnvironment env_;

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