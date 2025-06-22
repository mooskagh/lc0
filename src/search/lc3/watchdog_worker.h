#include "search/lc3/context.h"

#pragma once

namespace lczero {
namespace lc3 {

class WatchdogWorker {
 public:
  WatchdogWorker(const Context& context) : ctx_(context) {}

  void CheckOnce();
  std::vector<Move> BuildPV() const;

 private:
  const Context ctx_;
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