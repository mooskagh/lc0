#pragma once

#include <vector>

#include "search/lc3/metrics/search_metrics.h"
#include "src/utils/metrics/exponential_aggregator.h"

namespace lczero {
namespace lc3 {

// TODO note to self: this is a stats keeper class
class GameStats {
 public:
  void NewGame();
  void NewSearchSession();

  ExponentialAggregator<SearchMetrics>& live() { return live_stats_; }

  template <typename T>
  void Feed(T&& stat) {
    live_stats_.RecordMetrics(std::forward<T>(stat));
  }

 private:
  struct MoveStats {
    SearchMetrics metrics;
    float duration_seconds = 0.0f;
  };

  ExponentialAggregator<SearchMetrics> live_stats_;
  // Stats at the end of previous search sessions.
  std::vector<MoveStats> previous_search_session_stats_;
};

}  // namespace lc3
}  // namespace lczero