#include "search/lc3/metrics/game_stats.h"

namespace lczero {
namespace lc3 {

void GameStats::NewGame() {
  previous_move_stats_.clear();
  live_stats_.Reset();
}

}  // namespace lc3
}  // namespace lczero
