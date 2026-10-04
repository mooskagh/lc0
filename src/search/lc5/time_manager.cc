#include "search/lc5/time_manager.h"

#include <algorithm>

#include "chess/uciloop.h"

namespace lczero::lc5 {
namespace {

std::optional<int64_t> GetTimeBudget(TimeManager::Config config,
                                     const GoParams& params,
                                     bool black_to_move) {
  if (params.ponder) return std::nullopt;
  if (params.movetime) return params.movetime;
  if (params.infinite) return std::nullopt;
  const auto& remaining = black_to_move ? params.btime : params.wtime;
  if (!remaining) return std::nullopt;
  if (*remaining <= config.move_overhead_ms) return 0;

  const int64_t usable = *remaining - config.move_overhead_ms;
  const int64_t increment = std::max<int64_t>(
      0, (black_to_move ? params.binc : params.winc).value_or(0));
  long double fraction = config.alphazero_time_pct / 100.0L;
  if (params.movestogo && *params.movestogo > 0) {
    fraction = std::max(fraction, 1.0L / *params.movestogo);
  }
  // Divide the clock and future increments over the implied move horizon.
  // The current move's increment is only received after we finish searching.
  const long double budget = usable * fraction + increment * (1 - fraction);
  return static_cast<int64_t>(
      std::clamp(budget, 1.0L, static_cast<long double>(usable)));
}

}  // namespace

TimeManager::TimeManager(Config config, const GoParams& params,
                         bool black_to_move,
                         std::chrono::steady_clock::time_point start_time) {
  if (const auto budget = GetTimeBudget(config, params, black_to_move)) {
    deadline_ = start_time + std::chrono::milliseconds(*budget);
  }
}

TimeManager::Decision TimeManager::Evaluate(
    std::chrono::steady_clock::time_point now) const {
  if (deadline_ && now >= *deadline_) {
    return {.should_stop = true, .next_check = std::nullopt};
  }
  return {.should_stop = false, .next_check = deadline_};
}

}  // namespace lczero::lc5
