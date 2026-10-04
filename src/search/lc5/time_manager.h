#pragma once

#include <chrono>
#include <cstdint>
#include <optional>

namespace lczero {

struct GoParams;

namespace lc5 {

class TimeManager {
 public:
  struct Config {
    int64_t move_overhead_ms;
    float alphazero_time_pct;
  };
  struct Decision {
    bool should_stop;
    std::optional<std::chrono::steady_clock::time_point> next_check;
  };

  TimeManager(Config config, const GoParams& params, bool black_to_move,
              std::chrono::steady_clock::time_point start_time);
  Decision Evaluate(std::chrono::steady_clock::time_point now) const;

 private:
  std::optional<std::chrono::steady_clock::time_point> deadline_;
};

}  // namespace lc5
}  // namespace lczero
