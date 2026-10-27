/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

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
