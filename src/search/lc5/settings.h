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

#include "neural/backend.h"
#include "search/lc5/time_manager.h"
#include "utils/optionsdict.h"
#include "utils/optionsparser.h"

namespace lczero::lc5 {

enum class FpuStrategy { kAbsolute, kReduction };

class Settings {
 public:
  struct Resolved {
    int threads;
    int eval_threads;
    int minibatch_size;
    int max_active_visits;
    int history_key_length;
    int max_batch_delay_ms;
    float cpuct;
    float cpuct_base;
    float cpuct_factor;
    FpuStrategy fpu_strategy;
    float fpu_value;
    bool restart_waiters;
  };

  explicit Settings(const OptionsDict& options);
  static void Populate(OptionsParser* options);
  // Fills automatic (zero) sizes from the backend and validates them.
  Resolved Resolve(const BackendAttributes& backend) const;

  int history_key_length() const { return configured_.history_key_length; }
  const TimeManager::Config& time_management() const {
    return time_management_;
  }

 private:
  Resolved configured_;  // As set by the user; zero sizes are automatic.
  TimeManager::Config time_management_;
};

}  // namespace lczero::lc5
