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
