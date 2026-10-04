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
  Resolved Resolve(const BackendAttributes& backend) const;

  int threads() const { return threads_; }
  int eval_threads() const { return eval_threads_; }
  int minibatch_size() const { return minibatch_size_; }
  int max_active_visits() const { return max_active_visits_; }
  int history_key_length() const { return history_key_length_; }
  const TimeManager::Config& time_management() const {
    return time_management_;
  }

 private:
  int threads_;
  int eval_threads_;
  int minibatch_size_;
  int max_active_visits_;
  int history_key_length_;
  int max_batch_delay_ms_;
  float cpuct_;
  float cpuct_base_;
  float cpuct_factor_;
  FpuStrategy fpu_strategy_;
  float fpu_value_;
  TimeManager::Config time_management_;
};

}  // namespace lczero::lc5
