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

#include "search/lc5/settings.h"

#include <algorithm>
#include <thread>

#include "search/classic/stoppers/factory.h"
#include "utils/exception.h"

namespace lczero::lc5 {
namespace {

const OptionId kThreads{"threads", "Threads", "Lc5 visit worker threads."};
const OptionId kEvalThreads{"eval-threads", "EvalThreads",
                            "Lc5 concurrent backend evaluators."};
const OptionId kMinibatchSize{"minibatch-size", "MinibatchSize",
                              "Lc5 neural evaluation batch target."};
const OptionId kMaxActiveVisits{"max-active-visits", "MaxActiveVisits",
                                "Hard bound on live Lc5 logical visits."};
const OptionId kHistoryKeyLength{
    "history-key-length", "HistoryKeyLength",
    "History positions beyond current used in Lc5 keys; shorter keys merge "
    "more histories and reuse the first canonical expansion."};
const OptionId kMaxBatchDelay{"max-batch-delay-ms", "MaxBatchDelayMs",
                              "Maximum Lc5 partial-batch delay in ms."};
const OptionId kCpuct{"cpuct", "CPuct", "Lc5 PUCT initial constant."};
const OptionId kCpuctBase{"cpuct-base", "CPuctBase", "Lc5 PUCT growth base."};
const OptionId kCpuctFactor{"cpuct-factor", "CPuctFactor",
                            "Lc5 PUCT growth multiplier."};
const OptionId kFpuStrategy{"fpu-strategy", "FpuStrategy",
                            "Lc5 first-play urgency strategy."};
const OptionId kFpuValue{"fpu-value", "FpuValue",
                         "Lc5 absolute FPU value or FPU reduction."};
const OptionId kRestartWaiters{
    "restart-waiters", "RestartWaiters",
    "Visits that waited for a node's evaluation select again from the root "
    "instead of continuing below that node."};
const OptionId kActiveVisitsRatio{
    "active-visits-ratio", "ActiveVisitsRatio",
    "Live visits are limited to 16 plus this fraction of the root's visits "
    "(and to MaxActiveVisits); 0 disables the limit."};

}  // namespace

void Settings::Populate(OptionsParser* options) {
  options->Add<IntOption>(kThreads, 0, 512) = 0;
  options->Add<IntOption>(kEvalThreads, 0, 128) = 0;
  // Backend maximum is only known when a run starts; Resolve validates it.
  options->Add<IntOption>(kMinibatchSize, 0, 1000000) = 0;
  options->Add<IntOption>(kMaxActiveVisits, 0, 1000000) = 0;
  options->Add<IntOption>(kHistoryKeyLength, 0, 7) = 7;
  options->Add<IntOption>(kMaxBatchDelay, 0, 100) = 2;
  options->Add<FloatOption>(kCpuct, 0.0f, 100.0f) = 1.745f;
  options->Add<FloatOption>(kCpuctBase, 1.0f, 1.0e9f) = 38739.0f;
  options->Add<FloatOption>(kCpuctFactor, 0.0f, 1000.0f) = 3.894f;
  options->Add<ChoiceOption>(
      kFpuStrategy, std::vector<std::string>{"absolute", "reduction"}) =
      "reduction";
  options->Add<FloatOption>(kFpuValue, -100.0f, 100.0f) = 0.33f;
  options->Add<BoolOption>(kRestartWaiters) = false;
  options->Add<FloatOption>(kActiveVisitsRatio, 0.0f, 100.0f) = 0.25f;
  // Time management and stoppers are classic's, with the same options.
  classic::PopulateTimeManagementOptions(classic::RunType::kUci, options);
}

Settings::Settings(const OptionsDict& options)
    : configured_{
          .threads = options.Get<int>(kThreads),
          .eval_threads = options.Get<int>(kEvalThreads),
          .minibatch_size = options.Get<int>(kMinibatchSize),
          .max_active_visits = options.Get<int>(kMaxActiveVisits),
          .history_key_length = options.Get<int>(kHistoryKeyLength),
          .max_batch_delay_ms = options.Get<int>(kMaxBatchDelay),
          .cpuct = options.Get<float>(kCpuct),
          .cpuct_base = options.Get<float>(kCpuctBase),
          .cpuct_factor = options.Get<float>(kCpuctFactor),
          .fpu_strategy = options.Get<std::string>(kFpuStrategy) == "absolute"
                              ? FpuStrategy::kAbsolute
                              : FpuStrategy::kReduction,
          .fpu_value = options.Get<float>(kFpuValue),
          .restart_waiters = options.Get<bool>(kRestartWaiters),
          .active_visits_ratio = options.Get<float>(kActiveVisitsRatio)} {}

Settings::Resolved Settings::Resolve(const BackendAttributes& backend) const {
  Resolved resolved = configured_;
  const int backend_max = std::max(1, backend.maximum_batch_size);
  if (resolved.minibatch_size > backend_max) {
    throw Exception("Lc5 MinibatchSize exceeds backend maximum batch size");
  }
  if (resolved.eval_threads == 0) {
    resolved.eval_threads =
        std::clamp(backend.suggested_num_search_threads, 1, 128);
  }
  if (resolved.minibatch_size == 0) {
    resolved.minibatch_size =
        std::clamp(backend.recommended_batch_size, 1, backend_max);
  }
  if (resolved.threads == 0) {
    const int hardware = std::max(1u, std::thread::hardware_concurrency());
    resolved.threads =
        backend.runs_on_cpu
            ? std::max(1, backend.suggested_num_search_threads)
            : std::min(hardware, std::max(2, 4 * resolved.eval_threads));
  }
  if (resolved.max_active_visits == 0) {
    // Visits in flight dilute small searches, so keep only enough to fill
    // the evaluators.
    resolved.max_active_visits =
        std::max(256, 3 * resolved.eval_threads * resolved.minibatch_size / 2);
  }
  if (resolved.max_active_visits < resolved.eval_threads) {
    throw Exception(
        "Lc5 MaxActiveVisits must allow at least one visit per evaluator");
  }
  return resolved;
}

}  // namespace lczero::lc5
