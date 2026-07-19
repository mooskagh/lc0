#include "search/lc5/settings.h"

#include <algorithm>
#include <thread>

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
const OptionId kMoveOverhead{"move-overhead", "MoveOverheadMs",
                             "Time reserved for communication overhead."};
const OptionId kAlphazeroTimePct{
    "alphazero-time-pct", "AlphaZeroTimePct",
    "Percentage of the remaining clock allocated to each Lc5 move."};

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
  options->Add<ChoiceOption>(kFpuStrategy,
                             std::vector<std::string>{"absolute", "reduction"}) =
      "reduction";
  options->Add<FloatOption>(kFpuValue, -100.0f, 100.0f) = 0.33f;
  options->Add<IntOption>(kMoveOverhead, 0, 100000000) = 200;
  options->Add<FloatOption>(kAlphazeroTimePct, 0.0f, 100.0f) = 12.0f;
}

Settings::Settings(const OptionsDict& options)
    : threads_(options.Get<int>(kThreads)),
      eval_threads_(options.Get<int>(kEvalThreads)),
      minibatch_size_(options.Get<int>(kMinibatchSize)),
      max_active_visits_(options.Get<int>(kMaxActiveVisits)),
      history_key_length_(options.Get<int>(kHistoryKeyLength)),
      max_batch_delay_ms_(options.Get<int>(kMaxBatchDelay)),
      cpuct_(options.Get<float>(kCpuct)),
      cpuct_base_(options.Get<float>(kCpuctBase)),
      cpuct_factor_(options.Get<float>(kCpuctFactor)),
      fpu_strategy_(options.Get<std::string>(kFpuStrategy) == "absolute"
                        ? FpuStrategy::kAbsolute
                        : FpuStrategy::kReduction),
      fpu_value_(options.Get<float>(kFpuValue)),
      move_overhead_ms_(options.Get<int>(kMoveOverhead)),
      alphazero_time_pct_(options.Get<float>(kAlphazeroTimePct)) {}

std::optional<int64_t> Settings::GetTimeBudget(const GoParams& params,
                                               bool black_to_move) const {
  if (params.infinite || params.ponder) return std::nullopt;
  const auto& remaining = black_to_move ? params.btime : params.wtime;
  if (!remaining) return std::nullopt;
  if (*remaining <= move_overhead_ms_) return 0;
  return static_cast<int64_t>((*remaining - move_overhead_ms_) *
                              (alphazero_time_pct_ / 100.0));
}

Settings::Resolved Settings::Resolve(const BackendAttributes& backend) const {
  const int automatic_evaluators =
      std::clamp(backend.suggested_num_search_threads, 1, 128);
  const int evaluators = eval_threads_ == 0 ? automatic_evaluators : eval_threads_;
  const int backend_max = std::max(1, backend.maximum_batch_size);
  const int recommended = std::clamp(backend.recommended_batch_size, 1, backend_max);
  if (minibatch_size_ > backend_max) {
    throw Exception("Lc5 MinibatchSize exceeds backend maximum batch size");
  }
  const int batch = minibatch_size_ == 0 ? recommended : minibatch_size_;
  const unsigned hardware = std::max(1u, std::thread::hardware_concurrency());
  const int automatic_workers =
      backend.runs_on_cpu
          ? std::max(1, backend.suggested_num_search_threads)
          : std::min<int>(hardware, std::max(2, 4 * evaluators));
  const int workers = threads_ == 0 ? automatic_workers : threads_;
  const int active = max_active_visits_ == 0
                         ? std::max(1024, 4 * evaluators * batch)
                         : max_active_visits_;
  if (active < evaluators) {
    throw Exception("Lc5 MaxActiveVisits must allow at least one visit per evaluator");
  }
  return {.threads = workers,
          .eval_threads = evaluators,
          .minibatch_size = batch,
          .max_active_visits = active,
          .history_key_length = history_key_length_,
          .max_batch_delay_ms = max_batch_delay_ms_,
          .cpuct = cpuct_,
          .cpuct_base = cpuct_base_,
          .cpuct_factor = cpuct_factor_,
          .fpu_strategy = fpu_strategy_,
          .fpu_value = fpu_value_};
}

}  // namespace lczero::lc5
