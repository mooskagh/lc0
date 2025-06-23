#include "search/lc3/settings.h"

#include "utils/optionsdict.h"

namespace lczero {
namespace lc3 {

namespace {

const OptionId gNumGatherThreads(
    "gather-threads", "GatherThreads",
    "Number of threads to use for gathering MCTS nodes.");
const OptionId gNumEvalThreads(
    "eval-threads", "EvalThreads",
    "Number of threads to use for evaluating MCTS nodes.");
const OptionId gNumBackpropThreads(
    "backprop-threads", "BackpropThreads",
    "Number of threads to use for backpropagating MCTS nodes.");

}  // namespace

void Settings::Populate(OptionsParser* options) {
  options->Add<IntOption>(gNumGatherThreads, 1, 128) = 2;
  options->Add<IntOption>(gNumEvalThreads, 1, 128) = 2;
  options->Add<IntOption>(gNumBackpropThreads, 1, 128) = 2;
}

}  // namespace lc3
}  // namespace lczero