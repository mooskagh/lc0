#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "search/lc3/channels.h"
#include "search/lc3/context.h"

namespace lczero {
namespace lc3 {

class BackpropWorker {
 public:
  BackpropWorker(const Context& context, BackpropWorkerChannels channels)
      : ctx_(context), channels_(std::move(channels)) {}

  void Run();

 private:
  void OneStep();
  struct BackPropItem;

  std::vector<BackPropItem> FetchBackpropTasks();
  std::pair<std::optional<BackpropWorker::BackPropItem>, bool>
  ProcessSingleBackpropTask(EvalItem* item);
  static BackPropItem EvalItemToBackpropItem(EvalItem* item, size_t num_visits);

  Context ctx_;
  BackpropWorkerChannels channels_;
};

};  // namespace lc3
}  // namespace lczero