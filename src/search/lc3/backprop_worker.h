#pragma once

#include <cstddef>
#include <vector>

#include "search/lc3/channels.h"
#include "search/lc3/context.h"

namespace lczero {
namespace lc3 {

class BackpropWorker {
 public:
  BackpropWorker(const Context& context, BackpropWorkerChannels channels)
      : ctx_(context), channels_(std::move(channels)) {}

  void OneStep();

 private:
  struct BackPropItem;

  std::vector<BackPropItem> FetchEvalResults();
  static BackPropItem EvalItemToBackpropItem(EvalItem* item, size_t num_visits);

  Context ctx_;
  BackpropWorkerChannels channels_;
};

};  // namespace lc3
}  // namespace lczero