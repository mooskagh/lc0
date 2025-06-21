#pragma once

#include <cstddef>
#include <vector>

#include "search/lc3/context.h"

namespace lczero {
namespace lc3 {

class BackpropWorker {
 public:
  BackpropWorker(const Context& context, size_t backprop_task_idx)
      : ctx_(context), backprop_task_idx_(backprop_task_idx) {}

  void OneStep();

 private:
  struct BackPropItem;

  std::vector<BackPropItem> FetchEvalResults();
  static BackPropItem EvalItemToBackpropItem(EvalItem* item, size_t num_visits);

  Context ctx_;
  const size_t backprop_task_idx_;
};

};  // namespace lc3
}  // namespace lczero