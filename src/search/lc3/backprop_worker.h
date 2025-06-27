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
  BackpropWorker(const Context& context, EvalItemReceiver* backprop_receiver)
      : ctx_(context), backprop_receiver_(backprop_receiver) {}

  void Run();

 private:
  void OneStep();
  struct BackPropItem;

  std::vector<BackPropItem> FetchBackpropTasks();
  std::pair<std::optional<BackpropWorker::BackPropItem>, bool>
  ProcessSingleBackpropTask(EvalItem* item);
  static BackPropItem EvalItemToBackpropItem(EvalItem* item, size_t num_visits);

  Context ctx_;
  EvalItemReceiver* const backprop_receiver_;
};

};  // namespace lc3
}  // namespace lczero