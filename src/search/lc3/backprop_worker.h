#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "search/lc3/channels.h"

namespace lczero {
namespace lc3 {

struct BackpropWorkerEnvironment {
  EvalItemReceiver* const backprop_receiver;
  NodeRepository* node_repository;
  EvalItemPool* eval_item_pool;
};

class BackpropWorker {
 public:
  BackpropWorker(BackpropWorkerEnvironment env) : env_(std::move(env)) {}

  void Run();

 private:
  void OneStep();
  struct BackPropItem;

  std::vector<BackPropItem> FetchBackpropTasks();
  std::pair<std::optional<BackpropWorker::BackPropItem>, bool>
  ProcessSingleBackpropTask(EvalItem* item);
  static BackPropItem EvalItemToBackpropItem(EvalItem* item, size_t num_visits);
  void DisposeEvalItem(EvalItem* item);

  BackpropWorkerEnvironment env_;
};

};  // namespace lc3
}  // namespace lczero