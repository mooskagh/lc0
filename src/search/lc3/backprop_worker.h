#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "search/lc3/channels.h"

namespace lczero {
namespace lc3 {

struct BackpropWorkerEnvironment {
  NodeEventReceiver* const backprop_receiver;
  NodeRepository* node_repository;
  NodeEventPool* eval_item_pool;
};

class BackpropWorker {
 public:
  BackpropWorker(BackpropWorkerEnvironment env) : env_(std::move(env)) {}

  void Run();

 private:
  bool OneStep();
  struct BackPropItem;

  std::optional<std::vector<BackPropItem>> FetchBackpropTasks();
  std::pair<std::optional<BackpropWorker::BackPropItem>, bool> HandleNodeEvent(
      NodeEvent* event);
  static BackPropItem NodeEventToBackpropItem(NodeEvent* event,
                                              size_t num_visits);
  void DisposeNodeEvent(NodeEvent* event);

  BackpropWorkerEnvironment env_;
};

}  // namespace lc3
}  // namespace lczero