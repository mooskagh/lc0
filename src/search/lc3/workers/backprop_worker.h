#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "search/lc3/node_repository/node_repository.h"
#include "search/lc3/workers/node_event_queue.h"

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
  struct CombinedBackPropItem;

  std::vector<BackPropItem> FetchBackpropTasks();
  static CombinedBackPropItem CollectSameVariationUpdates(
      std::vector<BackPropItem>& backprop_heap);
  static BackPropItem NodeEventToBackpropItem(NodeEvent* event,
                                              size_t num_visits);
  void UpdateLeafNode(NodeEvent* event, size_t num_visits_to_apply);
  void DisposeNodeEvent(NodeEvent* event);

  BackpropWorkerEnvironment env_;
};

}  // namespace lc3
}  // namespace lczero