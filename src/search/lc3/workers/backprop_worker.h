#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "search/lc3/node_repository/node_repository.h"
#include "search/lc3/search_policy/search_policy.h"
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
  using Policy = SearchPolicy;
  bool OneStep();
  struct NodeUpdate;

  std::vector<NodeUpdate> FetchBackpropTasks();
  static NodeUpdate CollectSameVariationUpdates(
      std::vector<NodeUpdate>& backprop_heap);
  void UpdateLeafNode(NodeEvent* event,
                      const NodeHandle::NodeAggregates& node_value);
  void DisposeNodeEvent(NodeEvent* event);
  NodeHandle::NodeAggregates UpdateNodeInRepository(
  const NodeUpdate& update);

  BackpropWorkerEnvironment env_;
};

}  // namespace lc3
}  // namespace lczero