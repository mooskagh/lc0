#pragma once

#include <span>

#include "neural/backend.h"
#include "search/lc3/metrics/nodes_metric.h"
#include "search/lc3/workers/node_event_queue.h"

namespace lczero {
namespace lc3 {

struct WorkTreeNode;
class GameStats;

struct EvalWorkerEnvironment {
  NodeEventReceiver* const eval_receiver;
  NodeEventSender backprop_sender;
  absl::Mutex* const gather_worker_unblocker;
  Backend* const backend;
  GameStats* stats;
};

class EvalWorker {
 public:
  EvalWorker(EvalWorkerEnvironment env) : env_(std::move(env)) {}

  void Run();

 private:
  bool OneStep();
  void SendCompletedNodeEvent(NodeEvent*);
  void SendCompletedBatchItems();

  bool Collect();
  void EnqueueIncomingEvents(std::span<NodeEvent*> events);
  void EnqueueIncomingEvent(NodeEvent* event);

  EvalWorkerEnvironment env_;

  std::unique_ptr<BackendComputation> computation_;
  std::vector<NodeEvent*> batched_node_events_;
  EvalNodesMetrics nodes_metrics_;
};

}  // namespace lc3
}  // namespace lczero