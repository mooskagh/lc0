#pragma once

#include <span>

#include "neural/backend.h"
#include "search/lc3/workers/node_event_queue.h"

namespace lczero {
namespace lc3 {

struct WorkTreeNode;

struct EvalWorkerEnvironment {
  NodeEventReceiver* const eval_receiver;
  NodeEventSender backprop_sender;
  absl::Mutex* const gather_worker_unblocker;
  Backend* const backend;
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
};

}  // namespace lc3
}  // namespace lczero