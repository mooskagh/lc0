#pragma once

#include <memory>
#include <thread>
#include <vector>

#include "absl/synchronization/notification.h"
#include "search/lc3/backprop_worker.h"
#include "search/lc3/channels.h"
#include "search/lc3/eval_worker.h"
#include "search/lc3/gather_worker.h"
#include "search/lc3/positions.h"
#include "search/lc3/settings.h"
#include "search/lc3/watchdog_worker.h"
#include "utils/exception.h"
#include "utils/freelist.h"
#include "utils/thread_pool.h"
#include "utils/worker_pool.h"

namespace lczero {
namespace lc3 {

class SearchSession {
 public:
  SearchSession(ThreadPool* thread_pool, NodeRepository* node_repository,
                const GameState& game_state, Backend* backend,
                UciResponder* uci_responder, const OptionsDict* options);

  void Stop();
  void Abort();
  void Wait();
  // void StartSyncronized();

 private:
  void DrainPipeline();
  bool OkToGather() const { return eval_queue_.SizeApprox() < 1024; }

  // Working tree and current head in this tree.
  PositionTree position_tree_;
  Variation head_;

  // Channels.
  NodeEventReceiver eval_queue_;
  NodeEventReceiver backprop_queue_;
  GatherRateLimiter gather_rate_limiter_;

  // Settings.
  Settings settings_;

  // Workers and threads.
  WorkerPool<GatherWorker> gather_workers_;
  WorkerPool<EvalWorker> eval_workers_;
  WorkerPool<BackpropWorker> backprop_workers_;
  WorkerPool<WatchdogWorker> watchdog_worker_;

  // Worker states
  std::atomic<bool> gather_can_exit_{false};

  NodeEventPool node_event_pool_;
};

}  // namespace lc3
}  // namespace lczero