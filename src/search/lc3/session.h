#pragma once

#include <absl/synchronization/notification.h>

#include <memory>
#include <thread>
#include <vector>

#include "search/lc3/metrics/game_stats.h"
#include "search/lc3/node_repository/node_repository.h"
#include "search/lc3/search_policy/search_policy.h"
#include "search/lc3/settings.h"
#include "search/lc3/workers/backprop_worker.h"
#include "search/lc3/workers/eval_worker.h"
#include "search/lc3/workers/gather_worker.h"
#include "search/lc3/workers/node_event_queue.h"
#include "search/lc3/workers/watchdog_worker.h"
#include "utils/exception.h"
#include "utils/freelist.h"
#include "utils/thread_pool.h"
#include "utils/worker_pool.h"

namespace lczero {
namespace lc3 {

// Search session, i.e. one search from start to stop.
// It's main (or solely) purpose is to orchestrate the workers and the
// pipelines.
class SearchSession {
 public:
  SearchSession(ThreadPool* thread_pool, NodeRepository* node_repository,
                const GameState& game_state, Backend* backend,
                UciResponder* uci_responder, const OptionsDict* options,
                GameStats* game_stats);
  ~SearchSession();

  // Makes search stop. Doesn't wait for it to stop (call Wait() to wait for it
  // to stop). The interrupted search doesn't send bestmove.
  void Abort();
  // Same as Abort, but ensures (blockingly!) that search can respond the
  // bestmove, and makes it respond it. Afterwards, this function returns, but
  // the workers may still be running (call Wait() to wait for them to stop).
  void Stop();
  // Waits all workers to stop.
  void Wait();
  // void StartSyncronized();

 private:
  using Policy = SearchPolicy;
  void DrainPipeline();
  // Function that allows gather workers to starrt gathering a new batch.
  bool OkToGather() const { return eval_queue_.SizeApprox() < 1024; }

  // The temporary search tree for this session (not the persistent node
  // repository).
  PositionTree position_tree_;
  // The current position, or "search root", which may differ from the tree's
  // game-start root.
  Variation head_;

  // Channels/queues.
  NodeEventReceiver eval_queue_;
  NodeEventReceiver backprop_queue_;

  // A class which eval thread uses to notify gather threads that they may
  // consider starting gathering.
  // TODO Move OkToGather and this into something proper.
  GatherRateLimiter gather_rate_limiter_;

  // Search settings. This was called "params" in the old search.
  Settings settings_;

  // A freelist allocator for NodeEvent objects.
  // TODO Likely this is excessive, and unique_ptr would be simpler and good.
  NodeEventPool node_event_pool_;

  // Workers and threads.
  WorkerPool<GatherWorker> gather_workers_;
  WorkerPool<EvalWorker> eval_workers_;
  WorkerPool<BackpropWorker> backprop_workers_;
  // There's exactly one watchdog worker, we still use the pool for symmetry.
  WorkerPool<WatchdogWorker> watchdog_worker_;

  // Game stats.
  GameStats* game_stats_;
};

}  // namespace lc3
}  // namespace lczero