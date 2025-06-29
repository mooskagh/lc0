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

namespace lczero {
namespace lc3 {

class SearchSession {
 public:
  SearchSession(NodeRepository* node_repository, const GameState& game_state,
                Backend* backend, UciResponder* uci_responder,
                const OptionsDict* options);

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
  EvalItemReceiver eval_queue_;
  EvalItemReceiver backprop_queue_;
  GatherRateLimiter gather_rate_limiter_;

  // Settings.
  Settings settings_;

  // Workers and threads.
  std::vector<std::unique_ptr<GatherWorker>> gather_workers_;
  std::vector<std::thread> gather_threads_;
  std::vector<std::unique_ptr<EvalWorker>> eval_workers_;
  std::vector<std::thread> eval_threads_;
  std::vector<std::unique_ptr<BackpropWorker>> backprop_workers_;
  std::vector<std::thread> backprop_threads_;
  std::unique_ptr<WatchdogWorker> watchdog_worker_;
  std::thread watchdog_thread_;

  // Worker states
  absl::Notification watchdog_can_exit_;
  std::atomic<bool> ok_to_respond_bestmove_{true};

  EvalItemPool eval_item_pool_;
};

}  // namespace lc3
}  // namespace lczero