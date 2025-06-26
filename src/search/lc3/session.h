#pragma once

#include <memory>
#include <thread>
#include <vector>

#include "search/lc3/backprop_worker.h"
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
                SearchChannels* search_channels, const OptionsDict* options);

  void Abort();
  void Wait();
  // void StartSyncronized();

 private:
  PositionTree position_tree_;
  Variation head_;
  SearchChannels* search_channels_;
  Settings settings_;
  std::vector<std::unique_ptr<MctsGatherWorker>> gather_workers_;
  std::vector<std::unique_ptr<EvalWorker>> eval_workers_;
  std::vector<std::unique_ptr<BackpropWorker>> backprop_workers_;
  std::unique_ptr<WatchdogWorker> watchdog_worker_;
  EvalItemPool eval_item_pool_;

  std::vector<std::thread> threads_;
};

}  // namespace lc3
}  // namespace lczero