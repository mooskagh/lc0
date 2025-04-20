#pragma once

#include "search/lc3/mcts_worker.h"
#include "search/search.h"
#include "search/lc3/types.h"

namespace lczero {
namespace lc3 {

class Lc3Engine : public SearchBase {
 public:
  using SearchBase::SearchBase;

  void SetPosition(const GameState&) override;
  void StartSearch(const GoParams& go_params) override;
  void StartClock() override { TODO("Start clock"); }
  void StopSearch() override { NotImplemented(); }
  void AbortSearch() override;
  void WaitSearch() override;

 private:
  void EnsureSearchStopped();

  std::unique_ptr<MctsWorker> search_;
  NodeStorage storage_;
  // The positions that already occurred in the game.
  std::vector<PositionChain> position_history_;
  EvalQueue eval_queue_;
};

}  // namespace lc3
}  // namespace lczero