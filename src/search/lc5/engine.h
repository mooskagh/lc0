#pragma once

#include <chrono>
#include <memory>
#include <optional>

#include "chess/gamestate.h"
#include "search/lc5/graph.h"
#include "search/lc5/memory_node_store.h"
#include "search/lc5/search.h"
#include "search/search.h"

namespace lczero::lc5 {

class Lc5Engine final : public SearchBase {
 public:
  Lc5Engine(UciResponder* responder, const OptionsDict* options)
      : SearchBase(responder), options_(options) {}
  ~Lc5Engine() override;

  void SetBackend(Backend* backend) override;
  void SetPosition(const GameState& state) override;
  void StartSearch(const GoParams& params) override;
  void StartClock() override;
  void StopSearch() override;
  void AbortSearch() override;
  void WaitSearch() override;
  void NewGame() override;

 private:
  void AbortAndWait();
  void ClearGame();

  const OptionsDict* options_;
  GameGraph graph_;
  MemoryNodeStore store_;
  std::unique_ptr<SearchRun> run_;
  std::optional<Position> game_start_;
  std::optional<PositionHistory> root_history_;
  NodeKey root_key_;
  std::optional<std::chrono::steady_clock::time_point> clock_start_;
};

}  // namespace lczero::lc5
