/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#pragma once

#include <chrono>
#include <memory>
#include <optional>

#include "chess/gamestate.h"
#include "search/lc5/graph.h"
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
  friend class SearchRunTestPeer;
  void AbortAndWait();

  const OptionsDict* options_;
  // Retained without eviction for the whole game, so no NodeStore is needed.
  GameGraph graph_;
  std::unique_ptr<SearchRun> run_;
  std::optional<Position> game_start_;
  std::optional<PositionHistory> root_history_;
  std::optional<std::chrono::steady_clock::time_point> clock_start_;
  // Time a previous search left unused by stopping early.
  int64_t saved_time_ms_ = 0;
};

}  // namespace lczero::lc5
