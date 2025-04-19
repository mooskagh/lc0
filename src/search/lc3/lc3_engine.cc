/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2025 The LCZero Authors

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

#include "search/lc3/mcts_worker.h"
#include "search/lc3/positions.h"
#include "search/lc3/session.h"
#include "search/lc3/storage.h"
#include "search/register.h"
#include "search/search.h"

namespace lczero {
namespace lc3 {
namespace {

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
};

void Lc3Engine::AbortSearch() {
  if (search_) search_->Abort();
}

void Lc3Engine::WaitSearch() {
  if (search_) search_->Wait();
}

void Lc3Engine::EnsureSearchStopped() {
  AbortSearch();
  WaitSearch();
}

void Lc3Engine::SetPosition(const GameState& game_state) {
  EnsureSearchStopped();
  TODO("GC the storage_");
  position_history_ = GameStateToPositionChain(game_state);
}

void Lc3Engine::StartSearch(const GoParams& go_params) {
  TODO("Do not ignore go_params");
  EnsureSearchStopped();
  search_ = std::make_unique<MctsWorker>(&storage_, position_history_.back());
  search_->OneStep();
}

class Lc3Factory : public SearchFactory {
  std::string_view GetName() const override { return "lc3"; }
  std::unique_ptr<SearchBase> CreateSearch(UciResponder* responder,
                                           const OptionsDict*) const override {
    return std::make_unique<Lc3Engine>(responder);
  }
};

REGISTER_SEARCH(Lc3Factory)
}  // namespace
}  // namespace lc3
}  // namespace lczero