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

#include "search/lc5/engine.h"

#include <string>
#include <utility>

#include "chess/callbacks.h"
#include "chess/uciloop.h"
#include "search/lc5/settings.h"
#include "search/register.h"
#include "utils/exception.h"

namespace lczero::lc5 {

Lc5Engine::~Lc5Engine() { AbortAndWait(); }

void Lc5Engine::AbortAndWait() {
  if (!run_) return;
  run_->Abort();
  run_->Wait();
}

void Lc5Engine::SetBackend(Backend* backend) {
  if (backend_ == backend) return;
  AbortAndWait();
  backend_ = backend;
  if (visit_engine_) visit_engine_->Clear();
}

void Lc5Engine::SetPosition(const GameState& state) {
  AbortAndWait();
  // The graph survives position changes within the same game.
  if (game_start_ != state.startpos && visit_engine_) {
    visit_engine_->Clear();
  }
  game_start_ = state.startpos;
  root_history_.emplace();
  root_history_->Reset(state.startpos);
  for (Move move : state.moves) root_history_->Append(move);
}

void Lc5Engine::NewGame() {
  AbortAndWait();
  run_.reset();
  if (visit_engine_) visit_engine_->Clear();
  game_start_.reset();
  root_history_.reset();
  clock_start_.reset();
  saved_time_ms_ = 0;
}

void Lc5Engine::StartClock() {
  clock_start_ = std::chrono::steady_clock::now();
}

void Lc5Engine::StartSearch(const GoParams& params) {
  AbortAndWait();
  if (run_) saved_time_ms_ += run_->unused_time_ms();
  run_.reset();
  if (!backend_) throw Exception("Lc5 requires a backend");
  if (!root_history_) throw Exception("Lc5 position was not set");

  // A different policy keeps different statistics, so it starts a new graph.
  const std::string policy = PolicyName(*options_);
  if (!visit_engine_ || visit_engine_->name() != policy) {
    visit_engine_ = MakeVisitEngine(policy);
  }
  visit_engine_->Configure(*options_);
  const Settings settings(*options_);
  VisitOrigin root{
      .key = MakeNodeKey(*root_history_, settings.history_key_length()),
      .history = *root_history_,
      .backup_prefix = {}};
  const auto start = clock_start_.value_or(std::chrono::steady_clock::now());
  clock_start_.reset();
  run_ = std::make_unique<SearchRun>(
      visit_engine_.get(), /*store=*/nullptr, backend_, uci_responder_,
      settings.Resolve(backend_->GetAttributes()), settings.time_management(),
      std::move(root), params, start, std::exchange(saved_time_ms_, 0));
  run_->Start();
}

void Lc5Engine::StopSearch() {
  if (run_) run_->Stop();
}

void Lc5Engine::AbortSearch() {
  if (run_) run_->Abort();
}

void Lc5Engine::WaitSearch() {
  if (run_) run_->Wait();
}

namespace {
class Lc5Factory final : public SearchFactory {
 public:
  std::string_view GetName() const override { return "lc5"; }
  void PopulateParams(OptionsParser* options) const override {
    Settings::Populate(options);
    PopulatePolicyOptions(options);
  }
  std::unique_ptr<SearchBase> CreateSearch(
      UciResponder* responder, const OptionsDict* options) const override {
    return std::make_unique<Lc5Engine>(responder, options);
  }
};

REGISTER_SEARCH(Lc5Factory)
}  // namespace

}  // namespace lczero::lc5
