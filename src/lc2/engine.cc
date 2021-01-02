/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors

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

#include "lc2/engine.h"

#include "lc2/search/search.h"
#include "neural/factory.h"
#include "utils/configfile.h"

namespace lczero {
namespace lc2 {
namespace {

const OptionId kNumShardsId{
    "num-shards", "NumShards",
    "Number of shards in positions hashtable. Every shard may be processed "
    "independently by a separate thread."};

const OptionId kLogFileId{"logfile", "LogFile",
                          "Write log to that file. Special value <stderr> to "
                          "output the log to the console.",
                          'l'};

void PopulateUciOptions(OptionsParser* options) {
  NetworkFactory::PopulateOptions(options);
  options->Add<IntOption>(kNumShardsId, 1, 1024) = 1;  // 2
  options->Add<StringOption>(kLogFileId);
  ConfigFile::PopulateOptions(options);
}

}  // namespace

Engine::Engine()
    : uci_responder_(std::make_unique<CallbackUciResponder>(
          std::bind(&UciLoop::SendBestMove, this, std::placeholders::_1),
          std::bind(&UciLoop::SendInfo, this, std::placeholders::_1))) {
  ChessBoard board;
  int no_capture_ply;
  int full_moves;

  board.SetFromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                   &no_capture_ply, &full_moves);
  position_history_.Reset(board, no_capture_ply,
                          full_moves * 2 - (board.flipped() ? 1 : 2));

  PopulateUciOptions(&options_);
}

void Engine::RunLoop() {
  if (!ConfigFile::Init() || !options_.ProcessAllFlags()) return;
  Logging::Get().SetFilename(
      options_.GetOptionsDict().Get<std::string>(kLogFileId));
  assert(!node_keeper_);
  node_keeper_ = std::make_unique<NodeKeeper>(
      options_.GetOptionsDict().Get<int>(kNumShardsId));
  UciLoop::RunLoop();
}

void Engine::CmdGo(const GoParams&) {
  network_ = NetworkFactory::LoadNetwork(options_.GetOptionsDict());
  search_ = std::make_unique<Search>(network_.get(), uci_responder_.get(),
                                     position_history_, node_keeper_.get());
  search_->Start();
}

}  // namespace lc2
}  // namespace lczero