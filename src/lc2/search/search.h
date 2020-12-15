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

#pragma once

#include <thread>

#include "lc2/message/manager.h"
#include "lc2/node/keeper.h"
#include "lc2/search/nodes-worker.h"
#include "lc2/search/root-worker.h"
#include "neural/network.h"

namespace lczero {
namespace lc2 {

class Search {
 public:
  Search(Network* network, std::unique_ptr<UciResponder> uci,
         const PositionHistory& root, NodeKeeper* nodes);

  // Search starts (by spawning initial visits).
  void Start();
  void DispatchToRoot(Token token);
  void DispatchToNodes(Token token);
  void DispatchToEval(Token token);

  const PositionHistory& history_at_root() { return rootpos_; }

 private:
  const PositionHistory rootpos_;

  RootWorker root_worker_;
  std::vector<std::unique_ptr<NodesWorker>> nodes_workers_;
  MessageManager message_manager_;

  std::vector<std::thread> threads_;
};

}  // namespace lc2
}  // namespace lczero