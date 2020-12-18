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

#include "lc2/search/root-worker.h"

#include "lc2/search/search.h"
#include "utils/exception.h"

namespace lczero {
namespace lc2 {

RootWorker::RootWorker(Search* search, std::unique_ptr<UciResponder> uci)
    : search_(search), uci_responder_(std::move(uci)) {}

void RootWorker::RunBlocking() {
  // TODO(crem) Epoch must be persistent between searches.
  // uint32_t epoch = 0;

  int tokens_gathering = 0;
  int tokens_blacklisting = 0;
  int tokens_forward_prop = 0;

  auto messages = channel_.DequeueEverything();
  for (auto& msg : messages) {
    switch (msg->type) {
      case Message::kRootInitial:
        assert(tokens_gathering == 0);
        assert(tokens_blacklisting == 0);
        assert(tokens_forward_prop == 0);

        msg->type = Message::kNodeGather;
        // msg->epoch = epoch;
        msg->position_history = search_->history_at_root();
        // msg->attempt = 0;
        msg->is_root_node = true;
        search_->DispatchToNodes(std::move(msg));
        break;
      default:
        throw Exception("Unexpected message type " + std::to_string(msg->type) +
                        " in root worker.");
    }
  }
}

}  // namespace lc2
}  // namespace lczero