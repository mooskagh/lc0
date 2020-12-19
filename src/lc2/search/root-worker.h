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

#include <memory>

#include "chess/callbacks.h"
#include "lc2/message/channel.h"

namespace lczero {
namespace lc2 {

class Search;

// Root worker coordinates node gathering (by node worker) and evaluation (by
// eval worker), keeps the current best PV, outputs UCI stats and watches the
// clock.
class RootWorker {
 public:
  RootWorker(Search* search, std::unique_ptr<UciResponder> uci);
  void RunBlocking();

  Channel* channel() { return &channel_; }

 private:
  void HandleMessage(std::unique_ptr<Message>);
  void HandleInitialMessage(std::unique_ptr<Message>);
  void HandleCollisionMessage(std::unique_ptr<Message>);
  void HandleEvalSkipReadyMessage(std::unique_ptr<Message>);
  void HandleBackPropDoneMessage(std::unique_ptr<Message>);

  void SpawnGatherers(int arity);

  Search* const search_;
  const std::unique_ptr<UciResponder> uci_responder_;
  Channel channel_;

  // Current epoch.
  // TODO(crem) Epoch must be persistent between searches.
  uint32_t epoch_ = 0;
  // Spare messages, waiting to be sent when a new epoch starts.
  int messages_idling_ = 0;
  // Number of nodes currently being gathered (or evaled).
  int messages_sent_to_gather_ = 0;
  // Number of nodes currently forward-propagating.
  int messages_sent_to_forwardprop_ = 0;
  // Nodes sent to skip eval.
  int messages_skipping_eval_ = 0;
};

}  // namespace lc2
}  // namespace lczero