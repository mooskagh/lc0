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

RootWorker::RootWorker(Search* search, UciResponder* uci)
    : search_(search), stats_collector_(uci) {}

void RootWorker::RunBlocking() {
  // TODO(crem) Epoch must be persistent between searches.
  // uint32_t epoch = 0;

  while (true) {
    auto messages = channel_.DequeueEverything();
    bool had_updates = false;
    for (auto& msg : messages) {
      if (msg->type == Message::kRootBackPropDone) had_updates = true;
      HandleMessage(std::move(msg));
    }
    SpawnPVGatherer();
    if (had_updates) {
      ++epoch_;
      if (messages_idling_ > 0) SpawnGatherers(messages_idling_);
    }
  }
}

void RootWorker::HandleMessage(std::unique_ptr<Message> msg) {
  switch (msg->type) {
    case Message::kRootInitial:
      HandleInitialMessage(std::move(msg));
      return;
    case Message::kRootCollision:
      HandleCollisionMessage(std::move(msg));
      return;
    case Message::kRootEvalSkipped:
      HandleEvalSkipReadyMessage(std::move(msg));
      return;
    case Message::kRootBackPropDone:
      HandleBackPropDoneMessage(std::move(msg));
      return;
    case Message::kRootPVGathered:
      HandlePVGathered(std::move(msg));
      return;
    default:
      throw Exception("Unexpected message type " + std::to_string(msg->type) +
                      " in root worker.");
  }
}

void RootWorker::SpawnGatherers(int arity) {
  assert(messages_idling_ >= arity);
  messages_idling_ -= arity;
  messages_sent_to_gather_ += arity;

  auto msg = std::make_unique<Message>();
  msg->type = Message::kNodeGather;
  msg->arity = arity;
  msg->epoch = epoch_;
  msg->position_history = search_->history_at_root();
  // msg->attempt = 0;
  search_->DispatchToNodes(std::move(msg));
}

void RootWorker::SpawnPVGatherer() {
  auto msg = std::make_unique<Message>();
  msg->type = Message::kNodeGatherPV;
  msg->arity = 1;
  msg->epoch = epoch_;
  msg->position_history = search_->history_at_root();
  msg->pv = Message::PV{};
  search_->DispatchToNodes(std::move(msg));
}

void RootWorker::HandleInitialMessage(std::unique_ptr<Message> msg) {
  assert(messages_sent_to_gather_ == 0);
  assert(messages_skipping_eval_ == 0);
  assert(messages_idling_ == 0);
  messages_idling_ += msg->arity;
  SpawnGatherers(msg->arity);
}

void RootWorker::HandleCollisionMessage(std::unique_ptr<Message> msg) {
  assert(messages_sent_to_gather_ >= msg->arity);
  messages_sent_to_gather_ -= msg->arity;
  messages_skipping_eval_ += msg->arity;
  // TODO Later we'll handle collisions, but for now, just skip them in
  // eval.
  msg->type = Message::kEvalSkip;
  search_->DispatchToEval(std::move(msg));
}

void RootWorker::HandleEvalSkipReadyMessage(std::unique_ptr<Message> msg) {
  assert(messages_skipping_eval_ >= msg->arity);
  messages_skipping_eval_ -= msg->arity;
  messages_idling_ += msg->arity;
}

void RootWorker::HandleBackPropDoneMessage(std::unique_ptr<Message> msg) {
  assert(messages_sent_to_gather_ >= msg->arity);
  stats_collector_.AddNumEvals(msg->arity);
  messages_sent_to_gather_ -= msg->arity;
  messages_idling_ += msg->arity;
}

void RootWorker::HandlePVGathered(std::unique_ptr<Message> msg) {
  auto& pv = msg->pv->pv;
  if (pv.empty()) return;
  for (int i = search_->history_at_root().IsBlackToMove() ? 0 : 1;
       i < static_cast<int>(pv.size()); i += 2) {
    pv[i].Mirror();
  }
  stats_collector_.UpdatePv(pv);
}

}  // namespace lc2
}  // namespace lczero