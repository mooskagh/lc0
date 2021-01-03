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

#include "lc2/search/search.h"

#include "utils/logging.h"

namespace lczero {
namespace lc2 {
namespace {
bool matches_class(Message* msg, Message::Class c) {
  const int cls = msg->type >> 8;
  return (cls == 0) || (cls == static_cast<int>(c));
}
}  // namespace

Search::Search(Network* network, UciResponder* uci, const PositionHistory& root,
               NodeKeeper* nodes)
    : rootpos_(root),
      epoch_counter_(nodes->epoch_counter()),
      root_worker_(this, uci),
      eval_worker_(this, network) {
  for (auto& shard : *nodes->shards()) {
    nodes_workers_.push_back(std::make_unique<NodesWorker>(this, &shard));
  }

  threads_.emplace_back([this]() { root_worker_.RunBlocking(); });
  threads_.emplace_back([this]() { eval_worker_.RunBlocking(); });
  for (auto& worker : nodes_workers_) {
    threads_.emplace_back([&worker]() { worker->RunBlocking(); });
  }
}

Search::~Search() { JoinAllThreads(); }

void Search::Start() {
  auto msg = std::make_unique<Message>();

  msg->type = Message::kRootInitial;
  msg->arity = 512;  // TODO(crem) DO NOT SUBMIT, take that from params.

  DispatchToRoot(std::move(msg));
}

void Search::DispatchToRoot(std::unique_ptr<Message> msg) {
  LOGFILE << *msg;
  assert(matches_class(msg.get(), Message::Class::kRoot));
  root_worker_.channel()->Enqueue(std::move(msg));
}

void Search::DispatchToNodes(std::unique_ptr<Message> msg) {
  LOGFILE << *msg;
  assert(matches_class(msg.get(), Message::Class::kNode));
  auto hash = msg->position_history.Last().Hash();
  auto shard = hash % nodes_workers_.size();
  nodes_workers_[shard]->channel()->Enqueue(std::move(msg));
}

void Search::DispatchToEval(std::unique_ptr<Message> msg) {
  LOGFILE << *msg;
  assert(matches_class(msg.get(), Message::Class::kEval));
  eval_worker_.channel()->Enqueue(std::move(msg));
}

void Search::JoinAllThreads() {
  while (!threads_.empty()) {
    threads_.back().join();
    threads_.pop_back();
  }
}

}  // namespace lc2
}  // namespace lczero