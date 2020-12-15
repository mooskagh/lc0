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

namespace lczero {
namespace lc2 {
namespace {
bool matches_class(Message* msg, Message::Class c) {
  const int cls = msg->type >> 8;
  return (cls == 0) || (cls == static_cast<int>(c));
}
}  // namespace

Search::Search(Network*, std::unique_ptr<UciResponder> uci,
               const PositionHistory& root, NodeKeeper* nodes)
    : rootpos_(root), root_worker_(this, std::move(uci)) {
  for (auto& shard : *nodes->shards()) {
    nodes_workers_.push_back(std::make_unique<NodesWorker>(this, &shard));
  }

  threads_.emplace_back([this]() { root_worker_.RunBlocking(); });
  for (auto& worker : nodes_workers_) {
    threads_.emplace_back([&worker]() { worker->RunBlocking(); });
  }
}

void Search::Start() {
  auto token = message_manager_.CreateFreshToken();

  auto* msg = token.message();
  msg->type = Message::kRootInitial;
  msg->arity = 3;  // TODO(crem) DO NOT SUBMIT, take that from params.

  DispatchToRoot(std::move(token));
}

void Search::DispatchToRoot(Token token) {
  assert(matches_class(token.message(), Message::Class::kRoot));
  root_worker_.channel()->Enqueue(std::move(token));
}

void Search::DispatchToNodes(Token token) {
  auto* msg = token.message();
  assert(matches_class(token.message(), Message::Class::kNode));
  auto hash = msg->position_history.Last().Hash();
  auto shard = hash % nodes_workers_.size();
  nodes_workers_[shard]->channel()->Enqueue(std::move(token));
}

}  // namespace lc2
}  // namespace lczero