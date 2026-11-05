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

#include <absl/container/flat_hash_map.h>
#include <absl/synchronization/mutex.h>

#include <cassert>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

#include "search/lc5/graph/key.h"
#include "search/lc5/graph/node.h"
#include "search/lc5/session/messages.h"

namespace lczero::lc5 {

// A node being materialized and the visits waiting for it.
struct Ticket {
  NodeKey key;        // The node being materialized.
  VisitWakeup owner;  // The visit that opened the ticket; it does the I/O.
  std::vector<VisitWakeup> waiters;  // Later visits that reached the node.
};

// The open tickets, at most one per node. Any visit worker opens or joins
// one; only its owner's worker closes it.
//
// The callbacks run under the table's mutex, and attach a ticket to its node
// or install the node: lock order is tickets -> one graph shard. So a visit
// that finds no open ticket for a node also finds the node installed.
class TicketTable {
 public:
  struct Joined {
    MaterializationTicketId id;
    bool owner;          // Opened the ticket.
    size_t waiters = 0;  // Waiters on the ticket, if joined as one.
  };

  // `attach(id)` attaches ticket `id` to node `key`, a new ticket unless one
  // is open for it, and returns whether to wait for it. If so, `wakeup` joins
  // the ticket, as its owner if it is new.
  template <typename Attach>
  std::optional<Joined> Join(NodeKey key, VisitWakeup wakeup, Attach attach) {
    absl::MutexLock lock(mutex_);
    const auto existing = id_by_key_.find(key);
    const bool owner = existing == id_by_key_.end();
    const MaterializationTicketId id = owner ? next_id_ : existing->second;
    if (!attach(id)) return std::nullopt;
    wakeup.ticket = id;
    if (owner) {
      ++next_id_;
      tickets_.emplace(id, Ticket{key, wakeup, {}});
      id_by_key_.emplace(key, id);
      return Joined{.id = id, .owner = true};
    }
    auto& waiters = tickets_.at(id).waiters;
    waiters.push_back(wakeup);
    return Joined{.id = id, .owner = false, .waiters = waiters.size()};
  }

  // `close(key)` gets open ticket `id`'s node `key` its content and returns
  // whether it did. If so, closes the ticket and returns it.
  template <typename Op>
  std::optional<Ticket> Close(MaterializationTicketId id, Op close) {
    absl::MutexLock lock(mutex_);
    const auto it = tickets_.find(id);
    assert(it != tickets_.end());
    if (!close(it->second.key)) return std::nullopt;
    Ticket ticket = std::move(it->second);
    tickets_.erase(it);
    id_by_key_.erase(ticket.key);
    return ticket;
  }

  bool empty() const {
    absl::MutexLock lock(mutex_);
    assert(tickets_.size() == id_by_key_.size());
    return tickets_.empty();
  }

 private:
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<NodeKey, MaterializationTicketId> id_by_key_
      ABSL_GUARDED_BY(mutex_);
  std::unordered_map<MaterializationTicketId, Ticket> tickets_
      ABSL_GUARDED_BY(mutex_);
  MaterializationTicketId next_id_ ABSL_GUARDED_BY(mutex_) = 1;
};

}  // namespace lczero::lc5
