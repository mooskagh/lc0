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

#include <cstddef>
#include <cstdint>

#include "chess/callbacks.h"
#include "search/lc5/graph.h"
#include "search/lc5/metrics.h"
#include "search/lc5/settings.h"
#include "search/lc5/visit.h"

namespace lczero::lc5 {

// Per-search state the engine reads or updates on each call.
struct VisitContext {
  const Settings::Resolved& settings;
  Metrics& metrics;
};

// Root numbers the controller needs for limits and early stopping.
struct RootStats {
  bool expanded = false;
  uint64_t visits = 0;  // Node visits; 0 if the root is absent.
  size_t edges = 0;
  uint64_t best_edge_visits = 0;
  uint64_t second_edge_visits = 0;
};

// The search-algorithm side of a visit: everything that reads or writes node
// and edge statistics. SearchRun owns visits, tickets, batching and threads,
// and calls only this. Implementations must be thread-safe; each call locks at
// most one graph shard at a time.
class VisitEngine {
 public:
  enum class DescendResult { kMaterialize, kBackup };
  enum class ResumeAction { kSelect, kRestart };

  virtual ~VisitEngine() = default;

  // Selects and reserves edges from visit.current_key downwards, extending
  // visit.path and visit.history, until the last path step is a node that has
  // to be materialized (kMaterialize) or visit.result holds a value to back up
  // (kBackup: a terminal node or an in-search repetition).
  virtual DescendResult Descend(Visit& visit, const VisitContext& ctx) = 0;
  // Backs visit.result up visit.path.
  virtual void Backup(const Visit& visit, const VisitContext& ctx) = 0;
  // Releases every edge reservation on visit.path without committing a visit.
  virtual void Cancel(const Visit& visit, const VisitContext& ctx) = 0;
  // A waiter's node finished materializing and it is about to select again.
  virtual ResumeAction OnResume(const Visit& visit,
                                const VisitContext& ctx) = 0;
  // Fills the score, WDL and PV of `info`.
  virtual void Report(const VisitOrigin& root, const VisitContext& ctx,
                      ThinkingInfo& info) const = 0;
  virtual RootStats Root(NodeKey root) const = 0;

  // Node lifecycle for the materialization machinery.
  virtual FindOrCreateResult FindOrCreateMaterializing(
      NodeKey key, MaterializationTicketId ticket) = 0;
  // Expands the node; an accepted install counts `payload.leaf` as the node's
  // first visit.
  virtual InstallResult Install(NodeKey key, MaterializationTicketId ticket,
                                const ExpansionPayload& payload) = 0;
  virtual void CancelMaterialization(NodeKey key,
                                     MaterializationTicketId ticket) = 0;
  virtual size_t GraphSize() const = 0;
};

}  // namespace lczero::lc5
