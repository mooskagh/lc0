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

#include "search/lc5/graph.h"
#include "search/lc5/visit_engine.h"

namespace lczero::lc5 {

// PUCT search over a GameGraph, which it does not own.
class PolicyEngine final : public VisitEngine {
 public:
  explicit PolicyEngine(GameGraph* graph) : graph_(graph) {}

  DescendResult Descend(Visit& visit, const VisitContext& ctx) override;
  void Backup(const Visit& visit, const VisitContext& ctx) override;
  void Cancel(const Visit& visit, const VisitContext& ctx) override;
  ResumeAction OnResume(const Visit& visit, const VisitContext& ctx) override;
  void Report(const VisitOrigin& root, const VisitContext& ctx,
              ThinkingInfo& info) const override;
  RootStats Root(NodeKey root) const override;

  FindOrCreateResult FindOrCreateMaterializing(
      NodeKey key, MaterializationTicketId ticket) override;
  InstallResult Install(NodeKey key, MaterializationTicketId ticket,
                        const ExpansionPayload& payload) override;
  void CancelMaterialization(NodeKey key,
                             MaterializationTicketId ticket) override;
  size_t GraphSize() const override;

 private:
  GameGraph* const graph_;
};

}  // namespace lczero::lc5
