/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors

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

#include "lc2/chess/position-key.h"
#include "lc2/mcts/node.h"
#include "lc2/storage/storage.h"

namespace lc2 {

using NodeStorage = Storage<uint64_t, NodeHead, NodeTail>;

class BatchInfo {
 public:
  void Reset();
  void EnqueuePosition(const PositionContext& context,
                       lczero::ChessBoard& board);

 private:
  std::vector<PositionContext> contexts_;
  std::vector<lczero::ChessBoard> boards_;
  std::vector<NodeHead> node_heads_;
};

// Does batch gathering and backpropagation (shortly speaking, MCTS).
// Tries to gather a batch of @size elements from @node_storage, starting from
// @position.
void GatherBatch(const PositionContext& context, lczero::ChessBoard& board,
                 NodeStorage* const storage, size_t size, BatchInfo* batch);

}  // namespace lc2