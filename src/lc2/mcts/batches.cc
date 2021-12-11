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

#include "lc2/mcts/batches.h"

namespace lc2 {

void BatchInfo::Reset() {
  assert(boards_.empty());  
  assert(contexts_.empty());  
  assert(positions_keys_.empty());  
  assert(node_heads_.empty());  
  assert(node_tails_.empty());  
}

void BatchInfo::EnqueuePosition(const PositionContext& context,
                                lczero::ChessBoard& board) {
  contexts_.emplace_back(context);
  boards_.emplace_back(board);
  positions_keys_.emplace_back(PositionKey::FromBoard(board));
}

void GatherBatch(const PositionContext& context, lczero::ChessBoard& board,
                 NodeStorage* const storage, size_t size, BatchInfo* batch) {
  batch->Reset();
  batch->EnqueuePosition(context, board);

  size_t cur = 0;
}

}  // namespace lc2