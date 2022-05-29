/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2022 The LCZero Authors

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

#include "chess/callbacks.h"
#include "mcts/search.h"

namespace lczero {

class Search::Responder {
 public:
  Responder(const Search& search, std::unique_ptr<UciResponder> responder);

  void SendUciInfo() const;
  void OutputBestMove(Move bestmove, Move pondermove) const;
  void OutputComment(std::string_view text) const;
  void SendMovesStats() const;
  void MaybeOutputInfo() const;

 private:
  int Depth() const;

  const Search& search_;
  const std::unique_ptr<UciResponder> responder_;
  mutable Edge* previous_edge_ GUARDED_BY(search_.nodes_mutex_) = nullptr;
  mutable ThinkingInfo previous_info_ GUARDED_BY(search_.nodes_mutex_);
};

}  // namespace lczero