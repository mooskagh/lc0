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

#include <condition_variable>
#include <cstdint>
#include <deque>

#include "lc2/message/manager.h"
#include "utils/mutex.h"

namespace lczero {
namespace lc2 {

// TODO(crem)
// Channels are easy to make lockless. However let's first confirm them that
// it's a bottleneck.
class Channel {
 public:
  // Adds a token to a queue.
  void Enqueue(Token&& token);
  // Gets a token from a queue. Blocks until token is available if empty.
  Token Dequeue();
  // Gets all tokens from a queue. Blocks until token is available if empty.
  std::vector<Token> DequeueEverything();

 private:
  Mutex mutex_;
  std::deque<Token> tokens_ GUARDED_BY(mutex_);
  std::condition_variable cond_var_;
};

}  // namespace lc2
}  // namespace lczero