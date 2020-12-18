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

#include "lc2/message/message.h"

namespace lczero {
namespace lc2 {

std::unique_ptr<Message> Message::SplitOff(int how_much) {
  assert(how_much > 0);
  assert(how_much < arity);

  auto res = std::make_unique<Message>(*this);
  res->arity = how_much;
  arity -= how_much;
  return res;
}

std::string MessageTypeToString(Message::Type t) {
  switch (t) {
    case Message::kUnknown:
      return "kUnknown";
    case Message::kAuxAbort:
      return "kAuxAbort";
    case Message::kNodeGather:
      return "kNodeGather";
    case Message::kNodeBlacklist:
      return "kNodeBlacklist";
    case Message::kNodeForwardProp:
      return "kNodeForwardProp";
    case Message::kRootInitial:
      return "kRootInitial";
    case Message::kRootCollision:
      return "kRootCollision";
    case Message::kRootEvalReady:
      return "kRootEvalReady";
    case Message::kRootEvalSkipReady:
      return "kRootEvalSkipReady";
    case Message::kRootOutOfOrderEvalReady:
      return "kRootOutOfOrderEvalReady";
    case Message::kRootBlacklistDone:
      return "kRootBlacklistDone";
    case Message::kRootForwardPropDone:
      return "kRootForwardPropDone";
    case Message::kEvalEval:
      return "kEvalEval";
    case Message::kEvalSkip:
      return "kEvalSkip";
  };
  return "Unexpected value";
}

std::ostream& operator<<(std::ostream& os, const Message& m) {
  os << "Message(" << MessageTypeToString(m.type);
  os << ")[" << m.arity << ":" << m.epoch << ":depth("
     << m.position_history.GetLength() << ")";
  if (m.eval_result) os << ":E";
  os << "]";
  return os;
}

}  // namespace lc2
}  // namespace lczero