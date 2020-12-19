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

#include "lc2/search/stats-collector.h"

namespace lczero {
namespace lc2 {

StatsCollector::StatsCollector(UciResponder* responder)
    : uci_responder_(responder),
      start_time_(std::chrono::steady_clock::now()) {}

void StatsCollector::UpdatePv(const MoveList& pv) {
  if (pv != pv_) {
    pv_ = pv;
    OutputThinkingInfo();
  } else {
    MaybeOutputThinkingInfo();
  }
}

void StatsCollector::AddNumEvals(uint32_t num) {
  evals_ += num;
  MaybeOutputThinkingInfo();
}

void StatsCollector::OutputThinkingInfo() {
  std::vector<ThinkingInfo> infos;
  auto& info = infos.emplace_back();
  const auto current_time = std::chrono::steady_clock::now();

  if (!pv_.empty()) info.pv = pv_;
  info.time = std::chrono::duration_cast<std::chrono::milliseconds>(
                  current_time - start_time_)
                  .count();
  info.nodes = evals_;
  if (info.time > 0) info.nps = info.nodes * 1000 / info.time;
  uci_responder_->OutputThinkingInfo(&infos);
  last_info_ = current_time;
}

void StatsCollector::MaybeOutputThinkingInfo() {
  const auto current_time = std::chrono::steady_clock::now();
  if (current_time - last_info_ > std::chrono::seconds(5)) {
    OutputThinkingInfo();
  }
}

}  // namespace lc2
}  // namespace lczero