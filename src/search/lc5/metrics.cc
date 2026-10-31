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

#include "search/lc5/metrics.h"

#include <sstream>

namespace lczero::lc5 {

std::string Metrics::Format(size_t graph_size, size_t active,
                            size_t ready_eval) const {
  std::ostringstream out;
  out << "lc5 eps_evals=" << nn_evaluations.load()
      << " batches=" << evaluation_batches.load()
      << " cache=" << cache_hits.load()
      << " store_hit=" << node_store_hits.load()
      << " suspended=" << visits_suspended.load()
      << " restarted=" << visits_restarted.load()
      << " reselected=" << visits_reselected.load()
      << " proven=" << proven_visits.load()
      << " twofold=" << two_fold_visits.load() << " active=" << active
      << " ready_eval=" << ready_eval << " graph=" << graph_size
      << " mailbox=" << mailbox_notifications.load()
      << " io_high=" << io_jobs_high_water.load()
      << " rejected=" << publications_rejected.load();
  return out.str();
}

}  // namespace lczero::lc5
