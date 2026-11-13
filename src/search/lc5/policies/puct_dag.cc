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

#include <algorithm>
#include <cstdint>
#include <string_view>

#include "search/lc5/graph/graph.h"
#include "search/lc5/graph/value.h"
#include "search/lc5/policies/policy_engine.h"
#include "search/lc5/policies/puct.h"
#include "search/lc5/policies/registry.h"

namespace lczero::lc5 {
namespace {

// PUCT over a graph whose transpositions share their value: an edge caches its
// child's current value (from the last visit through it) instead of averaging
// the samples that went through it, and a node's value is the visit-weighted
// mean of its own samples and its edges' cached values. Every node sends its
// current value to its parent, so a parent sees everything its child learned,
// also through other parents.
//
// Two-fold repetitions and evicted intermediates (whose child node's full
// state is unavailable) are path-dependent samples and fold by averaging;
// out-of-order concurrent backups from older child snapshots keep the fresher
// cached edge mean while preserving the visit-sum invariant. With a short
// HistoryKeyLength, more positions share a node and its value.
struct PuctDag : PuctPolicy<PuctDag> {
  static constexpr std::string_view kName = "puct-dag";

  // Tracks the child's visit count reflected by the edge's cached mean so an
  // out-of-order concurrent backup from an earlier child snapshot cannot
  // overwrite a fresher mean or amplify stale diffs by edge_stats.visits.
  struct EdgeStats : ValueStats {
    uint32_t last_child_visits = 0;
  };
  struct BackupValue {
    VisitValue value;
    GameResultBounds bounds = {};
    bool path_dependent = false;
    uint32_t child_visits = 0;
  };

  static BackupValue MakeLeafBackup(const VisitValue& value, bool exact,
                                    bool path_dependent = false) {
    return {value,
            exact ? GameResultBounds::Exact(value.q, value.m)
                  : GameResultBounds{},
            path_dependent, /*child_visits=*/1};
  }
  static BackupValue MakeSkipBackup(const BackupValue& backup) {
    return {backup.value.Parent(), backup.bounds.Parent(),
            backup.path_dependent, backup.child_visits};
  }
  static BackupValue MakeStaleBackup(const BackupValue& backup) {
    return {backup.value.Parent(), GameResultBounds{},
            /*path_dependent=*/true, /*child_visits=*/0};
  }

  // Sent to the parent during backprop: the node's current mean and bounds.
  static BackupValue MakeCurrentBackup(const Node<PuctDag>& node) {
    const ValueStats& stats = node.stats;
    return {VisitValue{.q = stats.Q(), .d = stats.D(), .m = stats.M()}.Parent(),
            stats.bounds.Parent(), /*path_dependent=*/false, stats.visits};
  }

  // Folds `backup` into the node and, if given, the edge the visit took, and
  // returns the node's current value and bounds. Through an edge with a
  // fresher node-intrinsic value, the node's mean swaps the edge's old
  // contribution (n visits at its cached value) for the new one (n + 1 visits
  // at the child's value). Stale out-of-order backups keep the fresher cached
  // edge mean, and path-dependent samples (in-search 2-fold draws, evicted
  // intermediates) or node-only updates fold by averaging.
  static BackupValue BackupThroughNode(Node<PuctDag>& node,
                                       TriedEdge<PuctDag>* edge,
                                       const BackupValue& backup) {
    ValueStats& stats = node.stats;
    const VisitValue& value = backup.value;
    if (edge && !backup.path_dependent) {
      EdgeStats& edge_stats = edge->stats;
      if (backup.child_visits >= edge_stats.last_child_visits) {
        const float n = static_cast<float>(edge_stats.visits++);
        const float total = static_cast<float>(++stats.visits);
        stats.q += ((n + 1.0f) * value.q - n * edge_stats.q - stats.q) / total;
        stats.d += ((n + 1.0f) * value.d - n * edge_stats.d - stats.d) / total;
        stats.m += ((n + 1.0f) * value.m - n * edge_stats.m - stats.m) / total;
        // Against float drift.
        stats.q = std::clamp(stats.q, -1.0f, 1.0f);
        stats.d = std::clamp(stats.d, 0.0f, 1.0f);
        edge_stats.q = value.q;
        edge_stats.d = value.d;
        edge_stats.m = value.m;
        edge_stats.last_child_visits = backup.child_visits;
        UpdateBounds(node, edge, backup.bounds);
      } else {
        ++edge_stats.visits;
        stats.Add({edge_stats.q, edge_stats.d, edge_stats.m});
      }
    } else {
      stats.Add(value);
      if (edge) edge->stats.Add(value);
      if (!backup.path_dependent) UpdateBounds(node, edge, backup.bounds);
    }
    return MakeCurrentBackup(node);
  }
};
static_assert(sizeof(TriedEdge<PuctDag>) <= 28);
static_assert(sizeof(Node<PuctDag>) == 64);

}  // namespace

// The default policy.
REGISTER_LC5_POLICY(PuctDag, 110)

}  // namespace lczero::lc5
