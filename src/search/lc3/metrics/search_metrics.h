#pragma once

#include "src/search/lc3/metrics/nodes_metric.h"
#include "src/utils/metrics/group.h"

namespace lczero {
namespace lc3 {

using SearchMetrics = MetricGroup<GatherNodesMetrics>;

}  // namespace lc3
}  // namespace lczero