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
      << " suspended=" << visits_suspended.load() << " active=" << active
      << " ready_eval=" << ready_eval << " graph=" << graph_size
      << " mailbox=" << mailbox_notifications.load()
      << " io_high=" << io_jobs_high_water.load()
      << " rejected=" << publications_rejected.load();
  return out.str();
}

}  // namespace lczero::lc5
