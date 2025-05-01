#include "search/lc3/backprop_worker.h"

#include <array>
#include <vector>

#include "search/lc3/channels.h"

namespace lczero {
namespace lc3 {

void BackpropWorker::OneStep() {
  std::vector<EvalItem*> items;  // TODO replace with non-initialized vector,
  // without need for buffer.
  {
    absl::MutexLock lock(&ctx_.search_channels->request_consumer_mutex_);
    std::array<EvalItem*, 1024> buffer;
    size_t num_items =
        ctx_.search_channels->FetchEvalResults(buffer, /*block=*/true);
    do {
      items.insert(items.end(), buffer.begin(), buffer.begin() + num_items);
      num_items = ctx_.search_channels->FetchEvalResults(buffer,
                                                         /*block=*/false);
    } while (num_items > 0);
  }

  struct Backpropagation {
    size_t depth;
    EvalItem* original_item;
    size_t num_visits_to_apply;
    size_t num_visits_to_undo;
    double q;
    float d;
    float m;
  };
}

}  // namespace lc3
}  // namespace lczero