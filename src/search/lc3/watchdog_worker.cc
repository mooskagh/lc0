#include "search/lc3/watchdog_worker.h"

namespace lczero {
namespace lc3 {

void WatchdogWorker::CheckOnce() {
  AccessLock lock = ctx_.node_repository->GetAccessLock();
  std::optional<NodeView> root_view = lock.FetchReadOnly((*ctx_.head)->hash);
  if (!root_view) return;
  CERR << root_view->GetN();
}
}  // namespace lc3
}  // namespace lczero
