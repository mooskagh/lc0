#include "search/lc3/context.h"

#pragma once

namespace lczero {
namespace lc3 {

class WatchdogWorker {
 public:
  WatchdogWorker(const Context& context) : ctx_(context) {}

  void CheckOnce();
  std::vector<Move> BuildPV() const;

 private:
  const Context ctx_;
};

}  // namespace lc3
}  // namespace lczero