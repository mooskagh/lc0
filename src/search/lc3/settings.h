#pragma once

#include "utils/optionsdict.h"
#include "utils/optionsparser.h"

namespace lczero {
namespace lc3 {

// UCI options for the Lc3 algorithm (in classic algorithm it was called
// "params").
class Settings {
 public:
  Settings(const OptionsDict& options);
  Settings(const Settings&) = delete;

  static void Populate(OptionsParser* options);

  // Parameter getters
  int GetNumGatherThreads() const { return kNumGatherThreads; }
  int GetNumEvalThreads() const { return kNumEvalThreads; }
  int GetNumBackpropThreads() const { return kNumBackpropThreads; }

 private:
  const OptionsDict& options_;
  // Cached parameter values
  const int kNumGatherThreads;
  const int kNumEvalThreads;
  const int kNumBackpropThreads;
};

}  // namespace lc3
}  // namespace lczero