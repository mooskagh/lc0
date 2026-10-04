#include "search/lc5/settings.h"

#include <gtest/gtest.h>

#include "utils/exception.h"

namespace lczero::lc5 {
namespace {

TEST(Lc5SettingsTest, DefaultsResolveAgainstBackend) {
  OptionsParser parser;
  Settings::Populate(&parser);
  Settings settings(parser.GetOptionsDict());
  BackendAttributes attrs{.has_mlh = true,
                          .has_wdl = true,
                          .runs_on_cpu = false,
                          .suggested_num_search_threads = 2,
                          .recommended_batch_size = 64,
                          .maximum_batch_size = 128};
  auto resolved = settings.Resolve(attrs);
  EXPECT_EQ(resolved.eval_threads, 2);
  EXPECT_EQ(resolved.minibatch_size, 64);
  EXPECT_GE(resolved.max_active_visits, 1024);
  EXPECT_EQ(resolved.history_key_length, 7);
  EXPECT_EQ(resolved.fpu_strategy, FpuStrategy::kReduction);
}

TEST(Lc5SettingsTest, RejectsBatchAboveBackendMaximum) {
  OptionsParser parser;
  Settings::Populate(&parser);
  parser.SetUciOption("MinibatchSize", "129");
  Settings settings(parser.GetOptionsDict());
  BackendAttributes attrs{.has_mlh = true,
                          .has_wdl = true,
                          .runs_on_cpu = true,
                          .suggested_num_search_threads = 1,
                          .recommended_batch_size = 64,
                          .maximum_batch_size = 128};
  EXPECT_THROW(settings.Resolve(attrs), Exception);
}

TEST(Lc5SettingsTest, ExtractsDefaultTimingConfiguration) {
  OptionsParser parser;
  Settings::Populate(&parser);
  Settings settings(parser.GetOptionsDict());

  EXPECT_EQ(settings.time_management().move_overhead_ms, 200);
  EXPECT_FLOAT_EQ(settings.time_management().alphazero_time_pct, 3.0f);
}

TEST(Lc5SettingsTest, ExtractsCustomTimingConfiguration) {
  OptionsParser parser;
  Settings::Populate(&parser);
  parser.SetUciOption("MoveOverheadMs", "100");
  parser.SetUciOption("AlphaZeroTimePct", "25");
  Settings settings(parser.GetOptionsDict());

  EXPECT_EQ(settings.time_management().move_overhead_ms, 100);
  EXPECT_FLOAT_EQ(settings.time_management().alphazero_time_pct, 25.0f);
}

}  // namespace
}  // namespace lczero::lc5

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
