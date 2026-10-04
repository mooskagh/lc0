#include "search/lc5/settings.h"

#include <gtest/gtest.h>

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

TEST(Lc5SettingsTest, ComputesBudgetFromActiveClock) {
  OptionsParser parser;
  Settings::Populate(&parser);
  Settings settings(parser.GetOptionsDict());
  GoParams params{.wtime = 10200, .btime = 5200};

  EXPECT_EQ(settings.GetTimeBudget(params, false), 1200);
  EXPECT_EQ(settings.GetTimeBudget(params, true), 600);
}

TEST(Lc5SettingsTest, TimeBudgetOptionsAreConfigurable) {
  OptionsParser parser;
  Settings::Populate(&parser);
  parser.SetUciOption("MoveOverheadMs", "100");
  parser.SetUciOption("AlphaZeroTimePct", "25");
  Settings settings(parser.GetOptionsDict());

  EXPECT_EQ(settings.GetTimeBudget(GoParams{.wtime = 4100}, false), 1000);
  EXPECT_EQ(settings.GetTimeBudget(GoParams{.wtime = 50}, false), 0);
}

TEST(Lc5SettingsTest, IncludesOnlyActiveSideIncrement) {
  OptionsParser parser;
  Settings::Populate(&parser);
  Settings settings(parser.GetOptionsDict());
  GoParams params{.wtime = 10200, .btime = 5200, .winc = 100, .binc = 200};

  EXPECT_EQ(settings.GetTimeBudget(params, false), 1288);
  EXPECT_EQ(settings.GetTimeBudget(params, true), 776);
  params.winc = -100;
  EXPECT_EQ(settings.GetTimeBudget(params, false), 1200);
}

TEST(Lc5SettingsTest, HonorsShorterMovesToGoHorizon) {
  OptionsParser parser;
  Settings::Populate(&parser);
  Settings settings(parser.GetOptionsDict());
  GoParams params{.wtime = 4200, .winc = 100, .movestogo = 4};

  EXPECT_EQ(settings.GetTimeBudget(params, false), 1075);
  params.movestogo = 1;
  EXPECT_EQ(settings.GetTimeBudget(params, false), 4000);
  for (int moves : {0, -1, 100}) {
    params.movestogo = moves;
    EXPECT_EQ(settings.GetTimeBudget(params, false), 568);
  }
}

TEST(Lc5SettingsTest, CapsBudgetAndPreservesOverhead) {
  OptionsParser parser;
  Settings::Populate(&parser);
  Settings settings(parser.GetOptionsDict());

  EXPECT_EQ(
      settings.GetTimeBudget(GoParams{.wtime = 300, .winc = 10000}, false),
      100);
  EXPECT_EQ(settings.GetTimeBudget(GoParams{.wtime = 201, .winc = 80}, false),
            1);
  EXPECT_EQ(settings.GetTimeBudget(GoParams{.wtime = 201}, false), 1);
  EXPECT_EQ(settings.GetTimeBudget(GoParams{.wtime = 200, .winc = 80}, false),
            0);
  EXPECT_EQ(settings.GetTimeBudget(GoParams{.wtime = 100, .winc = 80}, false),
            0);
  parser.SetUciOption("AlphaZeroTimePct", "0");
  EXPECT_EQ(Settings(parser.GetOptionsDict())
                .GetTimeBudget(GoParams{.wtime = 1000}, false),
            1);
}

TEST(Lc5SettingsTest, IncrementMaintainsBudgetOverLongGame) {
  OptionsParser parser;
  Settings::Populate(&parser);
  Settings settings(parser.GetOptionsDict());
  GoParams params{.wtime = 8000, .winc = 80};

  for (int move = 0; move < 200; ++move) {
    const auto budget = settings.GetTimeBudget(params, false);
    ASSERT_TRUE(budget.has_value());
    EXPECT_GE(*budget, 80) << "move " << move;
    EXPECT_LE(*budget, *params.wtime - 200);
    *params.wtime += *params.winc - *budget;
  }
}

TEST(Lc5SettingsTest, TimeBudgetDoesNotLimitUnclockedSearches) {
  OptionsParser parser;
  Settings::Populate(&parser);
  Settings settings(parser.GetOptionsDict());

  EXPECT_EQ(settings.GetTimeBudget(GoParams{}, false), std::nullopt);
  EXPECT_EQ(
      settings.GetTimeBudget(GoParams{.wtime = 1000, .infinite = true}, false),
      std::nullopt);
  EXPECT_EQ(
      settings.GetTimeBudget(GoParams{.btime = 1000, .ponder = true}, true),
      std::nullopt);
}

}  // namespace
}  // namespace lczero::lc5

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
