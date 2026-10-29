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
  EXPECT_GE(resolved.max_active_visits, 256);
  EXPECT_EQ(resolved.history_key_length, 7);
  EXPECT_EQ(resolved.fpu_strategy, FpuStrategy::kReduction);
}

TEST(Lc5SettingsTest, CapsDefaultMaxActiveVisits) {
  OptionsParser parser;
  Settings::Populate(&parser);
  parser.SetUciOption("EvalThreads", "128");
  Settings settings(parser.GetOptionsDict());
  BackendAttributes attrs{.has_mlh = true,
                          .has_wdl = true,
                          .runs_on_cpu = false,
                          .suggested_num_search_threads = 2,
                          .recommended_batch_size = 1024,
                          .maximum_batch_size = 1024};
  EXPECT_EQ(settings.Resolve(attrs).max_active_visits, kMaxActiveVisitsLimit);
  EXPECT_EQ(kMaxActiveVisitsLimit, 65535);
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
