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

#include "search/lc5/policies.h"

#include <algorithm>
#include <vector>

#include "search/lc5/policy_engine.h"
#include "search/lc5/puct.h"
#include "utils/exception.h"

namespace lczero::lc5 {
namespace {

const OptionId kPolicy{"lc5-policy", "Lc5Policy",
                       "Lc5 search policy: how node and edge statistics are "
                       "kept, backed up and used for selection."};

struct PolicyEntry {
  std::string_view name;
  void (*populate)(OptionsParser*);
  std::unique_ptr<VisitEngine> (*make)();
};

// Adding a policy is one row here. The first row is the default.
const PolicyEntry kPolicies[] = {
    {PuctAveraging::kName, &PuctAveraging::Params::Populate,
     []() -> std::unique_ptr<VisitEngine> {
       return std::make_unique<PolicyEngine<PuctAveraging>>();
     }},
};

}  // namespace

void PopulatePolicyOptions(OptionsParser* options) {
  std::vector<std::string> names;
  for (const auto& policy : kPolicies) names.emplace_back(policy.name);
  options->Add<ChoiceOption>(kPolicy, names) = names.front();
  // Policies may share a parameter set (and so its options), which is
  // registered once.
  std::vector<void (*)(OptionsParser*)> populated;
  for (const auto& policy : kPolicies) {
    if (std::find(populated.begin(), populated.end(), policy.populate) !=
        populated.end()) {
      continue;
    }
    populated.push_back(policy.populate);
    policy.populate(options);
  }
}

std::string PolicyName(const OptionsDict& options) {
  return options.Get<std::string>(kPolicy);
}

std::unique_ptr<VisitEngine> MakeVisitEngine(std::string_view name) {
  for (const auto& policy : kPolicies) {
    if (policy.name == name) return policy.make();
  }
  throw Exception("Unknown Lc5Policy: " + std::string(name));
}

}  // namespace lczero::lc5
