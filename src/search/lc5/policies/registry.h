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

#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "search/lc5/visit_engine.h"
#include "utils/optionsdict.h"
#include "utils/optionsparser.h"

namespace lczero::lc5 {

template <typename P>
class PolicyEngine;

// A singleton that keeps every registered lc5 policy. Policies register
// themselves with REGISTER_LC5_POLICY; the one with the highest priority is
// the default.
class PolicyRegistry {
 public:
  struct Entry {
    std::string_view name;
    int priority;
    void (*populate)(OptionsParser*);
    std::unique_ptr<VisitEngine> (*make)();
  };

  static PolicyRegistry* Get();
  void Add(const Entry& entry);
  // Sorted by priority (highest first), then by name.
  const std::vector<Entry>& entries() const { return entries_; }

  template <typename P>
  struct Register {
    explicit Register(int priority) {
      PolicyRegistry::Get()->Add({P::kName, priority, &P::Params::Populate,
                                  []() -> std::unique_ptr<VisitEngine> {
                                    return std::make_unique<PolicyEngine<P>>();
                                  }});
    }
  };

 private:
  PolicyRegistry() = default;

  std::vector<Entry> entries_;
};

// Registers policy P (with P::kName and P::Params::Populate) as an Lc5Policy
// choice. The registering file must include policy_engine.h.
#define REGISTER_LC5_POLICY(P, priority)                             \
  namespace {                                                        \
  [[maybe_unused]] static ::lczero::lc5::PolicyRegistry::Register<P> \
      reg_lc5_policy_##P(priority);                                  \
  }

// Adds the Lc5Policy option and every policy's own options.
void PopulatePolicyOptions(OptionsParser* options);
// The policy chosen by the Lc5Policy option.
std::string PolicyName(const OptionsDict& options);
// A fresh engine with an empty graph for the named policy; throws if unknown.
std::unique_ptr<VisitEngine> MakeVisitEngine(std::string_view name);

}  // namespace lczero::lc5
