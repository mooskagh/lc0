/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2024 The LCZero Authors

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

#include <functional>
#include <string>

#include "neural/network.h"
#include "neural/register.h"

namespace lczero {

class NetworkAsBackendFactory : public BackendFactory {
 public:
  using FactoryFunc = std::function<std::unique_ptr<Network>(
      const std::optional<WeightsFile>&, const OptionsDict&)>;

  NetworkAsBackendFactory(const std::string& name, FactoryFunc factory,
                          int priority = 0);

  int GetPriority() const override { return priority_; }
  std::string_view GetName() const override { return name_; }
  std::unique_ptr<Backend> Create(const OptionsDict&) override;

 private:
  std::string name_;
  FactoryFunc factory_;
  int priority_;
};

}  // namespace lczero