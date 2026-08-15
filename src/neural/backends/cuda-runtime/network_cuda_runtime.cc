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

#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "neural/factory.h"
#include "network_fingerprint.h"
#include "runtime/cuda_runtime.h"
#include "utils/exception.h"

namespace lczero {
namespace {

std::string ReadExecutableFile(const std::string& path) {
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file) {
    throw Exception("Cannot read lc0ex executable from " + path + ".");
  }

  std::string result((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
  if (file.bad()) {
    throw Exception("Error while reading lc0ex executable from " + path + ".");
  }
  return result;
}

pblczero::NeuralExecutable LoadExecutableFile(const std::string& path) {
  pblczero::NeuralExecutable executable;
  const auto serialized = ReadExecutableFile(path);
  executable.ParseFromString(serialized);

  return executable;
}

void CheckNetworkFingerprint(const WeightsFile& weights,
                             const pblczero::NeuralExecutable& executable) {
  pblczero::Net executable_fingerprint;
  executable_fingerprint.ParseFromString(executable.metadata());

  const auto network_fingerprint = lc0ex::BuildNetworkFingerprint(weights);
  if (network_fingerprint.OutputAsString() !=
      executable_fingerprint.OutputAsString()) {
    throw Exception(
        "The lc0ex executable was created for a different network architecture.");
  }
}

class CudaRuntimeNetwork;

class CudaRuntimeNetworkComputation final : public NetworkComputation {
 public:
  explicit CudaRuntimeNetworkComputation(CudaRuntimeNetwork* network);

  void AddInput(InputPlanes&& input) override {
    inputs_.push_back(std::move(input));
  }

  void ComputeBlocking() override {}

  int GetBatchSize() const override {
    return static_cast<int>(inputs_.size());
  }

  float GetQVal(int /*sample*/) const override { return 0.0f; }
  float GetDVal(int /*sample*/) const override { return 0.0f; }
  float GetPVal(int /*sample*/, int /*move_id*/) const override { return 0.0f; }
  float GetMVal(int /*sample*/) const override { return 0.0f; }

 private:
  std::unique_ptr<lc0ex::Execution> execution_;
  std::vector<InputPlanes> inputs_;
};

class CudaRuntimeNetwork final : public Network {
 public:
  CudaRuntimeNetwork(const WeightsFile& weights, const OptionsDict& options)
      : capabilities_{weights.format().network_format().input(),
                      weights.format().network_format().output(),
                      weights.format().network_format().moves_left()},
        concurrency_(options.GetOrDefault<int>("concurrency", 1)) {
    const std::string lc0ex_path = options.Get<std::string>("lc0ex");
    if (lc0ex_path.empty()) {
      throw Exception("The cuda-runtime backend requires an lc0ex path.");
    }

    const auto executable_proto = LoadExecutableFile(lc0ex_path);
    CheckNetworkFingerprint(weights, executable_proto);

    runtime_ = lc0ex::CreateCudaRuntime();
    executable_ = runtime_->Load(executable_proto);
  }

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<CudaRuntimeNetworkComputation>(this);
  }

  std::unique_ptr<lc0ex::Execution> CreateExecution() {
    const auto programs = executable_->GetPrograms();
    const auto& program = executable_->FindProgram(programs.front().name);
    return executable_->CreateExecution(program);
  }

 private:
  const NetworkCapabilities capabilities_;
  // Reserved for future execution policy. It must not affect search threads.
  const int concurrency_;
  std::unique_ptr<lc0ex::Runtime> runtime_;
  std::unique_ptr<lc0ex::Executable> executable_;

  friend class CudaRuntimeNetworkComputation;
};

CudaRuntimeNetworkComputation::CudaRuntimeNetworkComputation(
    CudaRuntimeNetwork* network)
    : execution_(network->CreateExecution()) {}

std::unique_ptr<Network> MakeCudaRuntimeNetwork(
    const std::optional<WeightsFile>& weights, const OptionsDict& options) {
  if (!weights) {
    throw Exception("The cuda-runtime backend requires a network file.");
  }
  return std::make_unique<CudaRuntimeNetwork>(*weights, options);
}

REGISTER_NETWORK("cuda-runtime", MakeCudaRuntimeNetwork, 1)

}  // namespace
}  // namespace lczero
