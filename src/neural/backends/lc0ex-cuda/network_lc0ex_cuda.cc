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

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "absl/algorithm/container.h"
#include "neural/backend.h"
#include "neural/loader.h"
#include "neural/onnx/converter.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "network_fingerprint.h"
#include "runtime/lc0ex_cuda.h"
#include "utils/exception.h"
#include "utils/logging.h"

namespace lczero {
namespace {

constexpr std::string_view kBackendName = "lc0ex-cuda";

pblczero::NeuralExecutable LoadExecutableFile(const std::string& path) {
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file) {
    throw Exception("Cannot read lc0ex executable from " + path + ".");
  }

  std::string serialized((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
  if (file.bad()) {
    throw Exception("Error while reading lc0ex executable from " + path +
                    ".");
  }

  pblczero::NeuralExecutable executable;
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

bool HasMatchingShape(const lc0ex::BufferInfo& buffer,
                      const pblczero::TensorProto& initializer) {
  if (initializer.dims_size() != buffer.shape.size()) {
    return false;
  }
  for (std::size_t i = 0; i < initializer.dims_size(); ++i) {
    const auto dimension = initializer.dims(i);
    if (dimension < 0 ||
        static_cast<std::uint64_t>(dimension) != buffer.shape[i]) {
      return false;
    }
  }
  return true;
}

void ValidateInitializer(const lc0ex::BufferInfo& buffer,
                         const pblczero::TensorProto& initializer) {
  if (static_cast<int>(buffer.data_type) !=
      static_cast<int>(initializer.data_type())) {
    throw Exception("Data type mismatch for lc0ex buffer '" + buffer.name +
                    "'.");
  }
  if (!HasMatchingShape(buffer, initializer)) {
    throw Exception("Shape mismatch for lc0ex buffer '" + buffer.name +
                    "'.");
  }
  if (static_cast<std::uint64_t>(initializer.raw_data().size()) !=
      buffer.size_bytes) {
    throw Exception("Size mismatch for lc0ex buffer '" + buffer.name +
                    "'.");
  }
}

WeightsToOnnxConverterOptions MakeConverterOptions(
    const OptionsDict& backend_options) {
  WeightsToOnnxConverterOptions converter_options;
  converter_options.opset = backend_options.GetOrDefault<int>("opset", 17);
  converter_options.ir = backend_options.GetOrDefault<int>("ir", -1);
  converter_options.alt_mish =
      backend_options.GetOrDefault<bool>("alt_mish", false);
  converter_options.alt_layernorm =
      backend_options.GetOrDefault<bool>("alt_layernorm", false);
  converter_options.no_shape =
      backend_options.GetOrDefault<bool>("no_shape", false);
  converter_options.policy_head =
      backend_options.GetOrDefault<std::string>("policy_head", "vanilla");
  converter_options.value_head =
      backend_options.GetOrDefault<std::string>("value_head", "winner");
  converter_options.no_wdl_softmax = true;

  std::string datatype;
  if (backend_options.Exists<std::string>("datatype")) {
    datatype = backend_options.Get<std::string>("datatype");
  } else {
    const bool fp16 = backend_options.GetOrDefault<bool>("fp16", true);
    datatype = fp16 ? "f16" : "f32";
  }
  converter_options.data_type =
      WeightsToOnnxConverterOptions::StringToDataType(datatype);
  return converter_options;
}

void UploadWeights(const WeightsFile& weights, lc0ex::Executable& executable,
                   const OptionsDict& backend_options) {
  std::optional<WeightsFile> converted_weights;
  if (!weights.has_onnx_model()) {
    CERR << "Converting weights to ONNX first.";
    converted_weights =
        ConvertWeightsToOnnx(weights, MakeConverterOptions(backend_options));
  }

  const auto& onnx_weights = converted_weights ? *converted_weights : weights;

  pblczero::ModelProto onnx;
  onnx.ParseFromString(onnx_weights.onnx_model().model());

  std::unordered_map<std::string, const pblczero::TensorProto*>
      initializers_by_name;
  initializers_by_name.reserve(onnx.graph().initializer_size());
  std::string duplicate_initializer;
  const bool unique_initializers = absl::c_all_of(
      onnx.graph().initializer(), [&](const auto& initializer) {
        const auto name = std::string(initializer.name());
        if (!initializers_by_name.emplace(name, &initializer).second) {
          duplicate_initializer = name;
          return false;
        }
        return true;
      });
  if (!unique_initializers) {
    throw Exception("The ONNX model contains duplicate initializer '" +
                    duplicate_initializer + "'.");
  }

  std::string missing_buffer;
  if (absl::c_any_of(executable.GetBuffers(), [&](const auto& buffer) {
        if (initializers_by_name.find(buffer.name) ==
            initializers_by_name.end()) {
          missing_buffer = buffer.name;
          return true;
        }
        return false;
      })) {
    throw Exception("The lc0ex buffer '" + missing_buffer +
                    "' has no corresponding ONNX initializer.");
  }

  CERR << "Uploading ONNX initializers to the lc0ex runtime.";
  for (const auto& initializer : onnx.graph().initializer()) {
    const auto* buffer = executable.FindBuffer(initializer.name());
    if (!buffer) {
      CERR << "WARNING: ONNX initializer '" << initializer.name()
           << "' has no corresponding lc0ex buffer.";
      continue;
    }

    ValidateInitializer(*buffer, initializer);
    const auto source = std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(initializer.raw_data().data()),
        initializer.raw_data().size());
    executable.GetBuffer(*buffer).CopyFromHost(source);
  }
}

BackendAttributes MakeBackendAttributes(const WeightsFile& weights) {
  const auto& format = weights.format().network_format();
  return {
      .has_mlh =
          format.moves_left() != pblczero::NetworkFormat::MOVES_LEFT_NONE,
      .has_wdl = format.output() == pblczero::NetworkFormat::OUTPUT_WDL,
      .runs_on_cpu = false,
      .suggested_num_search_threads = 2,
      .recommended_batch_size = 256,
      .maximum_batch_size = 1024,
  };
}

class Lc0exCudaBackend;

class Lc0exCudaBackendComputation final : public BackendComputation {
 public:
  explicit Lc0exCudaBackendComputation(Lc0exCudaBackend* backend);

  size_t UsedBatchSize() const override { return entries_.size(); }

  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    entries_.push_back({pos, result});
    return ENQUEUED_FOR_EVAL;
  }

  void ComputeBlocking() override;

 private:
  struct Entry {
    EvalPosition position;
    EvalResultPtr result;
  };

  Lc0exCudaBackend* backend_;
  std::vector<Entry> entries_;
};

class Lc0exCudaBackend final : public Backend {
 public:
  Lc0exCudaBackend(const WeightsFile& weights, const OptionsDict& options,
                   const OptionsDict& backend_options)
      : attributes_(MakeBackendAttributes(weights)),
        concurrency_(backend_options.GetOrDefault<int>("concurrency", 1)),
        backend_options_(
            options.Get<std::string>(SharedBackendParams::kBackendOptionsId)),
        weights_path_(options.Get<std::string>(SharedBackendParams::kWeightsId)) {
    UpdateConfiguration(options);

    const std::string lc0ex_path = backend_options.Get<std::string>("lc0ex");
    if (lc0ex_path.empty()) {
      throw Exception("The lc0ex-cuda backend requires an lc0ex path.");
    }

    const auto executable_proto = LoadExecutableFile(lc0ex_path);
    CheckNetworkFingerprint(weights, executable_proto);

    const int gpu = backend_options.GetOrDefault<int>("gpu", 0);
    runtime_ = lc0ex::CreateLc0exCudaRuntime(gpu);
    executable_ = runtime_->Load(executable_proto);
    UploadWeights(weights, *executable_, backend_options);
  }

  BackendAttributes GetAttributes() const override { return attributes_; }

  std::unique_ptr<BackendComputation> CreateComputation() override {
    return std::make_unique<Lc0exCudaBackendComputation>(this);
  }

  UpdateConfigurationResult UpdateConfiguration(
      const OptionsDict& options) override {
    Backend::UpdateConfiguration(options);
    if (backend_options_ !=
        options.Get<std::string>(SharedBackendParams::kBackendOptionsId)) {
      return NEED_RESTART;
    }
    if (weights_path_ !=
        options.Get<std::string>(SharedBackendParams::kWeightsId)) {
      return NEED_RESTART;
    }
    return UPDATE_OK;
  }

  std::unique_ptr<lc0ex::Execution> CreateExecution(
      std::size_t /*batch_size*/) {
    const auto programs = executable_->GetPrograms();
    const auto* program = executable_->FindProgram(programs.front().name);
    if (!program) {
      throw Exception("The lc0ex executable's first program could not be found.");
    }
    return executable_->CreateExecution(*program);
  }

 private:
  const BackendAttributes attributes_;
  // Reserved for future execution policy. It must not affect search threads.
  const int concurrency_;
  const std::string backend_options_;
  const std::string weights_path_;
  std::unique_ptr<lc0ex::Runtime> runtime_;
  std::unique_ptr<lc0ex::Executable> executable_;

  friend class Lc0exCudaBackendComputation;
};

Lc0exCudaBackendComputation::Lc0exCudaBackendComputation(
    Lc0exCudaBackend* backend)
    : backend_(backend) {
  entries_.reserve(backend_->GetAttributes().maximum_batch_size);
}

void Lc0exCudaBackendComputation::ComputeBlocking() {
  [[maybe_unused]] const auto execution =
      backend_->CreateExecution(entries_.size());

  // The executable tensor ABI is not wired to the search result contract yet.
  // Preserve the legacy wrapper's current zero-logit behavior meanwhile.
  for (const auto& entry : entries_) {
    if (entry.result.q) *entry.result.q = 0.0f;
    if (entry.result.d) *entry.result.d = 0.0f;
    if (entry.result.m) *entry.result.m = 0.0f;
    if (!entry.result.p.empty()) {
      const float value = 1.0f / entry.result.p.size();
      for (float& policy : entry.result.p) policy = value;
    }
  }
}

class Lc0exCudaBackendFactory final : public BackendFactory {
 public:
  int GetPriority() const override { return 1; }
  std::string_view GetName() const override { return kBackendName; }

  std::unique_ptr<Backend> Create(const OptionsDict& options) override {
    OptionsDict backend_options;
    backend_options.AddSubdictFromString(
        options.Get<std::string>(SharedBackendParams::kBackendOptionsId));

    const std::string weights_path =
        options.Get<std::string>(SharedBackendParams::kWeightsId);
    const std::optional<WeightsFile> weights = LoadWeights(weights_path);
    if (!weights) {
      throw Exception("The lc0ex-cuda backend requires a network file.");
    }

    auto backend = std::make_unique<Lc0exCudaBackend>(*weights, options,
                                                       backend_options);
    backend_options.CheckAllOptionsRead(std::string(kBackendName));
    return backend;
  }
};

REGISTER_BACKEND(Lc0exCudaBackendFactory)

}  // namespace
}  // namespace lczero
