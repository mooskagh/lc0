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

#include "lc0ex_cuda.h"

#include <cuda.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "utils/exception.h"

namespace lczero {
namespace lc0ex {
namespace {

constexpr std::uint32_t kMagic = 0x1c0e;
constexpr std::uint32_t kFormat = 1;

[[noreturn]] void ThrowCuda(CUresult status, const char* expression,
                            const char* file, int line) {
  const char* name = nullptr;
  const char* description = nullptr;
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &description);

  std::string message = expression;
  message += " failed at ";
  message += file;
  message += ":";
  message += std::to_string(line);
  message += ": ";
  message += name ? name : "unknown CUDA error";
  if (description) {
    message += " (";
    message += description;
    message += ")";
  }
  throw Exception(message);
}

#define LC0EX_CUDA_CHECK(expression)                            \
  do {                                                          \
    const CUresult lc0ex_status = (expression);                 \
    if (lc0ex_status != CUDA_SUCCESS)                           \
      ThrowCuda(lc0ex_status, #expression, __FILE__, __LINE__); \
  } while (false)

void IgnoreCuda(CUresult status) { (void)status; }

std::size_t ElementSize(pblczero::Buffer::DataType type) {
  switch (type) {
    case pblczero::Buffer::DATA_TYPE_F32:
      return sizeof(float);
    case pblczero::Buffer::DATA_TYPE_U8:
      return sizeof(std::uint8_t);
    case pblczero::Buffer::DATA_TYPE_F16:
      return sizeof(std::uint16_t);
    case pblczero::Buffer::DATA_TYPE_U64:
      return sizeof(std::uint64_t);
    case pblczero::Buffer::DATA_TYPE_BF16:
      return sizeof(std::uint16_t);
    case pblczero::Buffer::DATA_TYPE_UNKNOWN:
      break;
  }
  throw Exception("Unsupported or unknown buffer data type.");
}

std::uint64_t BufferSize(const pblczero::Buffer& buffer) {
  std::uint64_t elements = 1;
  for (const auto dimension : buffer.shape()) {
    elements *= dimension;
  }
  return elements * ElementSize(buffer.data_type());
}

BufferInfo MakeBufferInfo(const pblczero::Buffer& buffer) {
  return {
      std::string(buffer.name()), buffer.data_type(),
      std::vector<std::uint64_t>(buffer.shape().begin(), buffer.shape().end()),
      BufferSize(buffer)};
}

std::array<unsigned int, 3> LaunchDimensions(
    const std::vector<std::uint32_t>& dimensions) {
  std::array<unsigned int, 3> result = {1, 1, 1};
  for (std::size_t i = 0; i < dimensions.size(); ++i) {
    result[i] = dimensions[i];
  }
  return result;
}

std::pair<CUdeviceptr, CUdeviceptr> AllocateDeviceMemory(
    std::uint64_t size_bytes, std::uint64_t alignment_bytes) {
  const auto allocation_size = size_bytes + alignment_bytes - 1;

  CUdeviceptr base = 0;
  LC0EX_CUDA_CHECK(
      cuMemAlloc(&base, static_cast<std::size_t>(allocation_size)));

  const auto base_address = static_cast<std::uint64_t>(base);
  const auto aligned_address =
      (base_address + alignment_bytes - 1) & ~(alignment_bytes - 1);
  return {base, static_cast<CUdeviceptr>(aligned_address)};
}

struct Lc0exCudaAllocation {
  std::uint64_t size_bytes_ = 0;
  std::uint64_t alignment_bytes_ = 0;
  CUdeviceptr base_ = 0;
  CUdeviceptr address_ = 0;
};

struct Lc0exCudaBufferPlan {
  std::uint64_t offset_bytes_ = 0;
};

struct Lc0exCudaKernel {
  CUfunction function_ = nullptr;
  std::vector<pblczero::ParameterType> parameters_;
};

struct Lc0exCudaArgument {
  bool is_parameter_ = false;
  bool is_symbol_ = false;
  std::size_t index_ = 0;
  pblczero::Node::Argument::AllocationLocation::AllocationKind kind_ =
      pblczero::Node::Argument::AllocationLocation::ALLOCATION_UNKNOWN;
  std::uint64_t offset_ = 0;
  CUdeviceptr symbol_ = 0;
};

struct Lc0exCudaNode {
  CUfunction function_ = nullptr;
  std::array<unsigned int, 3> grid_ = {1, 1, 1};
  std::array<unsigned int, 3> block_ = {1, 1, 1};
  unsigned int dynamic_shared_memory_bytes_ = 0;
  std::vector<Lc0exCudaArgument> arguments_;
};

class Lc0exCudaExecutable;
class Lc0exCudaExecution;

class Lc0exCudaProgram final : public Program {
 public:
  const ProgramInfo& GetInfo() const override { return info_; }

  std::span<const BufferInfo> GetBuffers() const override {
    return {buffer_infos_.data(), buffer_infos_.size()};
  }

  const BufferInfo* FindBuffer(std::string_view name) const override {
    const auto iter = buffer_indices_.find(std::string(name));
    return iter == buffer_indices_.end() ? nullptr
                                         : &buffer_infos_[iter->second];
  }

  std::span<const ParameterInfo> GetParameters() const override {
    return {parameters_.data(), parameters_.size()};
  }

  ProgramInfo info_;
  std::vector<ParameterInfo> parameters_;
  std::unordered_map<std::string, std::size_t> parameter_indices_;
  Lc0exCudaAllocation execution_allocation_;
  std::vector<BufferInfo> buffer_infos_;
  std::vector<Lc0exCudaBufferPlan> buffer_plans_;
  std::unordered_map<std::string, std::size_t> buffer_indices_;
  std::vector<Lc0exCudaNode> nodes_;
};

class Lc0exCudaBuffer final : public Buffer {
 public:
  Lc0exCudaBuffer(Lc0exCudaExecutable* executable, const BufferInfo* info,
                  CUdeviceptr address)
      : executable_(executable), info_(info), address_(address) {}

  const BufferInfo& GetInfo() const override { return *info_; }
  void CopyFromHost(std::span<const std::byte> source) override;
  void CopyToHost(std::span<std::byte> destination) const override;

  Lc0exCudaExecutable* executable_;
  const BufferInfo* info_;
  CUdeviceptr address_;
};

class Lc0exCudaParameter final : public Parameter {
 public:
  Lc0exCudaParameter(Lc0exCudaExecution* execution, std::size_t slot,
                     pblczero::ParameterType type)
      : execution_(execution), slot_(slot), type_(type) {}

  const ParameterInfo& GetInfo() const override;
  bool IsSet() const override { return is_set_; }
  void Set(std::uint32_t value) override;
  void Set(const Buffer& buffer) override;
  void Reset() override;

  void* ArgumentAddress() {
    switch (type_) {
      case pblczero::ParameterType_PARAMETER_TYPE_U32:
        return &u32_;
      case pblczero::ParameterType_PARAMETER_TYPE_POINTER:
        return &pointer_;
      case pblczero::ParameterType_PARAMETER_TYPE_UNKNOWN:
        break;
    }
    throw Exception("Unknown parameter type.");
  }

  Lc0exCudaExecution* execution_;
  std::size_t slot_;
  pblczero::ParameterType type_;
  bool is_set_ = false;
  std::uint32_t u32_ = 0;
  CUdeviceptr pointer_ = 0;
};

class Lc0exCudaExecutable final : public Executable {
 public:
  explicit Lc0exCudaExecutable(CUdevice device) : device_(device) {}
  ~Lc0exCudaExecutable() override;

  const TargetInfo& GetTarget() const override { return target_; }
  std::string_view GetMetadata() const override { return metadata_; }
  std::span<const BufferInfo> GetBuffers() const override {
    return {buffer_infos_.data(), buffer_infos_.size()};
  }
  std::span<const ParameterInfo> GetParameters() const override {
    return {parameters_.data(), parameters_.size()};
  }
  std::span<const ProgramInfo> GetPrograms() const override {
    return {program_infos_.data(), program_infos_.size()};
  }

  const BufferInfo* FindBuffer(std::string_view name) const override {
    const auto iter = buffer_indices_.find(std::string(name));
    return iter == buffer_indices_.end() ? nullptr
                                         : &buffer_infos_[iter->second];
  }

  const ParameterInfo* FindParameter(std::string_view name) const override {
    const auto iter = parameter_indices_.find(std::string(name));
    return iter == parameter_indices_.end() ? nullptr
                                            : &parameters_[iter->second];
  }

  const Program* FindProgram(std::string_view name) const override {
    const auto iter = program_indices_.find(std::string(name));
    return iter == program_indices_.end() ? nullptr : &programs_[iter->second];
  }

  Buffer& GetBuffer(const BufferInfo& info) override;

  std::unique_ptr<Execution> CreateExecution(const Program& program) override;

  void Initialize() {
    LC0EX_CUDA_CHECK(cuDevicePrimaryCtxRetain(&context_, device_));
    context_retained_ = true;
    LC0EX_CUDA_CHECK(cuCtxSetCurrent(context_));
  }

  void SetCurrent() const { LC0EX_CUDA_CHECK(cuCtxSetCurrent(context_)); }

  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
  bool context_retained_ = false;

  TargetInfo target_;
  std::string metadata_;

  std::vector<CUmodule> modules_;

  Lc0exCudaAllocation persistent_allocation_;

  std::vector<ParameterInfo> parameters_;
  std::unordered_map<std::string, std::size_t> parameter_indices_;

  std::vector<BufferInfo> buffer_infos_;
  std::vector<Lc0exCudaBufferPlan> buffer_plans_;
  std::vector<std::unique_ptr<Lc0exCudaBuffer>> persistent_buffers_;
  std::unordered_map<std::string, std::size_t> buffer_indices_;

  std::vector<Lc0exCudaKernel> kernels_;

  std::vector<Lc0exCudaProgram> programs_;
  std::vector<ProgramInfo> program_infos_;
  std::unordered_map<std::string, std::size_t> program_indices_;
};

class Lc0exCudaExecution final : public Execution {
 public:
  Lc0exCudaExecution(Lc0exCudaExecutable* executable,
                     const Lc0exCudaProgram* program)
      : executable_(executable), program_(program) {}
  ~Lc0exCudaExecution() override;

  Buffer& GetBuffer(const BufferInfo& info) override {
    return *buffers_[program_->buffer_indices_.find(info.name)->second];
  }

  Parameter& GetParameter(std::string_view name) override {
    return parameters_[program_->parameter_indices_.find(std::string(name))
                           ->second];
  }

  void ResetParameters() override {
    for (auto& parameter : parameters_) parameter.Reset();
  }

  void Initialize() {
    executable_->SetCurrent();
    LC0EX_CUDA_CHECK(cuStreamCreate(&stream_, CU_STREAM_DEFAULT));

    if (program_->execution_allocation_.size_bytes_ != 0) {
      const auto memory = AllocateDeviceMemory(
          program_->execution_allocation_.size_bytes_,
          program_->execution_allocation_.alignment_bytes_);
      execution_base_ = memory.first;
      execution_address_ = memory.second;
    }

    buffers_.resize(program_->buffer_plans_.size());
    for (std::size_t i = 0; i < program_->buffer_plans_.size(); ++i) {
      const auto& plan = program_->buffer_plans_[i];
      auto address = execution_address_;
      address += plan.offset_bytes_;
      buffers_[i] = std::make_unique<Lc0exCudaBuffer>(
          executable_, &program_->buffer_infos_[i], address);
    }

    parameters_.reserve(program_->parameters_.size());
    for (std::size_t i = 0; i < program_->parameters_.size(); ++i) {
      parameters_.emplace_back(this, i, program_->parameters_[i].type);
    }

    launch_arguments_.resize(program_->nodes_.size());
    allocation_argument_values_.resize(program_->nodes_.size());
    for (std::size_t node_index = 0; node_index < program_->nodes_.size();
         ++node_index) {
      const auto& node = program_->nodes_[node_index];
      auto& arguments = launch_arguments_[node_index];
      auto& allocation_values = allocation_argument_values_[node_index];
      arguments.resize(node.arguments_.size());
      allocation_values.resize(node.arguments_.size());
      for (std::size_t argument_index = 0;
           argument_index < node.arguments_.size(); ++argument_index) {
        const auto& argument = node.arguments_[argument_index];
        if (argument.is_parameter_) {
          arguments[argument_index] =
              parameters_[argument.index_].ArgumentAddress();
        } else {
          auto& value = allocation_values[argument_index];
          if (argument.is_symbol_) {
            value = argument.symbol_;
          } else {
            value = argument.kind_ ==
                            pblczero::Node::Argument::AllocationLocation::
                                ALLOCATION_PERSISTENT
                        ? executable_->persistent_allocation_.address_
                        : execution_address_;
            value += argument.offset_;
          }
          arguments[argument_index] = &value;
        }
      }
    }
  }

  void Run() override {
    executable_->SetCurrent();
    in_flight_ = true;
    for (std::size_t i = 0; i < program_->nodes_.size(); ++i) {
      const auto& node = program_->nodes_[i];
      LC0EX_CUDA_CHECK(
          cuLaunchKernel(node.function_, node.grid_[0], node.grid_[1],
                         node.grid_[2], node.block_[0], node.block_[1],
                         node.block_[2], node.dynamic_shared_memory_bytes_,
                         stream_, launch_arguments_[i].data(), nullptr));
    }
  }

  void Synchronize() override {
    if (!in_flight_) return;
    executable_->SetCurrent();
    LC0EX_CUDA_CHECK(cuStreamSynchronize(stream_));
    in_flight_ = false;
  }

  Lc0exCudaExecutable* executable_;
  const Lc0exCudaProgram* program_;
  CUstream stream_ = nullptr;
  bool in_flight_ = false;
  CUdeviceptr execution_base_ = 0;
  CUdeviceptr execution_address_ = 0;
  std::vector<std::unique_ptr<Lc0exCudaBuffer>> buffers_;
  std::vector<Lc0exCudaParameter> parameters_;
  std::vector<std::vector<void*>> launch_arguments_;
  std::vector<std::vector<CUdeviceptr>> allocation_argument_values_;
};

const ParameterInfo& Lc0exCudaParameter::GetInfo() const {
  return execution_->program_->parameters_[slot_];
}

void Lc0exCudaParameter::Set(std::uint32_t value) {
  u32_ = value;
  is_set_ = true;
}

void Lc0exCudaParameter::Set(const Buffer& buffer) {
  pointer_ = static_cast<const Lc0exCudaBuffer&>(buffer).address_;
  is_set_ = true;
}

void Lc0exCudaParameter::Reset() {
  is_set_ = false;
  u32_ = 0;
  pointer_ = 0;
}

void Lc0exCudaBuffer::CopyFromHost(std::span<const std::byte> source) {
  executable_->SetCurrent();
  LC0EX_CUDA_CHECK(cuMemcpyHtoD(address_, source.data(), source.size()));
}

void Lc0exCudaBuffer::CopyToHost(std::span<std::byte> destination) const {
  executable_->SetCurrent();
  LC0EX_CUDA_CHECK(
      cuMemcpyDtoH(destination.data(), address_, destination.size()));
}

Lc0exCudaExecutable::~Lc0exCudaExecutable() {
  if (!context_retained_) return;
  if (cuCtxSetCurrent(context_) == CUDA_SUCCESS) {
    if (persistent_allocation_.base_) {
      IgnoreCuda(cuMemFree(persistent_allocation_.base_));
    }
    for (auto& module : modules_) {
      if (module) IgnoreCuda(cuModuleUnload(module));
    }
  }
  IgnoreCuda(cuDevicePrimaryCtxRelease(device_));
}

Lc0exCudaExecution::~Lc0exCudaExecution() {
  if (!executable_ || !executable_->context_retained_) return;
  if (cuCtxSetCurrent(executable_->context_) == CUDA_SUCCESS) {
    if (stream_) IgnoreCuda(cuStreamSynchronize(stream_));
    if (execution_base_) IgnoreCuda(cuMemFree(execution_base_));
    if (stream_) IgnoreCuda(cuStreamDestroy(stream_));
  }
}

Buffer& Lc0exCudaExecutable::GetBuffer(const BufferInfo& info) {
  return *persistent_buffers_[buffer_indices_.find(info.name)->second];
}

std::unique_ptr<Execution> Lc0exCudaExecutable::CreateExecution(
    const Program& program) {
  const auto* cuda_program = static_cast<const Lc0exCudaProgram*>(&program);
  auto execution = std::make_unique<Lc0exCudaExecution>(this, cuda_program);
  execution->Initialize();
  return execution;
}

void BuildModules(Lc0exCudaExecutable& executable,
                  const pblczero::NeuralExecutable& source) {
  executable.modules_.reserve(source.binaries_size());
  for (const auto& binary : source.binaries()) {
    CUmodule module = nullptr;
    LC0EX_CUDA_CHECK(cuModuleLoadData(&module, binary.data().data()));
    executable.modules_.push_back(module);
  }
}

void BuildAllocation(Lc0exCudaAllocation& destination,
                     const pblczero::Allocation& source) {
  destination.size_bytes_ = source.size_bytes();
  destination.alignment_bytes_ = source.alignment_bytes();
}

void BuildPersistentAllocation(Lc0exCudaExecutable& executable,
                               const pblczero::NeuralExecutable& source) {
  if (source.has_persistent_allocation()) {
    BuildAllocation(executable.persistent_allocation_,
                    source.persistent_allocation());
    const auto memory = AllocateDeviceMemory(
        executable.persistent_allocation_.size_bytes_,
        executable.persistent_allocation_.alignment_bytes_);
    executable.persistent_allocation_.base_ = memory.first;
    executable.persistent_allocation_.address_ = memory.second;
  }
}

void BuildParameters(Lc0exCudaExecutable& executable,
                     const pblczero::NeuralExecutable& source) {
  executable.parameters_.reserve(source.parameters_size());
  executable.parameter_indices_.reserve(source.parameters_size());
  for (const auto& parameter : source.parameters()) {
    const auto name = std::string(parameter.name());
    executable.parameters_.push_back({name, parameter.type()});
    executable.parameter_indices_.emplace(executable.parameters_.back().name,
                                          executable.parameters_.size() - 1);
  }
}

void BuildPersistentBuffers(Lc0exCudaExecutable& executable,
                            const pblczero::NeuralExecutable& source) {
  executable.buffer_infos_.reserve(source.buffers_size());
  executable.buffer_plans_.reserve(source.buffers_size());
  executable.buffer_indices_.reserve(source.buffers_size());
  for (const auto& buffer : source.buffers()) {
    auto info = MakeBufferInfo(buffer);
    Lc0exCudaBufferPlan plan{buffer.offset()};

    executable.buffer_infos_.push_back(std::move(info));
    executable.buffer_plans_.push_back(std::move(plan));
    executable.buffer_indices_.emplace(executable.buffer_infos_.back().name,
                                       executable.buffer_infos_.size() - 1);
  }

  executable.persistent_buffers_.resize(executable.buffer_plans_.size());
  for (std::size_t i = 0; i < executable.buffer_plans_.size(); ++i) {
    const auto& plan = executable.buffer_plans_[i];
    auto address = executable.persistent_allocation_.address_;
    address += plan.offset_bytes_;
    executable.persistent_buffers_[i] = std::make_unique<Lc0exCudaBuffer>(
        &executable, &executable.buffer_infos_[i], address);
  }
}

void BuildProgramBuffers(Lc0exCudaProgram& program,
                         const pblczero::Program& source) {
  program.buffer_infos_.reserve(source.buffers_size());
  program.buffer_plans_.reserve(source.buffers_size());
  program.buffer_indices_.reserve(source.buffers_size());
  for (const auto& buffer : source.buffers()) {
    auto info = MakeBufferInfo(buffer);

    program.buffer_infos_.push_back(std::move(info));
    program.buffer_plans_.push_back({buffer.offset()});
    program.buffer_indices_.emplace(program.buffer_infos_.back().name,
                                    program.buffer_infos_.size() - 1);
  }
}

void BuildKernels(Lc0exCudaExecutable& executable,
                  const pblczero::NeuralExecutable& source) {
  executable.kernels_.reserve(source.kernels_size());
  for (const auto& kernel : source.kernels()) {
    CUfunction function = nullptr;
    LC0EX_CUDA_CHECK(
        cuModuleGetFunction(&function, executable.modules_[kernel.binary_idx()],
                            std::string(kernel.function()).c_str()));

    Lc0exCudaKernel plan;
    plan.function_ = function;
    plan.parameters_.assign(kernel.parameters().begin(),
                            kernel.parameters().end());

    executable.kernels_.push_back(std::move(plan));
  }
}

void BuildProgram(Lc0exCudaExecutable& executable,
                  const pblczero::Program& source,
                  Lc0exCudaProgram* destination) {
  destination->info_.name = std::string(source.name());
  destination->info_.metadata = source.metadata();
  if (source.has_execution_allocation()) {
    BuildAllocation(destination->execution_allocation_,
                    source.execution_allocation());
  }
  BuildProgramBuffers(*destination, source);

  std::vector<Lc0exCudaNode> source_nodes;
  source_nodes.reserve(source.nodes_size());
  std::vector<std::size_t> indegree(source.nodes_size(), 0);
  std::vector<std::vector<std::size_t>> outgoing(source.nodes_size());

  for (std::size_t node_index = 0; node_index < source.nodes_size();
       ++node_index) {
    const auto& node = source.nodes(node_index);
    const auto& kernel = executable.kernels_[node.kernel_idx()];

    Lc0exCudaNode plan;
    plan.function_ = kernel.function_;
    plan.grid_ = LaunchDimensions(node.grid());
    plan.block_ = LaunchDimensions(node.block());
    plan.dynamic_shared_memory_bytes_ = node.dynamic_shared_memory_bytes();
    plan.arguments_.reserve(node.arguments_size());

    for (std::size_t argument_index = 0; argument_index < node.arguments_size();
         ++argument_index) {
      const auto& source_argument = node.arguments(argument_index);
      const bool is_parameter = source_argument.has_parameter_name();

      if (is_parameter) {
        const auto parameter_iter = executable.parameter_indices_.find(
            std::string(source_argument.parameter_name()));
        const auto& global_parameter =
            executable.parameters_[parameter_iter->second];

        Lc0exCudaArgument argument;
        argument.is_parameter_ = true;
        const auto local_iter =
            destination->parameter_indices_.find(global_parameter.name);
        if (local_iter == destination->parameter_indices_.end()) {
          const auto local_index = destination->parameters_.size();
          destination->parameters_.push_back(global_parameter);
          destination->parameter_indices_.emplace(
              destination->parameters_.back().name, local_index);
          argument.index_ = local_index;
        } else {
          argument.index_ = local_iter->second;
        }
        plan.arguments_.push_back(argument);
      } else if (source_argument.has_allocation()) {
        const auto& location = source_argument.allocation();
        plan.arguments_.push_back({
            false,
            false,
            0,
            location.kind(),
            location.offset(),
            0,
        });
      } else {
        const auto& symbol = source_argument.symbol();
        Lc0exCudaArgument argument;
        argument.is_symbol_ = true;
        std::size_t symbol_size = 0;
        const std::string symbol_name(symbol.symbol_name());
        LC0EX_CUDA_CHECK(cuModuleGetGlobal(
            &argument.symbol_, &symbol_size,
            executable.modules_[symbol.binary_idx()], symbol_name.c_str()));
        plan.arguments_.push_back(argument);
      }
    }

    source_nodes.push_back(std::move(plan));
    for (const auto dependency : node.dependencies()) {
      ++indegree[node_index];
      outgoing[dependency].push_back(node_index);
    }
  }

  std::vector<std::size_t> ready;
  ready.reserve(source_nodes.size());
  for (std::size_t i = 0; i < indegree.size(); ++i) {
    if (indegree[i] == 0) ready.push_back(i);
  }

  destination->nodes_.reserve(source_nodes.size());
  for (std::size_t ready_index = 0; ready_index < ready.size(); ++ready_index) {
    const auto node = ready[ready_index];
    destination->nodes_.push_back(std::move(source_nodes[node]));
    for (const auto dependent : outgoing[node]) {
      if (--indegree[dependent] == 0) ready.push_back(dependent);
    }
  }
}

void BuildPrograms(Lc0exCudaExecutable& executable,
                   const pblczero::NeuralExecutable& source) {
  executable.programs_.reserve(source.programs_size());
  executable.program_infos_.reserve(source.programs_size());
  executable.program_indices_.reserve(source.programs_size());
  for (const auto& program : source.programs()) {
    const auto name = std::string(program.name());

    Lc0exCudaProgram plan;
    BuildProgram(executable, program, &plan);
    executable.program_infos_.push_back(plan.info_);
    executable.programs_.push_back(std::move(plan));
    executable.program_indices_.emplace(name, executable.programs_.size() - 1);
  }
}

class Lc0exCudaRuntime final : public Runtime {
 public:
  explicit Lc0exCudaRuntime(int device_ordinal) {
    LC0EX_CUDA_CHECK(cuInit(0));
    LC0EX_CUDA_CHECK(cuDeviceGet(&device_, device_ordinal));
  }

  std::unique_ptr<Executable> Load(
      const pblczero::NeuralExecutable& source) override {
    if (source.magic() != kMagic) throw Exception("Invalid lc0ex magic.");
    if (source.format() != kFormat) {
      throw Exception("Unsupported lc0ex format generation.");
    }

    auto executable = std::make_unique<Lc0exCudaExecutable>(device_);
    executable->Initialize();
    executable->target_.vendor = pblczero::Target::VENDOR_NVIDIA;
    executable->target_.architecture =
        std::string(source.target().architecture());
    executable->metadata_ = source.metadata();

    BuildModules(*executable, source);
    BuildPersistentAllocation(*executable, source);
    BuildParameters(*executable, source);
    BuildPersistentBuffers(*executable, source);
    BuildKernels(*executable, source);
    BuildPrograms(*executable, source);
    return executable;
  }

 private:
  CUdevice device_ = 0;
};

}  // namespace

std::unique_ptr<Runtime> CreateLc0exCudaRuntime(int device_ordinal) {
  return std::make_unique<Lc0exCudaRuntime>(device_ordinal);
}

}  // namespace lc0ex
}  // namespace lczero
