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

#include "cuda_runtime.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda.h>

#include "utils/exception.h"

namespace lczero {
namespace lc0ex {
namespace {

constexpr std::uint32_t kMagic = 0x1c0e;
constexpr std::uint32_t kFormat = 1;
constexpr std::size_t kNoIndex = std::numeric_limits<std::size_t>::max();

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

#define LC0EX_CUDA_CHECK(expression)                                    \
  do {                                                                  \
    const CUresult lc0ex_status = (expression);                         \
    if (lc0ex_status != CUDA_SUCCESS)                                   \
      ThrowCuda(lc0ex_status, #expression, __FILE__, __LINE__);         \
  } while (false)

void IgnoreCuda(CUresult status) { (void)status; }

void Require(bool condition, std::string message) {
  if (!condition) throw Exception(std::move(message));
}

std::string Missing(std::string_view kind, std::string_view name) {
  std::string message = "Missing ";
  message += kind;
  message += " \"";
  message += name;
  message += "\".";
  return message;
}

std::string Duplicate(std::string_view kind, std::string_view name) {
  std::string message = "Duplicate ";
  message += kind;
  message += " \"";
  message += name;
  message += "\".";
  return message;
}

std::uint64_t CheckedAdd(std::uint64_t lhs, std::uint64_t rhs,
                         std::string_view what) {
  Require(rhs <= std::numeric_limits<std::uint64_t>::max() - lhs,
          std::string(what) + " overflows.");
  return lhs + rhs;
}

std::uint64_t CheckedMultiply(std::uint64_t lhs, std::uint64_t rhs,
                              std::string_view what) {
  Require(lhs == 0 || rhs <= std::numeric_limits<std::uint64_t>::max() / lhs,
          std::string(what) + " overflows.");
  return lhs * rhs;
}

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
    Require(dimension != 0, "Buffer shape dimensions must be nonzero.");
    elements = CheckedMultiply(elements, dimension, "Buffer element count");
  }
  return CheckedMultiply(elements, ElementSize(buffer.data_type()),
                         "Buffer size");
}

std::array<unsigned int, 3> LaunchDimensions(
    const std::vector<std::uint32_t>& dimensions, std::string_view kind) {
  Require(!dimensions.empty() && dimensions.size() <= 3,
          std::string(kind) + " dimensions must contain one to three values.");

  std::array<unsigned int, 3> result = {1, 1, 1};
  for (std::size_t i = 0; i < dimensions.size(); ++i) {
    Require(dimensions[i] != 0,
            std::string(kind) + " dimensions must be nonzero.");
    result[i] = dimensions[i];
  }
  return result;
}

std::pair<CUdeviceptr, CUdeviceptr> AllocateDeviceMemory(
    std::uint64_t size_bytes, std::uint64_t alignment_bytes) {
  const auto allocation_size =
      CheckedAdd(size_bytes, alignment_bytes - 1, "Device allocation size");
  Require(allocation_size <= std::numeric_limits<std::size_t>::max(),
          "Device allocation size does not fit in size_t.");

  CUdeviceptr base = 0;
  LC0EX_CUDA_CHECK(cuMemAlloc(&base, static_cast<std::size_t>(allocation_size)));

  const auto base_address = static_cast<std::uint64_t>(base);
  Require(base_address <= std::numeric_limits<std::uint64_t>::max() -
                              alignment_bytes + 1,
          "Device allocation address overflows.");
  const auto aligned_address =
      (base_address + alignment_bytes - 1) & ~(alignment_bytes - 1);
  return {base, static_cast<CUdeviceptr>(aligned_address)};
}

struct CudaModule {
  std::string name;
  std::string data;
  CUmodule module = nullptr;
};

struct CudaAllocation {
  std::string name;
  std::uint64_t size_bytes = 0;
  std::uint64_t alignment_bytes = 0;
  pblczero::Allocation::Lifetime lifetime =
      pblczero::Allocation::LIFETIME_UNKNOWN;
  CUdeviceptr base = 0;
  CUdeviceptr address = 0;
};

struct CudaBufferPlan {
  std::size_t info_index = kNoIndex;
  bool is_symbol = false;
  std::size_t allocation = kNoIndex;
  std::uint64_t offset_bytes = 0;
  CUdeviceptr symbol = 0;
};

struct CudaKernel {
  std::string name;
  CUfunction function = nullptr;
  std::vector<pblczero::ParameterType> parameters;
};

struct CudaArgument {
  bool is_parameter = false;
  std::size_t index = kNoIndex;
  pblczero::ParameterType type =
      pblczero::ParameterType_PARAMETER_TYPE_UNKNOWN;
};

struct CudaNode {
  CUfunction function = nullptr;
  std::array<unsigned int, 3> grid = {1, 1, 1};
  std::array<unsigned int, 3> block = {1, 1, 1};
  unsigned int dynamic_shared_memory_bytes = 0;
  std::vector<CudaArgument> arguments;
};

class CudaExecutable;
class CudaExecutionContext;
class CudaInvocation;

class CudaProgram final : public Program {
 public:
  const ProgramInfo& GetInfo() const override { return info; }

  std::span<const ParameterInfo> GetParameters() const override {
    return {parameters.data(), parameters.size()};
  }

  CudaExecutable* owner = nullptr;
  ProgramInfo info;
  std::vector<ParameterInfo> parameters;
  std::unordered_map<std::string, std::size_t> parameter_indices;
  std::vector<CudaNode> nodes;
};

class CudaBuffer final : public Buffer {
 public:
  CudaBuffer(CudaExecutionContext* context, const BufferInfo* info,
             const CudaBufferPlan* plan, CUdeviceptr address)
      : context(context), info(info), plan(plan), address(address) {}

  const BufferInfo& GetInfo() const override { return *info; }
  void CopyFromHost(std::span<const std::byte> source) override;
  void CopyToHost(std::span<std::byte> destination) const override;

  void* ArgumentAddress() { return &address; }

  CudaExecutionContext* context;
  const BufferInfo* info;
  const CudaBufferPlan* plan;
  CUdeviceptr address;
};

class CudaParameter final : public Parameter {
 public:
  CudaParameter(CudaInvocation* invocation, std::size_t slot,
                pblczero::ParameterType type)
      : invocation(invocation), slot(slot), type(type) {}

  const ParameterInfo& GetInfo() const override;
  bool IsSet() const override { return is_set; }
  void Set(std::uint32_t value) override;
  void Set(const Buffer& buffer) override;
  void Reset() override;

  void* ArgumentAddress() {
    switch (type) {
      case pblczero::ParameterType_PARAMETER_TYPE_U32:
        return &u32;
      case pblczero::ParameterType_PARAMETER_TYPE_POINTER:
        return &pointer;
      case pblczero::ParameterType_PARAMETER_TYPE_UNKNOWN:
        break;
    }
    throw Exception("Unknown parameter type.");
  }

  CudaInvocation* invocation;
  std::size_t slot;
  pblczero::ParameterType type;
  bool is_set = false;
  std::uint32_t u32 = 0;
  CUdeviceptr pointer = 0;
};

class CudaExecutable final : public Executable {
 public:
  explicit CudaExecutable(CUdevice device) : device(device) {}
  ~CudaExecutable() override;

  const TargetInfo& GetTarget() const override { return target; }
  std::string_view GetMetadata() const override { return metadata; }
  std::span<const BufferInfo> GetBuffers() const override {
    return {buffer_infos.data(), buffer_infos.size()};
  }
  std::span<const ParameterInfo> GetParameters() const override {
    return {parameters.data(), parameters.size()};
  }
  std::span<const ProgramInfo> GetPrograms() const override {
    return {program_infos.data(), program_infos.size()};
  }

  const BufferInfo& FindBuffer(std::string_view name) const override {
    const auto iter = buffer_indices.find(std::string(name));
    Require(iter != buffer_indices.end(), Missing("buffer", name));
    return buffer_infos[iter->second];
  }

  const ParameterInfo& FindParameter(std::string_view name) const override {
    const auto iter = parameter_indices.find(std::string(name));
    Require(iter != parameter_indices.end(), Missing("parameter", name));
    return parameters[iter->second];
  }

  const Program& FindProgram(std::string_view name) const override {
    const auto iter = program_indices.find(std::string(name));
    Require(iter != program_indices.end(), Missing("program", name));
    return programs[iter->second];
  }

  std::unique_ptr<ExecutionContext> CreateContext() const override;

  void Initialize() {
    LC0EX_CUDA_CHECK(cuDevicePrimaryCtxRetain(&context, device));
    context_retained = true;
    LC0EX_CUDA_CHECK(cuCtxSetCurrent(context));
  }

  void SetCurrent() const { LC0EX_CUDA_CHECK(cuCtxSetCurrent(context)); }

  CUdevice device = 0;
  CUcontext context = nullptr;
  bool context_retained = false;

  TargetInfo target;
  std::string metadata;

  std::vector<CudaModule> modules;
  std::unordered_map<std::string, std::size_t> module_indices;

  std::vector<CudaAllocation> allocations;
  std::unordered_map<std::string, std::size_t> allocation_indices;

  std::vector<ParameterInfo> parameters;
  std::unordered_map<std::string, std::size_t> parameter_indices;

  std::vector<BufferInfo> buffer_infos;
  std::vector<CudaBufferPlan> buffer_plans;
  std::unordered_map<std::string, std::size_t> buffer_indices;

  std::vector<CudaKernel> kernels;
  std::unordered_map<std::string, std::size_t> kernel_indices;

  std::vector<CudaProgram> programs;
  std::vector<ProgramInfo> program_infos;
  std::unordered_map<std::string, std::size_t> program_indices;
};

class CudaExecutionContext final : public ExecutionContext {
 public:
  explicit CudaExecutionContext(CudaExecutable* executable)
      : executable(executable) {}
  ~CudaExecutionContext() override;

  Buffer& GetBuffer(const BufferInfo& info) override {
    const auto iter = executable->buffer_indices.find(info.name);
    Require(iter != executable->buffer_indices.end(),
            Missing("buffer", info.name));
    Require(&executable->buffer_infos[iter->second] == &info,
            "Buffer descriptor belongs to another executable.");
    return buffers[iter->second];
  }

  Buffer& GetBuffer(std::string_view name) override {
    const auto iter = executable->buffer_indices.find(std::string(name));
    Require(iter != executable->buffer_indices.end(), Missing("buffer", name));
    return buffers[iter->second];
  }

  std::unique_ptr<Invocation> CreateInvocation(
      const Program& program) override;
  std::unique_ptr<Invocation> CreateInvocation(
      std::string_view program_name) override;

  void Synchronize() override {
    executable->SetCurrent();
    LC0EX_CUDA_CHECK(cuStreamSynchronize(stream));
  }

  void Initialize() {
    executable->SetCurrent();
    LC0EX_CUDA_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

    allocation_bases.resize(executable->allocations.size());
    allocation_addresses.resize(executable->allocations.size());
    for (std::size_t i = 0; i < executable->allocations.size(); ++i) {
      const auto& allocation = executable->allocations[i];
      switch (allocation.lifetime) {
        case pblczero::Allocation::LIFETIME_PERSISTENT:
          allocation_addresses[i] = allocation.address;
          break;
        case pblczero::Allocation::LIFETIME_EXECUTION: {
          const auto memory = AllocateDeviceMemory(
              allocation.size_bytes, allocation.alignment_bytes);
          allocation_bases[i] = memory.first;
          allocation_addresses[i] = memory.second;
          break;
        }
        case pblczero::Allocation::LIFETIME_UNKNOWN:
          throw Exception("Unknown allocation lifetime.");
      }
    }

    buffers.reserve(executable->buffer_plans.size());
    for (std::size_t i = 0; i < executable->buffer_plans.size(); ++i) {
      const auto& plan = executable->buffer_plans[i];
      CUdeviceptr address = plan.symbol;
      if (!plan.is_symbol) {
        address = allocation_addresses[plan.allocation];
        Require(plan.offset_bytes <=
                    std::numeric_limits<CUdeviceptr>::max() - address,
                "Buffer address overflows.");
        address += plan.offset_bytes;
      }
      buffers.emplace_back(this, &executable->buffer_infos[plan.info_index],
                           &plan, address);
    }
  }

  CudaExecutable* executable;
  CUstream stream = nullptr;
  std::vector<CUdeviceptr> allocation_bases;
  std::vector<CUdeviceptr> allocation_addresses;
  std::vector<CudaBuffer> buffers;
};

class CudaInvocation final : public Invocation {
 public:
  CudaInvocation(CudaExecutionContext* context, const CudaProgram* program)
      : context(context), program(program) {
    parameters.reserve(program->parameters.size());
    for (std::size_t i = 0; i < program->parameters.size(); ++i) {
      parameters.emplace_back(this, i, program->parameters[i].type);
    }

    launch_arguments.resize(program->nodes.size());
    for (std::size_t node_index = 0; node_index < program->nodes.size();
         ++node_index) {
      const auto& node = program->nodes[node_index];
      auto& arguments = launch_arguments[node_index];
      arguments.reserve(node.arguments.size());
      for (const auto& argument : node.arguments) {
        if (argument.is_parameter) {
          arguments.push_back(parameters[argument.index].ArgumentAddress());
        } else {
          arguments.push_back(
              context->buffers[argument.index].ArgumentAddress());
        }
      }
    }
  }

  Parameter& GetParameter(std::string_view name) override {
    const auto iter = program->parameter_indices.find(std::string(name));
    Require(iter != program->parameter_indices.end(),
            Missing("parameter", name));
    return parameters[iter->second];
  }

  void ResetParameters() override {
    for (auto& parameter : parameters) parameter.Reset();
  }

  void Run() override {
    for (const auto& parameter : parameters) {
      Require(parameter.is_set,
              "Invocation parameter is not set: \"" +
                  parameter.GetInfo().name + "\".");
    }

    context->executable->SetCurrent();
    for (std::size_t i = 0; i < program->nodes.size(); ++i) {
      const auto& node = program->nodes[i];
      LC0EX_CUDA_CHECK(cuLaunchKernel(
          node.function, node.grid[0], node.grid[1], node.grid[2],
          node.block[0], node.block[1], node.block[2],
          node.dynamic_shared_memory_bytes, context->stream,
          launch_arguments[i].data(), nullptr));
    }
  }

  CudaExecutionContext* context;
  const CudaProgram* program;
  std::vector<CudaParameter> parameters;
  std::vector<std::vector<void*>> launch_arguments;
};

const ParameterInfo& CudaParameter::GetInfo() const {
  return invocation->program->parameters[slot];
}

void CudaParameter::Set(std::uint32_t value) {
  Require(type == pblczero::ParameterType_PARAMETER_TYPE_U32,
          "Parameter is not a u32 parameter.");
  u32 = value;
  is_set = true;
}

void CudaParameter::Set(const Buffer& buffer) {
  Require(type == pblczero::ParameterType_PARAMETER_TYPE_POINTER,
          "Parameter is not a pointer parameter.");
  const auto* cuda_buffer = dynamic_cast<const CudaBuffer*>(&buffer);
  Require(cuda_buffer != nullptr,
          "Pointer parameter buffer belongs to another runtime.");
  Require(cuda_buffer->context == invocation->context,
          "Pointer parameter buffer belongs to another execution context.");
  pointer = cuda_buffer->address;
  is_set = true;
}

void CudaParameter::Reset() {
  is_set = false;
  u32 = 0;
  pointer = 0;
}

void CudaBuffer::CopyFromHost(std::span<const std::byte> source) {
  Require(source.size() == info->size_bytes,
          "Host source size does not match buffer size.");
  context->executable->SetCurrent();
  LC0EX_CUDA_CHECK(cuMemcpyHtoDAsync(address, source.data(), source.size(),
                                     context->stream));
}

void CudaBuffer::CopyToHost(std::span<std::byte> destination) const {
  Require(destination.size() == info->size_bytes,
          "Host destination size does not match buffer size.");
  context->executable->SetCurrent();
  LC0EX_CUDA_CHECK(cuMemcpyDtoHAsync(destination.data(), address,
                                     destination.size(), context->stream));
}

CudaExecutable::~CudaExecutable() {
  if (!context_retained) return;
  if (cuCtxSetCurrent(context) == CUDA_SUCCESS) {
    for (auto& allocation : allocations) {
      if (allocation.base) IgnoreCuda(cuMemFree(allocation.base));
    }
    for (auto& module : modules) {
      if (module.module) IgnoreCuda(cuModuleUnload(module.module));
    }
  }
  IgnoreCuda(cuDevicePrimaryCtxRelease(device));
}

CudaExecutionContext::~CudaExecutionContext() {
  if (!executable || !executable->context_retained) return;
  if (cuCtxSetCurrent(executable->context) == CUDA_SUCCESS) {
    if (stream) IgnoreCuda(cuStreamSynchronize(stream));
    for (const auto allocation : allocation_bases) {
      if (allocation) IgnoreCuda(cuMemFree(allocation));
    }
    if (stream) IgnoreCuda(cuStreamDestroy(stream));
  }
}

std::unique_ptr<Invocation> CudaExecutionContext::CreateInvocation(
    const Program& program) {
  const auto* cuda_program = dynamic_cast<const CudaProgram*>(&program);
  Require(cuda_program != nullptr && cuda_program->owner == executable,
          "Program belongs to another executable.");
  return std::make_unique<CudaInvocation>(this, cuda_program);
}

std::unique_ptr<Invocation> CudaExecutionContext::CreateInvocation(
    std::string_view program_name) {
  const auto iter = executable->program_indices.find(std::string(program_name));
  Require(iter != executable->program_indices.end(),
          Missing("program", program_name));
  return std::make_unique<CudaInvocation>(this, &executable->programs[iter->second]);
}

std::unique_ptr<ExecutionContext> CudaExecutable::CreateContext() const {
  auto context = std::make_unique<CudaExecutionContext>(
      const_cast<CudaExecutable*>(this));
  context->Initialize();
  return context;
}

void BuildModules(CudaExecutable& executable,
                  const pblczero::NeuralExecutable& source) {
  executable.modules.reserve(source.binaries_size());
  executable.module_indices.reserve(source.binaries_size());
  for (const auto& binary : source.binaries()) {
    Require(binary.has_name() && !binary.name().empty(),
            "Every binary must have a nonempty name.");
    Require(binary.has_format() &&
                binary.format() == pblczero::Binary::FORMAT_CUBIN,
            "Only CUBIN binaries are supported.");
    Require(binary.has_data() && !binary.data().empty(),
            "Every binary must contain data.");

    const auto name = std::string(binary.name());
    Require(executable.module_indices.find(name) ==
                executable.module_indices.end(),
            Duplicate("binary", name));

    executable.modules.push_back({name, std::string(binary.data()), nullptr});
    auto& module = executable.modules.back();
    LC0EX_CUDA_CHECK(cuModuleLoadData(&module.module, module.data.data()));
    executable.module_indices.emplace(module.name,
                                      executable.modules.size() - 1);
  }
}

void BuildAllocations(CudaExecutable& executable,
                      const pblczero::NeuralExecutable& source) {
  executable.allocations.reserve(source.allocations_size());
  executable.allocation_indices.reserve(source.allocations_size());
  for (const auto& allocation : source.allocations()) {
    Require(allocation.has_name() && !allocation.name().empty(),
            "Every allocation must have a nonempty name.");
    Require(allocation.has_size_bytes() && allocation.size_bytes() != 0,
            "Every allocation must have a nonzero size.");
    Require(allocation.has_alignment_bytes() &&
                allocation.alignment_bytes() != 0 &&
                std::has_single_bit(allocation.alignment_bytes()),
            "Every allocation must have a nonzero power-of-two alignment.");
    Require(allocation.has_lifetime(),
            "Every allocation must have a lifetime.");
    Require(allocation.lifetime() ==
                pblczero::Allocation::LIFETIME_PERSISTENT ||
                allocation.lifetime() ==
                    pblczero::Allocation::LIFETIME_EXECUTION,
            "Unknown allocation lifetime.");

    const auto name = std::string(allocation.name());
    Require(executable.allocation_indices.find(name) ==
                executable.allocation_indices.end(),
            Duplicate("allocation", name));

    auto& plan = executable.allocations.emplace_back(
        CudaAllocation{name, allocation.size_bytes(), allocation.alignment_bytes(),
                       allocation.lifetime(), 0, 0});
    executable.allocation_indices.emplace(plan.name,
                                          executable.allocations.size() - 1);

    if (plan.lifetime == pblczero::Allocation::LIFETIME_PERSISTENT) {
      const auto memory =
          AllocateDeviceMemory(plan.size_bytes, plan.alignment_bytes);
      plan.base = memory.first;
      plan.address = memory.second;
    }
  }
}

void BuildParameters(CudaExecutable& executable,
                     const pblczero::NeuralExecutable& source) {
  executable.parameters.reserve(source.parameters_size());
  executable.parameter_indices.reserve(source.parameters_size());
  for (const auto& parameter : source.parameters()) {
    Require(parameter.has_name() && !parameter.name().empty(),
            "Every parameter must have a nonempty name.");
    Require(parameter.has_type() &&
                (parameter.type() ==
                     pblczero::ParameterType_PARAMETER_TYPE_U32 ||
                 parameter.type() ==
                     pblczero::ParameterType_PARAMETER_TYPE_POINTER),
            "Every parameter must have a known type.");

    const auto name = std::string(parameter.name());
    Require(executable.parameter_indices.find(name) ==
                executable.parameter_indices.end(),
            Duplicate("parameter", name));
    Require(executable.buffer_indices.find(name) ==
                executable.buffer_indices.end(),
            "Buffer and parameter names must not overlap: \"" + name +
                "\".");

    executable.parameters.push_back({name, parameter.type()});
    executable.parameter_indices.emplace(executable.parameters.back().name,
                                         executable.parameters.size() - 1);
  }
}

void BuildBuffers(CudaExecutable& executable,
                  const pblczero::NeuralExecutable& source) {
  executable.buffer_infos.reserve(source.buffers_size());
  executable.buffer_plans.reserve(source.buffers_size());
  executable.buffer_indices.reserve(source.buffers_size());
  for (const auto& buffer : source.buffers()) {
    Require(buffer.has_name() && !buffer.name().empty(),
            "Every buffer must have a nonempty name.");
    Require(buffer.has_data_type() &&
                buffer.data_type() != pblczero::Buffer::DATA_TYPE_UNKNOWN,
            "Every buffer must have a known data type.");

    const auto size_bytes = BufferSize(buffer);
    const bool has_allocation = buffer.has_allocation_block();
    const bool has_symbol = buffer.has_module_symbol();
    Require(has_allocation != has_symbol,
            "Every buffer must have exactly one backing location.");

    const auto name = std::string(buffer.name());
    Require(executable.buffer_indices.find(name) ==
                executable.buffer_indices.end(),
            Duplicate("buffer", name));
    Require(executable.parameter_indices.find(name) ==
                executable.parameter_indices.end(),
            "Buffer and parameter names must not overlap: \"" + name +
                "\".");

    BufferInfo info{
        name,
        buffer.data_type(),
        std::vector<std::uint64_t>(buffer.shape().begin(), buffer.shape().end()),
        size_bytes,
        pblczero::Allocation::LIFETIME_UNKNOWN};
    CudaBufferPlan plan;

    if (has_allocation) {
      const auto& block = buffer.allocation_block();
      Require(block.has_allocation() && !block.allocation().empty(),
              "Every allocation block must name an allocation.");
      Require(block.has_offset_bytes(),
              "Every allocation block must have an offset.");
      const auto allocation_iter = executable.allocation_indices.find(
          std::string(block.allocation()));
      Require(allocation_iter != executable.allocation_indices.end(),
              Missing("allocation", block.allocation()));

      plan.allocation = allocation_iter->second;
      plan.offset_bytes = block.offset_bytes();
      const auto& allocation = executable.allocations[plan.allocation];
      const auto end = CheckedAdd(plan.offset_bytes, size_bytes,
                                  "Buffer allocation range");
      Require(end <= allocation.size_bytes,
              "Buffer does not fit in its allocation.");
      info.lifetime = allocation.lifetime;
    } else {
      const auto& symbol = buffer.module_symbol();
      Require(symbol.has_binary() && !symbol.binary().empty(),
              "Every module symbol must name a binary.");
      Require(symbol.has_symbol() && !symbol.symbol().empty(),
              "Every module symbol must name a symbol.");
      const auto module_iter = executable.module_indices.find(
          std::string(symbol.binary()));
      Require(module_iter != executable.module_indices.end(),
              Missing("binary", symbol.binary()));

      std::size_t symbol_size = 0;
      LC0EX_CUDA_CHECK(cuModuleGetGlobal(
          &plan.symbol, &symbol_size,
          executable.modules[module_iter->second].module,
          std::string(symbol.symbol()).c_str()));
      Require(size_bytes <= symbol_size,
              "Buffer does not fit in its module symbol.");
      plan.is_symbol = true;
      info.lifetime = pblczero::Allocation::LIFETIME_PERSISTENT;
    }

    plan.info_index = executable.buffer_infos.size();
    executable.buffer_infos.push_back(std::move(info));
    executable.buffer_plans.push_back(std::move(plan));
    executable.buffer_indices.emplace(executable.buffer_infos.back().name,
                                      executable.buffer_infos.size() - 1);
  }
}

void BuildKernels(CudaExecutable& executable,
                  const pblczero::NeuralExecutable& source) {
  executable.kernels.reserve(source.kernels_size());
  executable.kernel_indices.reserve(source.kernels_size());
  for (const auto& kernel : source.kernels()) {
    Require(kernel.has_name() && !kernel.name().empty(),
            "Every kernel must have a nonempty name.");
    Require(kernel.has_binary() && !kernel.binary().empty(),
            "Every kernel must name a binary.");
    Require(kernel.has_function() && !kernel.function().empty(),
            "Every kernel must name a function.");

    const auto name = std::string(kernel.name());
    Require(executable.kernel_indices.find(name) ==
                executable.kernel_indices.end(),
            Duplicate("kernel", name));
    const auto module_iter = executable.module_indices.find(
        std::string(kernel.binary()));
    Require(module_iter != executable.module_indices.end(),
            Missing("binary", kernel.binary()));

    CUfunction function = nullptr;
    LC0EX_CUDA_CHECK(cuModuleGetFunction(
        &function, executable.modules[module_iter->second].module,
        std::string(kernel.function()).c_str()));

    CudaKernel plan;
    plan.name = name;
    plan.function = function;
    plan.parameters.assign(kernel.parameters().begin(), kernel.parameters().end());
    for (const auto parameter : plan.parameters) {
      Require(parameter == pblczero::ParameterType_PARAMETER_TYPE_U32 ||
                  parameter == pblczero::ParameterType_PARAMETER_TYPE_POINTER,
              "Unknown kernel parameter type.");
    }

    executable.kernels.push_back(std::move(plan));
    executable.kernel_indices.emplace(executable.kernels.back().name,
                                      executable.kernels.size() - 1);
  }
}

void BuildProgram(CudaExecutable& executable, const pblczero::Program& source,
                  CudaProgram* destination) {
  Require(source.has_name() && !source.name().empty(),
          "Every program must have a nonempty name.");
  destination->owner = &executable;
  destination->info.name = std::string(source.name());
  if (source.has_metadata()) destination->info.metadata = source.metadata();

  std::unordered_map<std::string, std::size_t> node_indices;
  node_indices.reserve(source.nodes_size());
  std::vector<CudaNode> source_nodes;
  source_nodes.reserve(source.nodes_size());
  std::vector<std::vector<std::size_t>> dependencies;
  dependencies.reserve(source.nodes_size());

  for (const auto& node : source.nodes()) {
    Require(node.has_name() && !node.name().empty(),
            "Every node must have a nonempty name.");
    Require(node.has_kernel() && !node.kernel().empty(),
            "Every node must name a kernel.");
    const auto node_name = std::string(node.name());
    Require(node_indices.find(node_name) == node_indices.end(),
            Duplicate("node", node_name));
    node_indices.emplace(node_name, source_nodes.size());

    const auto kernel_iter =
        executable.kernel_indices.find(std::string(node.kernel()));
    Require(kernel_iter != executable.kernel_indices.end(),
            Missing("kernel", node.kernel()));
    const auto& kernel = executable.kernels[kernel_iter->second];

    Require(node.arguments_size() == kernel.parameters.size(),
            "Node argument count does not match its kernel ABI.");
    Require(node.has_dynamic_shared_memory_bytes(),
            "Every node must specify dynamic shared memory.");

    CudaNode plan;
    plan.function = kernel.function;
    plan.grid = LaunchDimensions(node.grid(), "Grid");
    plan.block = LaunchDimensions(node.block(), "Block");
    plan.dynamic_shared_memory_bytes = node.dynamic_shared_memory_bytes();
    plan.arguments.reserve(node.arguments_size());

    for (std::size_t argument_index = 0;
         argument_index < node.arguments_size(); ++argument_index) {
      const auto argument_name = std::string(node.arguments(argument_index));
      const auto buffer_iter = executable.buffer_indices.find(argument_name);
      const auto parameter_iter = executable.parameter_indices.find(argument_name);
      Require((buffer_iter != executable.buffer_indices.end()) !=
                  (parameter_iter != executable.parameter_indices.end()),
              Missing("buffer or parameter", argument_name));

      CudaArgument argument;
      argument.type = kernel.parameters[argument_index];
      if (buffer_iter != executable.buffer_indices.end()) {
        Require(argument.type ==
                    pblczero::ParameterType_PARAMETER_TYPE_POINTER,
                "Buffer arguments must have pointer kernel parameters.");
        argument.index = buffer_iter->second;
      } else {
        const auto& global_parameter =
            executable.parameters[parameter_iter->second];
        Require(global_parameter.type == argument.type,
                "Node argument type does not match its parameter declaration.");

        const auto local_iter =
            destination->parameter_indices.find(global_parameter.name);
        if (local_iter == destination->parameter_indices.end()) {
          const auto local_index = destination->parameters.size();
          destination->parameters.push_back(global_parameter);
          destination->parameter_indices.emplace(
              destination->parameters.back().name, local_index);
          argument.index = local_index;
        } else {
          argument.index = local_iter->second;
        }
        argument.is_parameter = true;
      }
      plan.arguments.push_back(argument);
    }

    source_nodes.push_back(std::move(plan));
    dependencies.emplace_back();
  }

  for (std::size_t i = 0; i < source.nodes_size(); ++i) {
    std::unordered_set<std::string> dependency_names;
    dependency_names.reserve(source.nodes(i).dependencies_size());
    for (const auto& dependency : source.nodes(i).dependencies()) {
      Require(dependency_names.insert(dependency).second,
              "Node dependencies must not be duplicated.");
      const auto dependency_iter = node_indices.find(dependency);
      Require(dependency_iter != node_indices.end(),
              Missing("node dependency", dependency));
      Require(dependency_iter->second != i,
              "A node cannot depend on itself.");
      dependencies[i].push_back(dependency_iter->second);
    }
  }

  std::vector<std::size_t> indegree(source_nodes.size(), 0);
  std::vector<std::vector<std::size_t>> outgoing(source_nodes.size());
  for (std::size_t i = 0; i < dependencies.size(); ++i) {
    indegree[i] = dependencies[i].size();
    for (const auto dependency : dependencies[i]) {
      outgoing[dependency].push_back(i);
    }
  }

  std::vector<std::size_t> ready;
  ready.reserve(source_nodes.size());
  for (std::size_t i = 0; i < indegree.size(); ++i) {
    if (indegree[i] == 0) ready.push_back(i);
  }

  destination->nodes.reserve(source_nodes.size());
  for (std::size_t ready_index = 0; ready_index < ready.size(); ++ready_index) {
    const auto node = ready[ready_index];
    destination->nodes.push_back(std::move(source_nodes[node]));
    for (const auto dependent : outgoing[node]) {
      Require(indegree[dependent] != 0, "Invalid dependency graph.");
      if (--indegree[dependent] == 0) ready.push_back(dependent);
    }
  }
  Require(destination->nodes.size() == source_nodes.size(),
          "Program dependency graph contains a cycle.");
}

void BuildPrograms(CudaExecutable& executable,
                   const pblczero::NeuralExecutable& source) {
  executable.programs.reserve(source.programs_size());
  executable.program_infos.reserve(source.programs_size());
  executable.program_indices.reserve(source.programs_size());
  for (const auto& program : source.programs()) {
    Require(program.has_name() && !program.name().empty(),
            "Every program must have a nonempty name.");
    const auto name = std::string(program.name());
    Require(executable.program_indices.find(name) ==
                executable.program_indices.end(),
            Duplicate("program", name));

    CudaProgram plan;
    BuildProgram(executable, program, &plan);
    executable.program_infos.push_back(plan.info);
    executable.programs.push_back(std::move(plan));
    executable.program_indices.emplace(name, executable.programs.size() - 1);
  }
}

class CudaRuntime final : public Runtime {
 public:
  explicit CudaRuntime(int device_ordinal) {
    LC0EX_CUDA_CHECK(cuInit(0));
    LC0EX_CUDA_CHECK(cuDeviceGet(&device, device_ordinal));
  }

  std::unique_ptr<Executable> Load(
      const pblczero::NeuralExecutable& source) override {
    Require(source.has_magic() && source.magic() == kMagic,
            "Invalid lc0ex magic.");
    Require(source.has_format() && source.format() == kFormat,
            "Unsupported lc0ex format generation.");
    Require(source.has_target(), "Executable must specify a target.");
    Require(source.target().has_vendor() &&
                source.target().vendor() == pblczero::Target::VENDOR_NVIDIA,
            "CUDA runtime requires an NVIDIA target.");
    Require(source.target().has_architecture() &&
                !source.target().architecture().empty(),
            "Executable target must specify an architecture.");

    auto executable = std::make_unique<CudaExecutable>(device);
    executable->Initialize();
    executable->target.vendor = pblczero::Target::VENDOR_NVIDIA;
    executable->target.architecture =
        std::string(source.target().architecture());
    if (source.has_metadata()) executable->metadata = source.metadata();

    BuildModules(*executable, source);
    BuildAllocations(*executable, source);
    BuildParameters(*executable, source);
    BuildBuffers(*executable, source);
    BuildKernels(*executable, source);
    BuildPrograms(*executable, source);
    return executable;
  }

 private:
  CUdevice device = 0;
};

}  // namespace

std::unique_ptr<Runtime> CreateCudaRuntime(int device_ordinal) {
  return std::make_unique<CudaRuntime>(device_ordinal);
}

}  // namespace lc0ex
}  // namespace lczero
