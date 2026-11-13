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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

#include "search/lc5/graph/node.h"

namespace lczero::lc5 {

// Serialization of a materialized Node<P> for PolicyEngine's NodeStore: its
// terminal kind, the policy's node statistics, its legal edges and the
// statistics of its tried edges. Reservations (in_flight) and the version
// belong to the live graph and are not stored.
//
// Layout (native byte order; serialized nodes are not meant to move between
// machines): SerializedNodeHeader, P::NodeStats, LegalEdge[edges],
// P::EdgeStats[tried] (where `tried` is deduced from the remaining byte count).

namespace internal {

struct SerializedNodeHeader {
  TerminalKind terminal;
  uint8_t edges;
};
static_assert(sizeof(SerializedNodeHeader) == 2);
static_assert(std::is_trivially_copyable_v<SerializedNodeHeader>);

// A policy whose statistics are not trivially copyable needs a serializer of
// its own.
template <typename P>
constexpr bool kTriviallySerializable =
    std::is_trivially_copyable_v<typename P::NodeStats> &&
    std::is_trivially_copyable_v<typename P::EdgeStats> &&
    std::is_trivially_copyable_v<LegalEdge>;

}  // namespace internal

template <typename P>
std::vector<uint8_t> SerializeNode(const Node<P>& node) {
  static_assert(internal::kTriviallySerializable<P>);
  assert(node.edges.size() <= std::numeric_limits<uint8_t>::max());
  assert(node.tried.size() <= node.edges.size());
  using NodeStats = typename P::NodeStats;
  using EdgeStats = typename P::EdgeStats;
  const internal::SerializedNodeHeader header{
      .terminal = node.terminal,
      .edges = static_cast<uint8_t>(node.edges.size())};
  std::vector<uint8_t> data(sizeof(header) + sizeof(NodeStats) +
                            node.edges.size() * sizeof(LegalEdge) +
                            node.tried.size() * sizeof(EdgeStats));
  uint8_t* out = data.data();
  const auto write = [&out](const void* src, size_t size) {
    if (size > 0) std::memcpy(out, src, size);
    out += size;
  };
  write(&header, sizeof(header));
  write(&node.stats, sizeof(NodeStats));
  write(node.edges.data(), node.edges.size() * sizeof(LegalEdge));
  for (const auto& tried : node.tried) write(&tried.stats, sizeof(EdgeStats));
  return data;
}

// The node from SerializeNode<P>, with no reservations.
template <typename P>
Node<P> DeserializeNode(std::span<const uint8_t> data) {
  static_assert(internal::kTriviallySerializable<P>);
  using NodeStats = typename P::NodeStats;
  using EdgeStats = typename P::EdgeStats;
  assert(data.size() >= sizeof(internal::SerializedNodeHeader));
  internal::SerializedNodeHeader header;
  std::memcpy(&header, data.data(), sizeof(header));
  const size_t base_size =
      sizeof(header) + sizeof(NodeStats) + header.edges * sizeof(LegalEdge);
  assert(data.size() >= base_size);
  assert((data.size() - base_size) % sizeof(EdgeStats) == 0);
  const size_t tried_count = (data.size() - base_size) / sizeof(EdgeStats);
  assert(tried_count <= header.edges);
  const uint8_t* in = data.data() + sizeof(header);
  const auto read = [&in](void* dst, size_t size) {
    if (size > 0) std::memcpy(dst, in, size);
    in += size;
  };
  Node<P> node{.terminal = header.terminal};
  read(&node.stats, sizeof(NodeStats));
  node.edges.resize(header.edges);
  read(node.edges.data(), header.edges * sizeof(LegalEdge));
  node.tried.resize(tried_count);
  for (auto& tried : node.tried) read(&tried.stats, sizeof(EdgeStats));
  return node;
}

}  // namespace lczero::lc5
