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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include "search/lc5/graph/node.h"

namespace lczero::lc5 {

// The record of a materialized node in a NodeStore: its terminal kind, the
// policy's node statistics, its legal edges and the statistics of its tried
// edges. Reservations (in_flight) and the version belong to the live graph and
// are not stored.
//
// Layout (native byte order; records are not meant to move between
// machines): RecordHeader, P::NodeStats, LegalEdge[edges],
// P::EdgeStats[tried].

inline constexpr uint8_t kNodeRecordVersion = 1;

// FNV-1a of the policy name, so a record from another policy is rejected.
constexpr uint64_t PolicyTag(std::string_view name) {
  uint64_t hash = 14695981039346656037ull;
  for (const char c : name) {
    hash = (hash ^ static_cast<uint8_t>(c)) * 1099511628211ull;
  }
  return hash;
}

struct RecordHeader {
  uint64_t policy_tag;
  uint8_t version;
  TerminalKind terminal;
  uint16_t edges;
  uint16_t tried;
  uint16_t reserved = 0;
};
static_assert(std::is_trivially_copyable_v<RecordHeader>);

// A policy whose statistics are not trivially copyable needs an encoder of
// its own.
template <typename P>
constexpr bool kTriviallyRecordable =
    std::is_trivially_copyable_v<typename P::NodeStats> &&
    std::is_trivially_copyable_v<typename P::EdgeStats> &&
    std::is_trivially_copyable_v<LegalEdge>;

template <typename P>
std::vector<uint8_t> EncodeNode(const Node<P>& node) {
  static_assert(kTriviallyRecordable<P>);
  using NodeStats = typename P::NodeStats;
  using EdgeStats = typename P::EdgeStats;
  const RecordHeader header{.policy_tag = PolicyTag(P::kName),
                            .version = kNodeRecordVersion,
                            .terminal = node.terminal,
                            .edges = static_cast<uint16_t>(node.edges.size()),
                            .tried = static_cast<uint16_t>(node.tried.size())};
  std::vector<uint8_t> record(sizeof(header) + sizeof(NodeStats) +
                              node.edges.size() * sizeof(LegalEdge) +
                              node.tried.size() * sizeof(EdgeStats));
  uint8_t* out = record.data();
  const auto write = [&out](const void* data, size_t size) {
    if (size > 0) std::memcpy(out, data, size);
    out += size;
  };
  write(&header, sizeof(header));
  write(&node.stats, sizeof(NodeStats));
  write(node.edges.data(), node.edges.size() * sizeof(LegalEdge));
  for (const auto& tried : node.tried) write(&tried.stats, sizeof(EdgeStats));
  return record;
}

// The node of a record from EncodeNode<P>, with no reservations; nullopt if
// the record is malformed or from another policy or format version.
template <typename P>
std::optional<Node<P>> DecodeNode(std::span<const uint8_t> record) {
  static_assert(kTriviallyRecordable<P>);
  using NodeStats = typename P::NodeStats;
  using EdgeStats = typename P::EdgeStats;
  RecordHeader header;
  if (record.size() < sizeof(header)) return std::nullopt;
  std::memcpy(&header, record.data(), sizeof(header));
  if (header.policy_tag != PolicyTag(P::kName) ||
      header.version != kNodeRecordVersion || header.tried > header.edges ||
      header.terminal > TerminalKind::kInsufficientMaterial ||
      (header.terminal != TerminalKind::kNonTerminal && header.edges > 0) ||
      record.size() != sizeof(header) + sizeof(NodeStats) +
                           header.edges * sizeof(LegalEdge) +
                           header.tried * sizeof(EdgeStats)) {
    return std::nullopt;
  }
  const uint8_t* in = record.data() + sizeof(header);
  const auto read = [&in](void* data, size_t size) {
    if (size > 0) std::memcpy(data, in, size);
    in += size;
  };
  Node<P> node{.terminal = header.terminal};
  read(&node.stats, sizeof(NodeStats));
  node.edges.resize(header.edges);
  read(node.edges.data(), header.edges * sizeof(LegalEdge));
  // Selection relies on this order.
  if (!std::ranges::is_sorted(node.edges, PreferOnTie)) return std::nullopt;
  node.tried.resize(header.tried);
  for (auto& tried : node.tried) read(&tried.stats, sizeof(EdgeStats));
  return node;
}

}  // namespace lczero::lc5
