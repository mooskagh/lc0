/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors

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

#include "lc2/mcts/node.h"

#include <absl/types/span.h>
#include <assert.h>

#include <algorithm>
#include <cstring>
#include <string_view>
namespace lc2 {
#if 0

namespace {

template <typename Out, typename In, size_t InSize>
void UnpackVectorFromHeadAndTail(const std::array<In, InSize>& head_in,
                                 std::string_view* tail_buffer,
                                 size_t total_count, std::vector<Out>* out) {
  out->resize(total_count);
  const auto convert_fn = [](const In& v) -> Out {
    return static_cast<Out>(v);
  };
  // Copy head.
  std::transform(head_in.begin(),
                 head_in.begin() + std::min(head_in.size(), total_count),
                 out->begin(), convert_fn);
  if (head_in.size() >= total_count) return;
  // Copy tail.
  assert(tail_buffer != nullptr);
  const size_t size_to_copy = (total_count - head_in.size()) * sizeof(In);
  assert(tail_buffer->size() >= size_to_copy);
  if constexpr (std::is_same_v<In, Out>) {
    std::memcpy(&(*out)[InSize], tail_buffer->data(), size_to_copy);
  } else {
    std::transform(
        reinterpret_cast<const In*>(tail_buffer->data()),
        reinterpret_cast<const In*>(tail_buffer->data() + size_to_copy),
        out->begin() + InSize, convert_fn);
  }
  tail_buffer->remove_prefix(size_to_copy);
}

template <typename Out, typename In, size_t InSize>
void UnpackVectorFromHead(const std::array<In, InSize>& head_in,
                          size_t total_count, std::vector<Out>* out) {
  UnpackVectorFromHeadAndTail(head_in, nullptr, std::min(total_count, InSize),
                              out);
}

template <typename T, size_t Size>
size_t BytesNotFitArray(const std::array<T, Size>&, size_t count) {
  if (count <= Size) return 0;
  return (count - Size) * sizeof(T);
}

template <typename In, typename Out, size_t OutSize>
void PackVectorIntoHeadAndTail(const std::vector<In>& in,
                               std::array<Out, OutSize>* out,
                               std::string* tail) {
  const auto convert_fn = [](const In& v) -> Out {
    return static_cast<Out>(v);
  };
  // Copy head.
  std::transform(in.begin(), in.begin() + std::min(in.size(), OutSize),
                 out->begin(), convert_fn);
  if (in.size() <= OutSize) return;
  // Copy tail.
  assert(tail != nullptr);
  const size_t size_to_write = (in.size() - OutSize) * sizeof(Out);
  if constexpr (std::is_same_v<In, Out>) {
    tail->append(reinterpret_cast<const char*>(&in[OutSize]), size_to_write);
  } else {
    std::for_each(in.begin() + OutSize, in.end(), [&](const auto val) {
      const auto value_to_write = static_cast<Out>(val);
      tail->append(reinterpret_cast<const char*>(&value_to_write),
                   sizeof(value_to_write));
    });
  }
}

}  // namespace

void UnpackedNode::UnpackFromHead(const NodeHead& head) {
  UnpackVectorFromHead(head.edge_p, head.num_edges, &p);
  UnpackVectorFromHead(head.moves, head.num_edges, &moves);
  UnpackVectorFromHead(head.edge_n, head.num_edges, &n);
  UnpackVectorFromHead(head.edge_q, head.num_edges, &q);
}

void UnpackedNode::UnpackFromHeadAndTail(const NodeHead& head,
                                         const NodeTail& tail) {
  std::string_view tail_buffer(tail.data(), tail.size());
  const size_t num_filled = static_cast<uint8_t>(tail.front());
  tail_buffer.remove_prefix(2);
  UnpackVectorFromHeadAndTail(head.edge_p, &tail_buffer, head.num_edges, &p);
  UnpackVectorFromHeadAndTail(head.moves, &tail_buffer, head.num_edges, &moves);
  UnpackVectorFromHeadAndTail(head.edge_n, &tail_buffer, num_filled, &n);
  UnpackVectorFromHeadAndTail(head.edge_q, &tail_buffer, num_filled, &q);
}

void UnpackedNode::UpdateNIntoHead(NodeHead* head) {
  PackVectorIntoHeadAndTail(n, &head->edge_n, nullptr);
  PackVectorIntoHeadAndTail(q, &head->edge_q, nullptr);
}

void UnpackedNode::UpdateNIntoHeadAndTail(NodeHead* head, NodeTail* tail) {
  tail[0] = static_cast<char>(n.size());
  // Truncate it before N and Q.
  tail->resize(2 + BytesNotFitArray(head->edge_p, head->num_edges) +
               BytesNotFitArray(head->moves, head->num_edges));
  PackVectorIntoHeadAndTail(n, &head->edge_n, tail);
  PackVectorIntoHeadAndTail(q, &head->edge_q, tail);
}
#endif

}  // namespace lc2