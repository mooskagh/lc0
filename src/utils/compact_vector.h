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
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <utility>

namespace lczero {

// Heap vector with compact size and capacity fields (default uint8_t, up to 255
// elements).
template <typename T, typename SizeType = uint8_t>
class CompactVector {
 public:
  static constexpr size_t kMaxSize = std::numeric_limits<SizeType>::max();

  CompactVector() = default;
  CompactVector(std::initializer_list<T> init) {
    reserve(init.size());
    for (const T& item : init) push_back(item);
  }
  CompactVector(const CompactVector& other) { *this = other; }
  CompactVector& operator=(const CompactVector& other) {
    if (this == &other) return *this;
    data_ = other.size_ ? std::make_unique<T[]>(other.size_) : nullptr;
    size_ = capacity_ = other.size_;
    std::ranges::copy(other, data_.get());
    return *this;
  }
  CompactVector(CompactVector&& other) noexcept
      : data_(std::move(other.data_)),
        size_(std::exchange(other.size_, 0)),
        capacity_(std::exchange(other.capacity_, 0)) {}
  CompactVector& operator=(CompactVector&& other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      size_ = std::exchange(other.size_, 0);
      capacity_ = std::exchange(other.capacity_, 0);
    }
    return *this;
  }

  size_t size() const { return size_; }
  size_t capacity() const { return capacity_; }
  bool empty() const { return size_ == 0; }
  T* data() { return data_.get(); }
  const T* data() const { return data_.get(); }
  T* begin() { return data_.get(); }
  const T* begin() const { return data_.get(); }
  T* end() { return data_.get() + size_; }
  const T* end() const { return data_.get() + size_; }
  T& operator[](size_t i) {
    assert(i < size_);
    return data_[i];
  }
  const T& operator[](size_t i) const {
    assert(i < size_);
    return data_[i];
  }
  T& back() {
    assert(size_ > 0);
    return data_[size_ - 1];
  }
  const T& back() const {
    assert(size_ > 0);
    return data_[size_ - 1];
  }

  void reserve(size_t n) {
    assert(n <= kMaxSize);
    if (n <= capacity_) return;
    auto next = std::make_unique<T[]>(n);
    std::ranges::move(*this, next.get());
    data_ = std::move(next);
    capacity_ = static_cast<SizeType>(n);
  }
  void resize(size_t n) {
    reserve(n);
    if (n > size_) std::fill(data_.get() + size_, data_.get() + n, T{});
    size_ = static_cast<SizeType>(n);
  }
  template <typename... Args>
  T& emplace_back(Args&&... args) {
    assert(size_ < kMaxSize);
    if (size_ == capacity_) {
      reserve(capacity_ == 0
                  ? 2
                  : std::min<size_t>(kMaxSize, size_t{capacity_} * 2));
    }
    return data_[size_++] = T{std::forward<Args>(args)...};
  }
  void push_back(const T& value) { emplace_back(value); }

 private:
  std::unique_ptr<T[]> data_;
  SizeType size_ = 0;
  SizeType capacity_ = 0;
};

}  // namespace lczero
