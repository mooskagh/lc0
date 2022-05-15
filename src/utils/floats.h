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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace lczero {

template <class To, class From>
static To bit_cast(From from) {
  if constexpr (std::is_same_v<From, To>) {
    return from;
  } else {
    To to;
    std::memcpy(&to, &from, sizeof(to));
    return to;
  }
}

template <bool kHasSign, bool kHasExponentSign, int kExponentBits>
class GenericFloat16 {
 public:
  GenericFloat16() = default;
  GenericFloat16(const GenericFloat16&) = default;
  explicit GenericFloat16(float val) {
    constexpr size_t kMantissaSize = 16 - kExponentBits - (kHasSign ? 1 : 0);
    uint32_t from = bit_cast<uint32_t>(val);
    const uint16_t mantissa = (from & ((1 << 23) - 1)) >> (23 - kMantissaSize);
    const uint16_t sign =
        (kHasSign && (from & 0x80000000)) ? (1 << kMantissaSize) : 0;
    const uint16_t exponent =
        ((((from >> 23) & 0xff) -
          (kHasExponentSign ? (127 - (1 << (kExponentBits - 1))) : 127))
         << (16 - kExponentBits));
    value = mantissa | sign | exponent;
  }
  explicit operator float() const {
    constexpr size_t kMantissaSize = 16 - kExponentBits - (kHasSign ? 1 : 0);
    const uint32_t mantissa = (value << (23 - kMantissaSize)) & ((1 << 23) - 1);
    const uint32_t exponent =
        ((value >> (16 - kExponentBits)) +
         (kHasExponentSign ? (127 - (1 << (kExponentBits - 1))) : 127))
        << 23;
    const uint32_t sign =
        kHasSign && (value & (1 << kMantissaSize)) != 0 ? 0x80000000 : 0;
    return bit_cast<float>(mantissa | exponent | sign);
  }

 private:
  uint16_t value;
};

// template<int kExponentBits>

// -1.0..1.0
using SigmoidFloat16 = GenericFloat16<true, false, 5>;
// 0.0..1.0
using ProbFloat16 = GenericFloat16<false, false, 5>;
// 0.0..110000
using MLFloat16 = GenericFloat16<false, true, 4>;

}  // namespace lczero