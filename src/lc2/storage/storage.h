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

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "utils/mutex.h"

#pragma once

namespace lc2 {

template <typename BlockHash, typename PrimaryBlock, typename SecondaryBlock>
class Storage {
 public:
  enum class Status {
    kFetched,  // The node existed and was fetched.
    kCreated,  // The node didn't exist and was created empty.
    // kCollision,  // The node existed, but empty, being created in other
    // place.
  };

  // using BlockHash = uint64_t;
  // using PrimaryBlock = uint8_t[64];
  // using SecondaryBlock = std::string;

  // Returns whether secondary block has to be fetched.
  // FetchPrimaryFunc = bool (*)(size_t, PrimaryBlock*, Status)
  // FetchSecondaryFunc = void (*)(size_t, PrimaryBlock*, SecondaryBlock*)

  // For every hash in @hashes, the function fetches or creates primary block
  // and calls primary_func(). If it returns `true`, also secondary block is
  // fetched and secondary_func() is called.
  // It is guaranteed that blocks don't change between the calls.
  // primary_func() and secondary_func() must be fast as they are called when
  // mutex is held.
  // FetchPrimaryFunc = bool (*)(size_t, PrimaryBlock*, Status)
  // FetchSecondaryFunc = void (*)(size_t, PrimaryBlock*, SecondaryBlock*)
  template <typename FetchPrimaryFunc, typename FetchSecondaryFunc>
  void FetchOrCreate(absl::Span<BlockHash> hashes,
                     FetchPrimaryFunc primary_func,
                     FetchSecondaryFunc secondary_func);

 private:
  absl::flat_hash_map<BlockHash, PrimaryBlock> primary_blocks_
      GUARDED_BY(mutex_);
  absl::flat_hash_map<BlockHash, SecondaryBlock> secondary_blocks_
      GUARDED_BY(mutex_);
  lczero::Mutex mutex_;

  static_assert(sizeof(typename absl::flat_hash_map<BlockHash,
                                                    PrimaryBlock>::value_type) %
                        32 ==
                    0,
                "Expected the size of hash entry to be multiple of 32 bytes");
  
};

template <typename BlockHash, typename PrimaryBlock, typename SecondaryBlock>
template <typename FetchPrimaryFunc, typename FetchSecondaryFunc>
inline void Storage<BlockHash, PrimaryBlock, SecondaryBlock>::FetchOrCreate(
    absl::Span<BlockHash> hashes, FetchPrimaryFunc primary_func,
    FetchSecondaryFunc secondary_func) {
  lczero::Mutex::Lock lock(mutex_);
  for (size_t i = 0; i < hashes.size(); ++i) {
    BlockHash hash = hashes[i];
    auto [iter, inserted] = primary_blocks_.emplace(hash, PrimaryBlock{});
    const bool fetch_secondary = primary_func(
        i, &iter->second, inserted ? Status::kCreated : Status::kFetched);
    if (fetch_secondary) {
      secondary_func(i, &iter->second, &secondary_blocks_[hash]);
    }
  }
}

}  // namespace lc2