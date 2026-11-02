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

#include <absl/base/thread_annotations.h>
#include <absl/synchronization/mutex.h>
#include <absl/time/time.h>

#include <chrono>

// lc5 locks with absl::Mutex, so clang checks the ABSL_GUARDED_BY
// annotations, and waits with absl::Condition, which the mutex re-evaluates
// on every unlock: there are no condition variables to notify.
//
// Abseil renamed Mutex::Lock()/Unlock() to lock()/unlock() and made MutexLock
// take a reference, deprecating the old spellings; the vendored fallback
// predates that. The helpers below compile against both.

namespace lczero::lc5 {

template <typename M>
void LockMutex(M& mutex) ABSL_EXCLUSIVE_LOCK_FUNCTION(mutex) {
  if constexpr (requires { mutex.lock(); }) {
    mutex.lock();
  } else {
    mutex.Lock();
  }
}

template <typename M>
void UnlockMutex(M& mutex) ABSL_UNLOCK_FUNCTION(mutex) {
  if constexpr (requires { mutex.unlock(); }) {
    mutex.unlock();
  } else {
    mutex.Unlock();
  }
}

// Holds `mutex` for its scope, like absl::MutexLock.
class ABSL_SCOPED_LOCKABLE MutexLock {
 public:
  explicit MutexLock(absl::Mutex& mutex) ABSL_EXCLUSIVE_LOCK_FUNCTION(mutex)
      : mutex_(mutex) {
    LockMutex(mutex_);
  }
  ~MutexLock() ABSL_UNLOCK_FUNCTION() { UnlockMutex(mutex_); }
  MutexLock(const MutexLock&) = delete;
  MutexLock& operator=(const MutexLock&) = delete;

 private:
  absl::Mutex& mutex_;
};

// A MutexLock that can be released before the end of its scope.
class ABSL_SCOPED_LOCKABLE ReleasableMutexLock {
 public:
  explicit ReleasableMutexLock(absl::Mutex& mutex)
      ABSL_EXCLUSIVE_LOCK_FUNCTION(mutex)
      : mutex_(&mutex) {
    LockMutex(*mutex_);
  }
  ~ReleasableMutexLock() ABSL_UNLOCK_FUNCTION() {
    if (mutex_) UnlockMutex(*mutex_);
  }
  void Release() ABSL_UNLOCK_FUNCTION() {
    UnlockMutex(*mutex_);
    mutex_ = nullptr;
  }
  ReleasableMutexLock(const ReleasableMutexLock&) = delete;
  ReleasableMutexLock& operator=(const ReleasableMutexLock&) = delete;

 private:
  absl::Mutex* mutex_;
};

// Waits on the held `mutex` until `condition` holds or `deadline` passes;
// returns whether the condition holds. time_point::max() never times out.
inline bool AwaitUntil(absl::Mutex& mutex, const absl::Condition& condition,
                       std::chrono::steady_clock::time_point deadline)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex) {
  if (deadline == std::chrono::steady_clock::time_point::max()) {
    mutex.Await(condition);
    return true;
  }
  return mutex.AwaitWithTimeout(
      condition, absl::FromChrono(deadline - std::chrono::steady_clock::now()));
}

}  // namespace lczero::lc5
