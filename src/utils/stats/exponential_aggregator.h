#pragma once

#include <absl/strings/str_cat.h>
#include <absl/synchronization/mutex.h>

#include <chrono>
#include <cmath>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace lczero {

enum class TimePeriod {
  k16Milliseconds = -6,
  k31Milliseconds,
  k63Milliseconds,
  k125Milliseconds,
  k250Milliseconds,
  k500Milliseconds,
  k1Second = 0,
  k2Seconds,
  k4Seconds,
  k8Seconds,
  k16Seconds,
  k32Seconds,
  k1Minute,
  k2Minutes,
  k4Minutes,
  k9Minutes,
  k17Minutes,
  k36Minutes,
  k1Hour,
  k2Hours,
  k5Hours,
  k9Hours,
};

template <typename Metric>
class ExponentialAggregator {
 public:
  constexpr static TimePeriod kBaseTimePeriod = TimePeriod::k16Milliseconds;

  // Resets the aggregator, clearing all buckets and live stats.
  void Reset();

  // Merges the passed metric into the live bucket, and clears it.
  template <typename T>
  void UpdateLiveStats(T&& stat);

  // Returns the latest completed stats for the given time period and time in
  // seconds since that period finished last time. If include_live_time is
  // false, it excludes the time since the last tick.
  std::pair<Metric, float> GetCompletedStatsAndAgeSeconds(
      TimePeriod period, bool include_live_time = false) const;

  // Returns the live stats that have been collected for at least `seconds`
  // seconds. Returns the stats and the time in seconds since the beginning of
  // the covered period. If `include_live_stats` is false, it excludes the time
  // since the last tick.
  std::pair<Metric, float> GetLiveStatsOfAtLeast(
      float seconds, bool include_live_stats = true) const;

  // Flushes the current live bucket into the exponential stats.
  // Must be called every kBaseTimePeriod seconds.
  // Returns the largest time period that was updated by this tick (all smaller
  // periods are also updated).
  TimePeriod Tick();

  constexpr uint64_t GetResolutionMicroseconds() const {
    return static_cast<uint64_t>(kPeriodSeconds * 1'000'000);
  }

 private:
  static constexpr float kPeriodSeconds =
      std::pow(2.0f, static_cast<int>(kBaseTimePeriod));
  mutable absl::Mutex mutex_;
  size_t tick_count_ ABSL_GUARDED_BY(mutex_);

  // Buckets for each time period, starting from kBaseTimePeriod.
  std::vector<Metric> buckets_ ABSL_GUARDED_BY(mutex_);
  std::chrono::steady_clock::time_point last_tick_time_ ABSL_GUARDED_BY(mutex_);

  mutable absl::Mutex live_mutex_;
  Metric live_bucket_ ABSL_GUARDED_BY(live_mutex_);
};



template <typename Metric>
void ExponentialAggregator<Metric>::Reset() {
  {
    absl::MutexLock lock(&mutex_);
    tick_count_ = 0;
    buckets_.clear();
    last_tick_time_ = std::chrono::steady_clock::now();
  }
  {
    absl::MutexLock live_lock(&live_mutex_);
    live_bucket_.Reset();
  }
}

template <typename Metric>
template <typename T>
void ExponentialAggregator<Metric>::UpdateLiveStats(T&& stat) {
  absl::MutexLock lock(&live_mutex_);
  live_bucket_.MergeFrom(std::forward<T>(stat));
  stat.Reset();
}

template <typename Metric>
std::pair<Metric, float>
ExponentialAggregator<Metric>::GetCompletedStatsAndAgeSeconds(
    TimePeriod period, bool include_live_time) const {
  absl::MutexLock lock(&mutex_);
  const size_t index =
      static_cast<int>(period) - static_cast<int>(kBaseTimePeriod);
  const float seconds_since_update =
      kPeriodSeconds * (tick_count_ % (1ULL << index)) + include_live_time
          ? (std::chrono::duration<float>(std::chrono::steady_clock::now() -
                                          last_tick_time_)
                 .count())
          : 0.0f;
  if (index >= buckets_.size()) return {Metric(), seconds_since_update};
  return {buckets_[index], seconds_since_update};
}

template <typename Metric>
std::pair<Metric, float> ExponentialAggregator<Metric>::GetLiveStatsOfAtLeast(
    float seconds, bool include_live_stats) const {
  absl::MutexLock lock(&live_mutex_);
  float seconds_since_update = 0.0f;
  Metric result;
  if (include_live_stats) {
    seconds_since_update =
        std::chrono::duration<float>(std::chrono::steady_clock::now() -
                                     last_tick_time_)
            .count();
    seconds -= seconds_since_update;
    result.MergeFrom(live_bucket_);
  }
  if (seconds <= 0.0f) return {result, seconds_since_update};
  size_t num_buckets =
      std::ceil(std::log2(seconds) - static_cast<int>(kBaseTimePeriod));
  uint64_t mask =
      (1ULL << num_buckets) + (tick_count_ & ((1ULL << num_buckets) - 1));
  while (mask) {
    size_t idx = std::countr_zero(mask);
    mask &= ~(1ULL << idx);
    if (idx < buckets_.size()) result.MergeFrom(buckets_[idx]);
    seconds_since_update += kPeriodSeconds * (1ULL << idx);
  }
  return {result, seconds_since_update};
}

template <typename Metric>
auto ExponentialAggregator<Metric>::Tick() -> TimePeriod {
  Metric carry;
  {
    absl::MutexLock live_lock(&live_mutex_);
    carry = std::move(live_bucket_);
    live_bucket_.Reset();
  }
  absl::MutexLock lock(&mutex_);
  tick_count_++;
  last_tick_time_ = std::chrono::steady_clock::now();

  for (size_t i = 0;; ++i) {
    const uint64_t interval_size = 1ULL << i;
    if ((tick_count_ % interval_size) != 0) {
      return static_cast<TimePeriod>((i - 1) +
                                     static_cast<int>(kBaseTimePeriod));
    }
    while (i >= buckets_.size()) buckets_.emplace_back();
    // We merge new into old, so it's important to swap the carry first.
    std::swap(carry, buckets_[i]);
    carry.MergeFrom(buckets_[i]);
  }
}

}  // namespace lczero