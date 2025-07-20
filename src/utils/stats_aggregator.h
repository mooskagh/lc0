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

// Metric is a struct that implements the following interface:
// - void Reset();  // Resets the metric to its initial state.
// - void MergeFrom(const Metric& other);  // Merges another metric into this
// one. Note that the incoming always happens later in time, so if e.g. merge
// keeps the latest value, it should update the current value with the incoming
// one.
// - (optional) std::string_view name() const;
// - (optional) std::string ToString() const; // If provided, returns a string
// representation of the metric.

// Group several metric types together. This allows us to have a single
// `MetricGroup` that contains multiple different metrics.

template <typename... StatRecords>
class MetricGroup {
 public:
  MetricGroup() = default;

  // Calls reset on all stats.
  void Reset();

  // Merges each individual stat from `other` into this group.
  void MergeFrom(const MetricGroup<StatRecords...>& other);

  // Merges a single stat from `other` into this group.
  template <typename T>
  void MergeFrom(const T& other);

  // Gets a const reference to a specific stat record.
  template <typename T>
  const T& Get() const;

  // Gets a mutable pointer to a specific stat record.
  template <typename T>
  T* GetMutable();

  // Returns a string representation of the group.
  std::string ToString() const;

 private:
  std::tuple<StatRecords...> stats_;
};

template <typename Metric>
class ExponentialAggregator {
 public:
  enum TimePeriod {
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

  constexpr static int kBaseTimePeriod = k16Milliseconds;

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

 private:
  static constexpr float kPeriodSeconds = std::pow(2.0f, kBaseTimePeriod);
  mutable absl::Mutex mutex_;
  size_t tick_count_ ABSL_GUARDED_BY(mutex_);

  // Buckets for each time period, starting from kBaseTimePeriod.
  std::vector<Metric> buckets_{1} ABSL_GUARDED_BY(mutex_);
  std::chrono::steady_clock::time_point last_tick_time_ ABSL_GUARDED_BY(mutex_);

  mutable absl::Mutex live_mutex_;
  Metric live_bucket_ ABSL_GUARDED_BY(live_mutex_);
};

template <typename... StatRecords>
void MetricGroup<StatRecords...>::Reset() {
  (std::get<StatRecords>(stats_).Reset(), ...);
}

template <typename... StatRecords>
void MetricGroup<StatRecords...>::MergeFrom(
    const MetricGroup<StatRecords...>& other) {
  (std::get<StatRecords>(stats_).MergeFrom(std::get<StatRecords>(other.stats_)),
   ...);
}

template <typename... StatRecords>
template <typename T>
void MetricGroup<StatRecords...>::MergeFrom(const T& other) {
  static_assert((std::is_same_v<T, StatRecords> || ...),
                "Type T must be one of the Stats types");
  std::get<T>(stats_).MergeFrom(other);
}

template <typename... StatRecords>
template <typename T>
const T& MetricGroup<StatRecords...>::Get() const {
  static_assert((std::is_same_v<T, StatRecords> || ...),
                "Type T must be one of the Stats types");
  return std::get<T>(stats_);
}

template <typename... StatRecords>
template <typename T>
T* MetricGroup<StatRecords...>::GetMutable() {
  static_assert((std::is_same_v<T, StatRecords> || ...),
                "Type T must be one of the Stats types");
  return &std::get<T>(stats_);
}

template <typename... StatRecords>
std::string MetricGroup<StatRecords...>::ToString() const {
  std::string result;
  (
      [&](const auto& stat) {
        if constexpr (requires { stat.ToString(); }) {
          if (!result.empty()) absl::StrAppend(&result, "\n");
          if constexpr (requires { stat.name(); }) {
            absl::StrAppend(&result, stat.name(), ": ");
          }
          absl::StrAppend(&result, "{", stat.ToString(), "}");
        }
      }(std::get<StatRecords>(stats_)),
      ...);
  return result;
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
  const size_t index = period - kBaseTimePeriod;
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
  size_t num_buckets = std::ceil(std::log2(seconds) - kBaseTimePeriod);
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
      return static_cast<TimePeriod>((i - 1) + kBaseTimePeriod);
    }
    while (i >= buckets_.size()) buckets_.emplace_back();
    // We merge new into old, so it's important to swap the carry first.
    std::swap(carry, buckets_[i]);
    carry.MergeFrom(buckets_[i]);
  }
}

}  // namespace lczero