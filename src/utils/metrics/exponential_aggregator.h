#pragma once

#include <absl/strings/str_cat.h>
#include <absl/synchronization/mutex.h>

#include <chrono>
#include <cmath>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace lczero {

enum class TimePeriod {
  kEmpty = -128,
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
  using Duration = std::chrono::nanoseconds;
  using Clock = std::chrono::steady_clock;

  // Resets the aggregator, clearing all buckets and live metrics.
  void Reset(Clock::time_point now = Clock::now());

  // Merges the passed metric into the live bucket, and clears it.
  template <typename T>
  void UpdateLiveMetrics(T&& metric);

  // Returns the latest completed metrics for the given time period and duration
  // since that period finished last time. If now is nullopt, it
  // excludes the time since the last metrics flush.
  std::pair<Metric, Duration> GetCompletedMetricsAndAge(
      TimePeriod period,
      std::optional<Clock::time_point> now = std::nullopt) const;

  // Returns the live metrics that have been collected for at least the
  // specified duration. Returns the metrics and the duration since the
  // beginning of the covered period. If `now` is nullopt, it excludes metrics
  // and the time since the last metrics flush.
  std::pair<Metric, Duration> GetLiveMetricsAtLeast(
      Duration duration,
      std::optional<Clock::time_point> now = Clock::now()) const;

  // Flushes the current live bucket into the exponential metrics and advances
  // time by the elapsed duration, potentially processing multiple ticks.
  // Returns the largest time period that was updated by this advance (all
  // smaller periods are also updated).
  TimePeriod Advance(Clock::time_point now = Clock::now());

  constexpr Duration GetResolution() const { return kPeriodDuration; }

 private:
  static constexpr Duration kPeriodDuration =
      std::chrono::duration_cast<Duration>(std::chrono::duration<double>(
          std::pow(2.0f, static_cast<int>(kBaseTimePeriod))));

  mutable absl::Mutex mutex_;
  size_t tick_count_ ABSL_GUARDED_BY(mutex_);

  // Buckets for each time period, starting from kBaseTimePeriod.
  std::vector<Metric> buckets_ ABSL_GUARDED_BY(mutex_);
  Clock::time_point last_tick_time_ ABSL_GUARDED_BY(mutex_);

  mutable absl::Mutex live_mutex_ ABSL_ACQUIRED_AFTER(mutex_);
  Metric live_bucket_ ABSL_GUARDED_BY(live_mutex_);
};

template <typename Metric>
void ExponentialAggregator<Metric>::Reset(
    std::chrono::steady_clock::time_point now) {
  absl::MutexLock lock(&mutex_);
  tick_count_ = 0;
  buckets_.clear();
  last_tick_time_ = now;

  absl::MutexLock live_lock(&live_mutex_);
  live_bucket_.Reset();
}

template <typename Metric>
template <typename T>
void ExponentialAggregator<Metric>::UpdateLiveMetrics(T&& metric) {
  absl::MutexLock lock(&live_mutex_);
  live_bucket_.MergeFrom(std::forward<T>(metric));
  metric.Reset();
}

template <typename Metric>
auto ExponentialAggregator<Metric>::GetCompletedMetricsAndAge(
    TimePeriod period, std::optional<Clock::time_point> now) const
    -> std::pair<Metric, Duration> {
  absl::MutexLock lock(&mutex_);
  const size_t index =
      static_cast<int>(period) - static_cast<int>(kBaseTimePeriod);
  const Duration duration_since_update =
      kPeriodDuration * (tick_count_ % (1ULL << index)) +
      (now.has_value() ? Duration(now.value() - last_tick_time_)
                       : Duration::zero());
  if (index >= buckets_.size()) return {Metric(), duration_since_update};
  return {buckets_[index], duration_since_update};
}

template <typename Metric>
auto ExponentialAggregator<Metric>::GetLiveMetricsAtLeast(
    Duration duration, std::optional<Clock::time_point> now) const
    -> std::pair<Metric, Duration> {
  Duration result_duration = Duration::zero();
  Metric result;

  {
    absl::MutexLock lock(&mutex_);
    if (now.has_value()) {
      Duration duration_since_update = *now - last_tick_time_;
      duration -= duration_since_update;
      result_duration += duration_since_update;
    }

    if (duration > Duration::zero()) {
      size_t num_buckets = std::max(
          1.0, std::ceil(
                   std::log2(std::chrono::duration<double>(duration).count())) -
                   static_cast<int>(kBaseTimePeriod));
      uint64_t mask =
          (1ULL << num_buckets) + (tick_count_ & ((1ULL << num_buckets) - 1));
      while (mask) {
        size_t idx = std::countr_zero(mask);
        mask &= ~(1ULL << idx);
        if (idx < buckets_.size()) result.MergeFrom(buckets_[idx]);
        result_duration += kPeriodDuration * (1ULL << idx);
      }
    }
  }

  if (now.has_value()) {
    absl::MutexLock live_lock(&live_mutex_);
    result.MergeFrom(live_bucket_);
  }

  return {result, result_duration};
}

template <typename Metric>
auto ExponentialAggregator<Metric>::Advance(Clock::time_point now)
    -> TimePeriod {
  absl::MutexLock lock(&mutex_);
  const int num_ticks = (now - last_tick_time_) / kPeriodDuration;
  if (num_ticks <= 0) return TimePeriod::kEmpty;
  last_tick_time_ += num_ticks * kPeriodDuration;

  Metric live_carry;
  {
    absl::MutexLock live_lock(&live_mutex_);
    live_carry = std::move(live_bucket_);
    live_bucket_.Reset();
  }

  const size_t initial_tick_count = tick_count_;

  auto one_tick = [&](Metric& carry) {
    ++tick_count_;

    for (size_t i = 0;; ++i) {
      const uint64_t interval_size = 1ULL << i;
      if ((tick_count_ % interval_size) != 0) break;
      while (i >= buckets_.size()) buckets_.emplace_back();
      // We merge new into old, so it's important to swap the carry first.
      std::swap(carry, buckets_[i]);
      carry.MergeFrom(buckets_[i]);
    }
  };

  one_tick(live_carry);
  for (int i = 1; i < num_ticks; ++i) {
    Metric empty_carry;
    one_tick(empty_carry);
  }

  return static_cast<TimePeriod>(
      std::bit_width(initial_tick_count ^ tick_count_) - 1 +
      static_cast<int>(kBaseTimePeriod));
}

}  // namespace lczero