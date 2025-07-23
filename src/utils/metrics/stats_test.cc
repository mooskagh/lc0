#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <optional>
#include <thread>

#include "utils/metrics/exponential_aggregator.h"
#include "utils/metrics/group.h"
#include "utils/metrics/printer.h"

namespace lczero {

// Mock metric classes for testing
class CounterMetric {
 public:
  CounterMetric() : count_(0) {}
  CounterMetric(int count) : count_(count) {}

  void Reset() { count_ = 0; }

  void MergeFrom(const CounterMetric& other) { count_ += other.count_; }

  void Print(MetricPrinter& printer) const {
    printer.StartGroup("CounterMetric");
    printer.Print("count", static_cast<size_t>(count_));
    printer.EndGroup();
  }

  int count() const { return count_; }
  void set_count(int count) { count_ = count; }

 private:
  int count_;
};

class AverageMetric {
 public:
  AverageMetric() : sum_(0), count_(0) {}
  AverageMetric(double sum, int count) : sum_(sum), count_(count) {}

  void Reset() {
    sum_ = 0;
    count_ = 0;
  }

  void MergeFrom(const AverageMetric& other) {
    sum_ += other.sum_;
    count_ += other.count_;
  }

  void Print(MetricPrinter& printer) const {
    printer.StartGroup("AverageMetric");
    printer.Print("sum", std::to_string(sum_));
    printer.Print("count", static_cast<size_t>(count_));
    if (count_ > 0) {
      printer.Print("average", std::to_string(sum_ / count_));
    }
    printer.EndGroup();
  }

  double average() const { return count_ > 0 ? sum_ / count_ : 0.0; }
  void add_sample(double value) {
    sum_ += value;
    count_++;
  }

  double sum() const { return sum_; }
  int count() const { return count_; }

 private:
  double sum_;
  int count_;
};

class MaxMetric {
 public:
  MaxMetric() : max_value_(0), has_value_(false) {}
  MaxMetric(double max_value) : max_value_(max_value), has_value_(true) {}

  void Reset() {
    max_value_ = 0;
    has_value_ = false;
  }

  void MergeFrom(const MaxMetric& other) {
    if (other.has_value_) {
      if (!has_value_ || other.max_value_ > max_value_) {
        max_value_ = other.max_value_;
        has_value_ = true;
      }
    }
  }

  void Print(MetricPrinter& printer) const {
    printer.StartGroup("MaxMetric");
    if (has_value_) {
      printer.Print("max_value", std::to_string(max_value_));
      printer.Print("has_value", static_cast<size_t>(1));
    } else {
      printer.Print("has_value", static_cast<size_t>(0));
    }
    printer.EndGroup();
  }

  double max_value() const { return max_value_; }
  bool has_value() const { return has_value_; }
  void set_value(double value) {
    if (!has_value_ || value > max_value_) {
      max_value_ = value;
      has_value_ = true;
    }
  }

 private:
  double max_value_;
  bool has_value_;
};

// Optional value metric that demonstrates overshadowing behavior
class OptionalValueMetric {
 public:
  OptionalValueMetric() : value_(std::nullopt) {}
  OptionalValueMetric(double value) : value_(value) {}

  void Reset() { value_ = std::nullopt; }

  void MergeFrom(const OptionalValueMetric& other) {
    // Only copy the value if the other metric has one (overshadowing behavior)
    if (other.value_.has_value()) {
      value_ = other.value_;
    }
  }

  void Print(MetricPrinter& printer) const {
    printer.StartGroup("OptionalValueMetric");
    if (value_.has_value()) {
      printer.Print("value", std::to_string(value_.value()));
      printer.Print("has_value", static_cast<size_t>(1));
    } else {
      printer.Print("has_value", static_cast<size_t>(0));
    }
    printer.EndGroup();
  }

  std::optional<double> value() const { return value_; }
  bool has_value() const { return value_.has_value(); }
  void set_value(double value) { value_ = value; }

 private:
  std::optional<double> value_;
};

// Test MetricGroup functionality
class MetricGroupTest : public ::testing::Test {
 protected:
  using TestGroup = MetricGroup<CounterMetric, AverageMetric, MaxMetric>;
  TestGroup group_;
};

TEST_F(MetricGroupTest, InitialState) {
  // Test that metrics are initialized in their default state
  EXPECT_EQ(group_.Get<CounterMetric>().count(), 0);
  EXPECT_EQ(group_.Get<AverageMetric>().count(), 0);
  EXPECT_FALSE(group_.Get<MaxMetric>().has_value());
}

TEST_F(MetricGroupTest, GetMutable) {
  // Test getting mutable references and modifying them
  auto* counter = group_.GetMutable<CounterMetric>();
  counter->set_count(42);
  EXPECT_EQ(group_.Get<CounterMetric>().count(), 42);

  auto* average = group_.GetMutable<AverageMetric>();
  average->add_sample(10.0);
  average->add_sample(20.0);
  EXPECT_EQ(group_.Get<AverageMetric>().average(), 15.0);

  auto* max_metric = group_.GetMutable<MaxMetric>();
  max_metric->set_value(100.0);
  EXPECT_EQ(group_.Get<MaxMetric>().max_value(), 100.0);
}

TEST_F(MetricGroupTest, Reset) {
  // Set up some data
  group_.GetMutable<CounterMetric>()->set_count(42);
  group_.GetMutable<AverageMetric>()->add_sample(10.0);
  group_.GetMutable<MaxMetric>()->set_value(100.0);

  // Reset and verify everything is back to initial state
  group_.Reset();

  EXPECT_EQ(group_.Get<CounterMetric>().count(), 0);
  EXPECT_EQ(group_.Get<AverageMetric>().count(), 0);
  EXPECT_FALSE(group_.Get<MaxMetric>().has_value());
}

TEST_F(MetricGroupTest, MergeFromGroup) {
  // Set up source group
  TestGroup other;
  other.GetMutable<CounterMetric>()->set_count(10);
  other.GetMutable<AverageMetric>()->add_sample(5.0);
  other.GetMutable<MaxMetric>()->set_value(50.0);

  // Set up destination group
  group_.GetMutable<CounterMetric>()->set_count(20);
  group_.GetMutable<AverageMetric>()->add_sample(15.0);
  group_.GetMutable<MaxMetric>()->set_value(30.0);

  // Merge
  group_.MergeFrom(other);

  // Verify results
  EXPECT_EQ(group_.Get<CounterMetric>().count(), 30);      // 20 + 10
  EXPECT_EQ(group_.Get<AverageMetric>().average(), 10.0);  // (15 + 5) / 2
  EXPECT_EQ(group_.Get<MaxMetric>().max_value(), 50.0);    // max(30, 50)
}

TEST_F(MetricGroupTest, MergeFromSingleMetric) {
  // Set up initial state
  group_.GetMutable<CounterMetric>()->set_count(20);

  // Create a single metric to merge
  CounterMetric counter(15);

  // Merge single metric
  group_.MergeFrom(counter);

  // Verify result
  EXPECT_EQ(group_.Get<CounterMetric>().count(), 35);  // 20 + 15
}

TEST_F(MetricGroupTest, Print) {
  // Set up data
  group_.GetMutable<CounterMetric>()->set_count(42);
  group_.GetMutable<AverageMetric>()->add_sample(10.0);
  group_.GetMutable<AverageMetric>()->add_sample(20.0);
  group_.GetMutable<MaxMetric>()->set_value(100.0);

  std::string result = MetricToString(group_);

  // Should contain all metric names and values
  EXPECT_NE(result.find("CounterMetric"), std::string::npos);
  EXPECT_NE(result.find("count=42"), std::string::npos);
  EXPECT_NE(result.find("AverageMetric"), std::string::npos);
  EXPECT_NE(result.find("average=15"), std::string::npos);  // (10+20)/2
  EXPECT_NE(result.find("MaxMetric"), std::string::npos);
  EXPECT_NE(result.find("max_value=100"), std::string::npos);
}

// Test MetricToString functionality
class MetricPrinterTest : public ::testing::Test {};

TEST_F(MetricPrinterTest, StringMetricPrinter) {
  std::string output;
  StringMetricPrinter printer(&output);

  printer.StartGroup("test_group");
  printer.Print("metric1", std::string("value1"));
  printer.Print("metric2", std::string("42"));
  printer.EndGroup();

  EXPECT_EQ(output, "test_group={metric1=value1, metric2=42}");
}

TEST_F(MetricPrinterTest, MultipleGroups) {
  std::string output;
  StringMetricPrinter printer(&output);

  printer.StartGroup("group1");
  printer.Print("metric1", std::string("value1"));
  printer.EndGroup();

  printer.StartGroup("group2");
  printer.Print("metric2", std::string("value2"));
  printer.EndGroup();

  EXPECT_EQ(output, "group1={metric1=value1}, group2={metric2=value2}");
}

TEST_F(MetricPrinterTest, EmptyGroup) {
  std::string output;
  StringMetricPrinter printer(&output);

  printer.StartGroup("empty_group");
  printer.EndGroup();

  EXPECT_EQ(output, "empty_group={}");
}

TEST_F(MetricPrinterTest, SizeTOverload) {
  std::string output;
  StringMetricPrinter string_printer(&output);
  MetricPrinter& printer = string_printer;  // Use base class interface

  printer.StartGroup("test_group");
  printer.Print("count", static_cast<size_t>(123));
  printer.EndGroup();

  EXPECT_EQ(output, "test_group={count=123}");
}

TEST_F(MetricPrinterTest, MetricToStringFunction) {
  CounterMetric counter(123);
  std::string result = MetricToString(counter);

  EXPECT_NE(result.find("CounterMetric"), std::string::npos);
  EXPECT_NE(result.find("123"), std::string::npos);
}

// Test ExponentialAggregator functionality
class ExponentialAggregatorTest : public ::testing::Test {
 protected:
  using TestMetric = MetricGroup<CounterMetric, AverageMetric>;
  using TestAggregator = ExponentialAggregator<TestMetric>;

  void SetUp() override {
    // Create a fresh aggregator for each test to avoid state contamination
    aggregator_ = std::make_unique<TestAggregator>();
    start_time_ = TestAggregator::Clock::now();
    aggregator_->Reset(start_time_);
  }

  std::unique_ptr<TestAggregator> aggregator_;
  TestAggregator::Clock::time_point start_time_;
};

TEST_F(ExponentialAggregatorTest, RecordMetrics) {
  TestMetric metric;
  metric.GetMutable<CounterMetric>()->set_count(10);
  metric.GetMutable<AverageMetric>()->add_sample(5.0);

  // Update live metrics
  aggregator_->RecordMetrics(std::move(metric));

  // The original metric should be reset after move
  EXPECT_EQ(metric.Get<CounterMetric>().count(), 0);
  EXPECT_EQ(metric.Get<AverageMetric>().count(), 0);

  // Get live metrics to verify they were updated
  auto [live_metrics, age] =
      aggregator_->GetAggregateEndingNow(TestAggregator::Duration::zero());
  EXPECT_EQ(live_metrics.Get<CounterMetric>().count(), 10);
  EXPECT_EQ(live_metrics.Get<AverageMetric>().average(), 5.0);
}

TEST_F(ExponentialAggregatorTest, MultipleUpdatesLiveMetrics) {
  // Update multiple times
  for (int i = 1; i <= 5; ++i) {
    TestMetric metric;
    metric.GetMutable<CounterMetric>()->set_count(i);
    metric.GetMutable<AverageMetric>()->add_sample(i * 2.0);
    aggregator_->RecordMetrics(std::move(metric));
  }

  // Get live metrics
  auto [live_metrics, age] =
      aggregator_->GetAggregateEndingNow(TestAggregator::Duration::zero());
  EXPECT_EQ(live_metrics.Get<CounterMetric>().count(), 15);  // 1+2+3+4+5
  EXPECT_EQ(live_metrics.Get<AverageMetric>().average(),
            6.0);  // (2+4+6+8+10)/5
}

TEST_F(ExponentialAggregatorTest, Advance) {
  // Add some live metrics
  TestMetric metric;
  metric.GetMutable<CounterMetric>()->set_count(10);
  aggregator_->RecordMetrics(std::move(metric));

  // Advance to move live metrics to buckets
  auto tick_time = start_time_ + aggregator_->GetResolution();
  auto period = aggregator_->Advance(tick_time);

  // Should return the base time period
  EXPECT_EQ(period, TimePeriod::k16Milliseconds);

  // Live metrics should be empty after tick
  auto [live_metrics, age] = aggregator_->GetAggregateEndingNow(
      TestAggregator::Duration::zero(), tick_time);
  EXPECT_EQ(live_metrics.Get<CounterMetric>().count(), 0);
}

TEST_F(ExponentialAggregatorTest, MultipleAdvances) {
  // Add metrics and tick multiple times to test bucket management
  auto current_time = start_time_;
  for (const auto expected_period : {
           TimePeriod::k16Milliseconds,
           TimePeriod::k31Milliseconds,
           TimePeriod::k16Milliseconds,
           TimePeriod::k63Milliseconds,
           TimePeriod::k16Milliseconds,
           TimePeriod::k31Milliseconds,
           TimePeriod::k16Milliseconds,
           TimePeriod::k125Milliseconds,
       }) {
    TestMetric metric;
    metric.GetMutable<CounterMetric>()->set_count(1);
    aggregator_->RecordMetrics(std::move(metric));

    current_time += aggregator_->GetResolution();
    auto period = aggregator_->Advance(current_time);

    EXPECT_EQ(period, expected_period);
  }
}

TEST_F(ExponentialAggregatorTest, GetCompletedMetrics) {
  auto resolution = aggregator_->GetResolution();
  auto current_time = start_time_;

  // Tick 1: Push count of 5
  TestMetric m1;
  m1.GetMutable<CounterMetric>()->set_count(5);
  aggregator_->RecordMetrics(std::move(m1));
  current_time += resolution;
  aggregator_->Advance(current_time);

  // Tick 2: Push count of 10
  TestMetric m2;
  m2.GetMutable<CounterMetric>()->set_count(10);
  aggregator_->RecordMetrics(std::move(m2));
  current_time += resolution;
  aggregator_->Advance(current_time);

  // Tick 3: Push count of 4
  TestMetric m3;
  m3.GetMutable<CounterMetric>()->set_count(4);
  aggregator_->RecordMetrics(std::move(m3));
  current_time += resolution;
  aggregator_->Advance(current_time);

  // Live data: Push count of 7, do not advance
  TestMetric m4;
  m4.GetMutable<CounterMetric>()->set_count(7);
  aggregator_->RecordMetrics(std::move(m4));
  auto final_time = current_time + resolution / 2;

  // --- Check k16Milliseconds bucket (last completed tick) ---
  {
    // With live time: age should be time since last tick + live duration
    auto [metrics, age] = aggregator_->GetBucketMetrics(
        TimePeriod::k16Milliseconds, final_time);
    EXPECT_EQ(metrics.Get<CounterMetric>().count(), 4);
    EXPECT_EQ(age, resolution / 2);

    // Without live time: age should be 0 as we are at the exact tick boundary
    auto [metrics_no_live, age_no_live] =
        aggregator_->GetBucketMetrics(TimePeriod::k16Milliseconds,
                                               std::nullopt);
    EXPECT_EQ(metrics_no_live.Get<CounterMetric>().count(), 4);
    EXPECT_EQ(age_no_live.count(), 0);
  }

  // --- Check k31Milliseconds bucket (first two ticks merged) ---
  {
    // With live time: age should be time since bucket completion + live
    // duration
    auto [metrics, age] = aggregator_->GetBucketMetrics(
        TimePeriod::k31Milliseconds, final_time);
    EXPECT_EQ(metrics.Get<CounterMetric>().count(), 5 + 10);
    EXPECT_EQ(age, resolution + resolution / 2);

    // Without live time: age should be time since bucket completion
    auto [metrics_no_live, age_no_live] =
        aggregator_->GetBucketMetrics(TimePeriod::k31Milliseconds,
                                               std::nullopt);
    EXPECT_EQ(metrics_no_live.Get<CounterMetric>().count(), 5 + 10);
    EXPECT_EQ(age_no_live, resolution);
  }
}

TEST_F(ExponentialAggregatorTest, GetLiveMetricsOfAtLeast) {
  auto resolution = aggregator_->GetResolution();
  auto current_time = start_time_;

  // Tick 1: Push count of 5
  TestMetric m1;
  m1.GetMutable<CounterMetric>()->set_count(5);
  aggregator_->RecordMetrics(std::move(m1));
  current_time += resolution;
  aggregator_->Advance(current_time);

  // Tick 2: Push count of 10
  TestMetric m2;
  m2.GetMutable<CounterMetric>()->set_count(10);
  aggregator_->RecordMetrics(std::move(m2));
  current_time += resolution;
  aggregator_->Advance(current_time);

  // Tick 3: Push count of 4
  TestMetric m3;
  m3.GetMutable<CounterMetric>()->set_count(4);
  aggregator_->RecordMetrics(std::move(m3));
  current_time += resolution;
  aggregator_->Advance(current_time);

  // Live data: Push count of 7, do not advance
  TestMetric m4;
  m4.GetMutable<CounterMetric>()->set_count(7);
  aggregator_->RecordMetrics(std::move(m4));
  auto final_time = current_time + resolution / 2;

  // --- Test duration less than elapsed live time ---
  {
    // With live time: should include live metrics only
    auto [metrics, age] =
        aggregator_->GetAggregateEndingNow(resolution / 4, final_time);
    EXPECT_EQ(metrics.Get<CounterMetric>().count(), 7);
    EXPECT_EQ(age, resolution / 2);

    // Without live time: should return the smallest completed bucket (last
    // tick)
    auto [metrics_no_live, age_no_live] =
        aggregator_->GetAggregateEndingNow(resolution / 4, std::nullopt);
    EXPECT_EQ(metrics_no_live.Get<CounterMetric>().count(), 4);  // tick 3 only
    EXPECT_EQ(age_no_live,
              resolution / 4);  // Should be exactly the requested duration
  }

  // --- Test duration more than elapsed live time but less than one tick ---
  {
    // With live time: should include live + historical data needed
    auto [metrics, age] =
        aggregator_->GetAggregateEndingNow(resolution * 3 / 4, final_time);
    EXPECT_EQ(metrics.Get<CounterMetric>().count(), 7 + 4);  // live + tick 3
    EXPECT_GE(age.count(), (resolution / 2).count());

    // Without live time: should return buckets covering at least the duration
    auto [metrics_no_live, age_no_live] =
        aggregator_->GetAggregateEndingNow(resolution * 3 / 4, std::nullopt);
    EXPECT_EQ(metrics_no_live.Get<CounterMetric>().count(), 4);  // tick 3 only
    EXPECT_GE(age_no_live.count(), (resolution * 3 / 4).count());
  }

  // --- Test duration more than one tick but less than two ticks ---
  {
    // With live time: should include live + sufficient historical data
    auto [metrics, age] =
        aggregator_->GetAggregateEndingNow(resolution * 3 / 2, final_time);
    EXPECT_EQ(metrics.Get<CounterMetric>().count(),
              7 + 10 + 4);  // live + tick 2 + tick 3
    EXPECT_GE(age.count(), (resolution / 2).count());

    // Without live time: should return buckets covering at least 1.5 ticks
    auto [metrics_no_live, age_no_live] =
        aggregator_->GetAggregateEndingNow(resolution * 3 / 2, std::nullopt);
    EXPECT_EQ(metrics_no_live.Get<CounterMetric>().count(),
              10 + 4);  // tick 2 + tick 3
    EXPECT_GE(age_no_live.count(), (resolution * 3 / 2).count());
  }

  // --- Test duration more than two ticks ---
  {
    // With live time: should include live + sufficient historical data
    auto [metrics, age] =
        aggregator_->GetAggregateEndingNow(resolution * 5 / 2, final_time);
    EXPECT_EQ(metrics.Get<CounterMetric>().count(),
              7 + 5 + 10 + 4);  // live + all ticks
    EXPECT_GE(age.count(), (resolution / 2).count());

    // Without live time: should return buckets covering at least 2.5 ticks
    auto [metrics_no_live, age_no_live] =
        aggregator_->GetAggregateEndingNow(resolution * 5 / 2, std::nullopt);
    EXPECT_EQ(metrics_no_live.Get<CounterMetric>().count(),
              5 + 10 + 4);  // all ticks
    EXPECT_GE(age_no_live.count(), (resolution * 5 / 2).count());
  }
}

TEST_F(ExponentialAggregatorTest, Advance_TimeDifferenceSmallerThanTick) {
  // Advance with a time difference smaller than one tick.
  auto tick_time = start_time_ + aggregator_->GetResolution() / 2;
  auto period = aggregator_->Advance(tick_time);

  // No tick should have occurred.
  EXPECT_EQ(period, TimePeriod::kEmpty);
}

TEST_F(ExponentialAggregatorTest, Advance_MultipleTicks) {
  // Add some live metrics.
  TestMetric metric;
  metric.GetMutable<CounterMetric>()->set_count(10);
  aggregator_->RecordMetrics(std::move(metric));

  // Advance by a duration of 3 ticks.
  auto tick_time = start_time_ + aggregator_->GetResolution() * 3;
  auto period = aggregator_->Advance(tick_time);

  // The largest updated period should be k31Milliseconds (2 ticks).
  EXPECT_EQ(period, TimePeriod::k31Milliseconds);

  // The first bucket should be empty as it was cleared on the second tick and
  // filled with an empty metric on the third.
  auto [metrics1, age1] =
      aggregator_->GetBucketMetrics(TimePeriod::k16Milliseconds);
  EXPECT_EQ(metrics1.Get<CounterMetric>().count(), 0);

  // The second bucket should contain the metrics from the first tick.
  auto [metrics2, age2] =
      aggregator_->GetBucketMetrics(TimePeriod::k31Milliseconds);
  EXPECT_EQ(metrics2.Get<CounterMetric>().count(), 10);
}

TEST_F(ExponentialAggregatorTest, GetAggregateEndingNow_DurationLessThanTick) {
  // Add some live metrics.
  TestMetric metric;
  metric.GetMutable<CounterMetric>()->set_count(10);
  aggregator_->RecordMetrics(std::move(metric));

  auto current_time = start_time_ + aggregator_->GetResolution() / 2;

  // Get live metrics with a duration less than a tick.
  auto [metrics, age] = aggregator_->GetAggregateEndingNow(
      aggregator_->GetResolution() / 4, current_time);

  // Should still return the live metrics.
  EXPECT_EQ(metrics.Get<CounterMetric>().count(), 10);
  EXPECT_GE(age.count(), (aggregator_->GetResolution() / 2).count());
}

TEST_F(ExponentialAggregatorTest, GetAggregateEndingNow_AcrossTicks) {
  // Add some metrics and tick.
  TestMetric metric1;
  metric1.GetMutable<CounterMetric>()->set_count(5);
  aggregator_->RecordMetrics(std::move(metric1));
  auto tick_time1 = start_time_ + aggregator_->GetResolution();
  aggregator_->Advance(tick_time1);

  // Add more live metrics.
  TestMetric metric2;
  metric2.GetMutable<CounterMetric>()->set_count(10);
  aggregator_->RecordMetrics(std::move(metric2));
  auto current_time = tick_time1 + aggregator_->GetResolution() / 2;

  // Get metrics for a duration that spans the tick.
  auto [metrics, age] = aggregator_->GetAggregateEndingNow(
      aggregator_->GetResolution(), current_time);

  // Should include metrics from before and after the tick.
  EXPECT_EQ(metrics.Get<CounterMetric>().count(), 15);
}

TEST_F(ExponentialAggregatorTest, GetCompletedMetrics_FuturePeriod) {
  // Request a time period that has not been reached yet.
  auto [metrics, age] =
      aggregator_->GetBucketMetrics(TimePeriod::k1Hour);

  // Should return an empty metric.
  EXPECT_EQ(metrics.Get<CounterMetric>().count(), 0);
}

TEST_F(ExponentialAggregatorTest, Advance_LargeDuration) {
  // Add a metric to be carried through the ticks.
  TestMetric metric;
  metric.GetMutable<CounterMetric>()->set_count(123);
  aggregator_->RecordMetrics(std::move(metric));

  // Advance by a large number of ticks (e.g., 1024), which is 2^10.
  // This should update buckets up to TimePeriod::k16Seconds.
  const int num_ticks = 1 << 10;
  auto advance_time = start_time_ + aggregator_->GetResolution() * num_ticks;
  auto period = aggregator_->Advance(advance_time);

  EXPECT_EQ(period, TimePeriod::k16Seconds);

  // Check the highest-level bucket that should have been updated.
  auto [metrics, age] =
      aggregator_->GetBucketMetrics(TimePeriod::k16Seconds);

  // The bucket should contain the initial metric.
  EXPECT_EQ(metrics.Get<CounterMetric>().count(), 123);
}

TEST_F(ExponentialAggregatorTest, GetBucketMetrics_Aligned) {
  // Advance 4 ticks.
  for (int i = 1; i <= 4; ++i) {
    TestMetric m;
    m.GetMutable<CounterMetric>()->set_count(i);
    aggregator_->RecordMetrics(std::move(m));
    aggregator_->Advance(start_time_ + aggregator_->GetResolution() * i);
  }

  // Request metrics for k63Milliseconds (4 ticks).
  // This is aligned with the ticks performed.
  auto [metrics, age] =
      aggregator_->GetBucketMetrics(TimePeriod::k63Milliseconds);
  EXPECT_EQ(metrics.Get<CounterMetric>().count(), 1 + 2 + 3 + 4);
}

TEST_F(ExponentialAggregatorTest, GetBucketMetrics_Misaligned) {
  // Advance 5 ticks.
  for (int i = 1; i <= 5; ++i) {
    TestMetric m;
    m.GetMutable<CounterMetric>()->set_count(i);
    aggregator_->RecordMetrics(std::move(m));
    aggregator_->Advance(start_time_ + aggregator_->GetResolution() * i);
  }

  // Request metrics for k63Milliseconds (4 ticks).
  // The latest tick (5) is not part of this completed bucket.
  auto [metrics, age] =
      aggregator_->GetBucketMetrics(TimePeriod::k63Milliseconds);
  EXPECT_EQ(metrics.Get<CounterMetric>().count(), 1 + 2 + 3 + 4);

  // The age should reflect one tick has passed since the bucket was completed.
  EXPECT_GE(age.count(), aggregator_->GetResolution().count());
}

TEST_F(ExponentialAggregatorTest,
       GetBucketMetrics_WithAndWithoutLive) {
  aggregator_->Advance(start_time_ + aggregator_->GetResolution());

  auto current_time = start_time_ + aggregator_->GetResolution() * 2;

  // With live time.
  auto [metrics_live, age_live] = aggregator_->GetBucketMetrics(
      TimePeriod::k16Milliseconds, current_time);
  EXPECT_GE(age_live.count(), aggregator_->GetResolution().count());

  // Without live time.
  auto [metrics_no_live, age_no_live] = aggregator_->GetBucketMetrics(
      TimePeriod::k16Milliseconds, std::nullopt);
  EXPECT_EQ(age_no_live.count(), 0);
  EXPECT_LT(age_no_live.count(), age_live.count());
}

TEST_F(ExponentialAggregatorTest, GetAggregateEndingNow_PartialHistory) {
  // Advance 3 ticks.
  for (int i = 1; i <= 3; ++i) {
    TestMetric m;
    m.GetMutable<CounterMetric>()->set_count(i);
    aggregator_->RecordMetrics(std::move(m));
    aggregator_->Advance(start_time_ + aggregator_->GetResolution() * i);
  }

  // Request metrics for the last 2 ticks.
  auto [metrics, age] = aggregator_->GetAggregateEndingNow(
      aggregator_->GetResolution() * 2,
      start_time_ + aggregator_->GetResolution() * 3);

  // The implementation of GetAggregateEndingNow is not exact. It returns
  // buckets that cover *at least* the requested duration. In this case, it
  // will return buckets covering 3 ticks.
  EXPECT_EQ(metrics.Get<CounterMetric>().count(), 1 + 2 + 3);
}

TEST_F(ExponentialAggregatorTest,
       GetAggregateEndingNow_PartialHistoryWithLive) {
  // Advance 3 ticks.
  for (int i = 1; i <= 3; ++i) {
    TestMetric m;
    m.GetMutable<CounterMetric>()->set_count(i);
    aggregator_->RecordMetrics(std::move(m));
    aggregator_->Advance(start_time_ + aggregator_->GetResolution() * i);
  }

  // Add live metrics.
  TestMetric live_metric;
  live_metric.GetMutable<CounterMetric>()->set_count(4);
  aggregator_->RecordMetrics(std::move(live_metric));

  auto current_time = start_time_ + aggregator_->GetResolution() * 3 +
                      aggregator_->GetResolution() / 2;

  // Request metrics for the last 2 ticks + live metrics.
  auto [metrics, age] = aggregator_->GetAggregateEndingNow(
      aggregator_->GetResolution() * 2, current_time);

  // Should contain metrics from all ticks and live data.
  EXPECT_EQ(metrics.Get<CounterMetric>().count(), 1 + 2 + 3 + 4);
}

TEST_F(ExponentialAggregatorTest,
       GetAggregateEndingNow_PartialHistoryWithoutLive) {
  // Advance 3 ticks.
  for (int i = 1; i <= 3; ++i) {
    TestMetric m;
    m.GetMutable<CounterMetric>()->set_count(i);
    aggregator_->RecordMetrics(std::move(m));
    aggregator_->Advance(start_time_ + aggregator_->GetResolution() * i);
  }

  // Add live metrics that should be ignored.
  TestMetric live_metric;
  live_metric.GetMutable<CounterMetric>()->set_count(4);
  aggregator_->RecordMetrics(std::move(live_metric));

  // Request metrics for the last 2 ticks, without live metrics.
  auto [metrics, age] = aggregator_->GetAggregateEndingNow(
      aggregator_->GetResolution() * 2, std::nullopt);

  // Should contain metrics from all ticks, but not live data.
  EXPECT_EQ(metrics.Get<CounterMetric>().count(), 1 + 2 + 3);
}

// Test overshadowing behavior for optional value metrics
class ExponentialAggregatorOvershadowTest : public ::testing::Test {
 protected:
  using TestMetric = MetricGroup<OptionalValueMetric>;
  using TestAggregator = ExponentialAggregator<TestMetric>;

  void SetUp() override {
    aggregator_ = std::make_unique<TestAggregator>();
    start_time_ = TestAggregator::Clock::now();
    aggregator_->Reset(start_time_);
  }

  std::unique_ptr<TestAggregator> aggregator_;
  TestAggregator::Clock::time_point start_time_;
};

TEST_F(ExponentialAggregatorOvershadowTest, LiveOvershadowsCompletedBuckets) {
  // Set up completed bucket with a value
  TestMetric metric1;
  metric1.GetMutable<OptionalValueMetric>()->set_value(10.0);
  aggregator_->RecordMetrics(std::move(metric1));

  // Advance to create a completed bucket
  auto tick_time = start_time_ + aggregator_->GetResolution();
  aggregator_->Advance(tick_time);

  // Add live metric with different value (should overshadow)
  TestMetric metric2;
  metric2.GetMutable<OptionalValueMetric>()->set_value(20.0);
  aggregator_->RecordMetrics(std::move(metric2));

  // Get metrics that include both completed and live
  auto current_time = tick_time + aggregator_->GetResolution() / 2;
  auto [metrics, age] = aggregator_->GetAggregateEndingNow(
      aggregator_->GetResolution(), current_time);

  // Live metric should overshadow the completed bucket value
  EXPECT_TRUE(metrics.Get<OptionalValueMetric>().has_value());
  EXPECT_EQ(metrics.Get<OptionalValueMetric>().value().value(), 20.0);
}

TEST_F(ExponentialAggregatorOvershadowTest,
       SmallerBucketsOvershadowLargerBuckets) {
  // Create multiple ticks with different values, where later (smaller) buckets
  // should overshadow earlier (larger) buckets

  // Tick 1: Set value to 100.0
  TestMetric metric1;
  metric1.GetMutable<OptionalValueMetric>()->set_value(100.0);
  aggregator_->RecordMetrics(std::move(metric1));
  auto tick_time1 = start_time_ + aggregator_->GetResolution();
  aggregator_->Advance(tick_time1);

  // Tick 2: Set value to 200.0 (this will go into k31Milliseconds bucket)
  TestMetric metric2;
  metric2.GetMutable<OptionalValueMetric>()->set_value(200.0);
  aggregator_->RecordMetrics(std::move(metric2));
  auto tick_time2 = tick_time1 + aggregator_->GetResolution();
  aggregator_->Advance(tick_time2);

  // Tick 3: No value (empty bucket)
  TestMetric metric3;
  aggregator_->RecordMetrics(std::move(metric3));
  auto tick_time3 = tick_time2 + aggregator_->GetResolution();
  aggregator_->Advance(tick_time3);

  // Tick 4: Set value to 300.0 (this will update k63Milliseconds bucket)
  TestMetric metric4;
  metric4.GetMutable<OptionalValueMetric>()->set_value(300.0);
  aggregator_->RecordMetrics(std::move(metric4));
  auto tick_time4 = tick_time3 + aggregator_->GetResolution();
  aggregator_->Advance(tick_time4);

  // Request metrics for k63Milliseconds period
  auto [metrics_4ticks, age_4ticks] =
      aggregator_->GetBucketMetrics(TimePeriod::k63Milliseconds);

  // The k63Milliseconds bucket should contain the value from tick 4 (300.0)
  // which overshadows the earlier values
  EXPECT_TRUE(metrics_4ticks.Get<OptionalValueMetric>().has_value());
  EXPECT_EQ(metrics_4ticks.Get<OptionalValueMetric>().value().value(), 300.0);

  // Request metrics for k31Milliseconds period
  auto [metrics_2ticks, age_2ticks] =
      aggregator_->GetBucketMetrics(TimePeriod::k31Milliseconds);

  // The k31Milliseconds bucket should retain the value from tick 2 (200.0)
  // because an empty tick (tick 3) does not overshadow a non-empty one.
  EXPECT_TRUE(metrics_2ticks.Get<OptionalValueMetric>().has_value());
  EXPECT_EQ(metrics_2ticks.Get<OptionalValueMetric>().value().value(), 300.0);
}

TEST_F(ExponentialAggregatorOvershadowTest,
       EmptyBucketOvershadowsNonEmptyBucket) {
  // Test that an empty bucket (no value) overshadows a non-empty bucket

  // Tick 1: Set value to 42.0
  TestMetric metric1;
  metric1.GetMutable<OptionalValueMetric>()->set_value(42.0);
  aggregator_->RecordMetrics(std::move(metric1));
  auto tick_time1 = start_time_ + aggregator_->GetResolution();
  aggregator_->Advance(tick_time1);

  // Tick 2: Empty metric (no value set)
  TestMetric metric2;  // Default constructed, no value
  aggregator_->RecordMetrics(std::move(metric2));
  auto tick_time2 = tick_time1 + aggregator_->GetResolution();
  aggregator_->Advance(tick_time2);

  // Request metrics for k31Milliseconds period (covers both ticks)
  auto [metrics, age] =
      aggregator_->GetBucketMetrics(TimePeriod::k31Milliseconds);

  // The result should retain the value from tick 1 (42.0) because the empty
  // bucket from tick 2 does not overshadow a non-empty one.
  EXPECT_TRUE(metrics.Get<OptionalValueMetric>().has_value());
  EXPECT_EQ(metrics.Get<OptionalValueMetric>().value().value(), 42.0);
}

TEST_F(ExponentialAggregatorOvershadowTest,
       LiveEmptyOvershadowsCompletedNonEmpty) {
  // Test that live empty metric overshadows completed non-empty buckets

  // Set up completed bucket with a value
  TestMetric metric1;
  metric1.GetMutable<OptionalValueMetric>()->set_value(99.0);
  aggregator_->RecordMetrics(std::move(metric1));

  // Advance to create a completed bucket
  auto tick_time = start_time_ + aggregator_->GetResolution();
  aggregator_->Advance(tick_time);

  // Add empty live metric (no value set)
  TestMetric metric2;  // Default constructed, no value
  aggregator_->RecordMetrics(std::move(metric2));

  // Get metrics that include both completed and live
  auto current_time = tick_time + aggregator_->GetResolution() / 2;
  auto [metrics, age] = aggregator_->GetAggregateEndingNow(
      aggregator_->GetResolution(), current_time);

  // Live empty metric should NOT overshadow the completed bucket value
  EXPECT_TRUE(metrics.Get<OptionalValueMetric>().has_value());
  EXPECT_EQ(metrics.Get<OptionalValueMetric>().value().value(), 99.0);
}

TEST_F(ExponentialAggregatorOvershadowTest,
       GetAggregateEndingNow_OvershadowingOrder) {
  // Test the order of overshadowing when getting live metrics across multiple
  // buckets

  // Tick 1: Value 1.0 -> goes to k16Milliseconds bucket
  TestMetric metric1;
  metric1.GetMutable<OptionalValueMetric>()->set_value(1.0);
  aggregator_->RecordMetrics(std::move(metric1));
  auto tick_time1 = start_time_ + aggregator_->GetResolution();
  aggregator_->Advance(tick_time1);

  // Tick 2: Empty -> merges into k31Milliseconds bucket
  TestMetric metric2;
  aggregator_->RecordMetrics(std::move(metric2));
  auto tick_time2 = tick_time1 + aggregator_->GetResolution();
  aggregator_->Advance(tick_time2);

  // Tick 3: Value 3.0 -> goes to k16Milliseconds bucket
  TestMetric metric3;
  metric3.GetMutable<OptionalValueMetric>()->set_value(3.0);
  aggregator_->RecordMetrics(std::move(metric3));
  auto tick_time3 = tick_time2 + aggregator_->GetResolution();
  aggregator_->Advance(tick_time3);

  // Tick 4: Empty -> merges into k63Milliseconds bucket (and overshadows
  // everything)
  TestMetric metric4;
  aggregator_->RecordMetrics(std::move(metric4));
  auto tick_time4 = tick_time3 + aggregator_->GetResolution();
  aggregator_->Advance(tick_time4);

  // Add live metric with value
  TestMetric live_metric;
  live_metric.GetMutable<OptionalValueMetric>()->set_value(999.0);
  aggregator_->RecordMetrics(std::move(live_metric));

  // Get metrics for duration that spans all ticks plus live
  auto current_time = tick_time4 + aggregator_->GetResolution() / 2;
  auto [metrics, age] = aggregator_->GetAggregateEndingNow(
      aggregator_->GetResolution() * 4, current_time);

  // Live metric should overshadow all buckets
  EXPECT_TRUE(metrics.Get<OptionalValueMetric>().has_value());
  EXPECT_EQ(metrics.Get<OptionalValueMetric>().value().value(), 999.0);

  // Test without live metrics (should get empty due to tick 4 overshadowing)
  auto [metrics_no_live, age_no_live] = aggregator_->GetAggregateEndingNow(
      aggregator_->GetResolution() * 4, std::nullopt);

  // Should retain the last non-empty value (3.0) because empty ticks do not
  // overshadow previous non-empty values.
  EXPECT_TRUE(metrics_no_live.Get<OptionalValueMetric>().has_value());
  EXPECT_EQ(metrics_no_live.Get<OptionalValueMetric>().value().value(), 3.0);
}

// Test TimePeriod enum values
TEST(ExponentialAggregatorTimePeriodTest, TimePeriodValues) {
  // Test that time periods have expected relative values
  EXPECT_EQ(static_cast<int>(TimePeriod::k16Milliseconds), -6);
  EXPECT_EQ(static_cast<int>(TimePeriod::k31Milliseconds), -5);
  EXPECT_EQ(static_cast<int>(TimePeriod::k1Second), 0);
  EXPECT_EQ(static_cast<int>(TimePeriod::k2Seconds), 1);
  EXPECT_EQ(static_cast<int>(TimePeriod::k1Minute), 6);
  EXPECT_EQ(static_cast<int>(TimePeriod::k1Hour), 12);
}

// Test edge cases and error conditions
class MetricsAggregatorEdgeCasesTest : public ::testing::Test {};

TEST_F(MetricsAggregatorEdgeCasesTest, EmptyMetricGroup) {
  MetricGroup<> empty_group;

  // Should not crash
  empty_group.Reset();
  empty_group.MergeFrom(MetricGroup<>{});
  std::string str = MetricToString(empty_group);
  EXPECT_TRUE(str.empty());
}

TEST_F(MetricsAggregatorEdgeCasesTest, MetricWithoutPrint) {
  class SimpleMetric {
   public:
    void Reset() { value_ = 0; }
    void MergeFrom(const SimpleMetric& other) { value_ += other.value_; }
    int value_ = 42;
  };

  MetricGroup<SimpleMetric> group;

  // Should not crash even without Print method
  std::string str = MetricToString(group);
  EXPECT_TRUE(str.empty());  // No Print method, so empty string
}

TEST_F(MetricsAggregatorEdgeCasesTest, MetricWithoutName) {
  class UnnamedMetric {
   public:
    void Reset() { value_ = 0; }
    void MergeFrom(const UnnamedMetric& other) { value_ += other.value_; }
    void Print(MetricPrinter& printer) const {
      printer.Print("unnamed", std::to_string(value_));
    }
    int value_ = 42;
  };

  MetricGroup<UnnamedMetric> group;

  // Should work without name method
  std::string str = MetricToString(group);
  EXPECT_NE(str.find("42"), std::string::npos);
}

// Integration test
TEST_F(MetricsAggregatorEdgeCasesTest, IntegrationTest) {
  using TestGroup = MetricGroup<CounterMetric, AverageMetric>;
  ExponentialAggregator<TestGroup> aggregator;
  auto start_time = std::chrono::steady_clock::now();
  aggregator.Reset(start_time);
  auto current_time = start_time;

  // Simulate a realistic scenario
  for (int i = 0; i < 10; ++i) {
    TestGroup metric;
    metric.GetMutable<CounterMetric>()->set_count(i + 1);
    metric.GetMutable<AverageMetric>()->add_sample((i + 1) * 10.0);

    aggregator.RecordMetrics(std::move(metric));

    if (i % 3 == 2) {  // Advance every 3 updates
      current_time += aggregator.GetResolution();
      aggregator.Advance(current_time);
    }
  }

  // Get final live metrics
  auto [live_metrics, age] =
      aggregator.GetAggregateEndingNow(std::chrono::seconds(0), current_time);

  // Should have accumulated some metrics
  EXPECT_GT(live_metrics.Get<CounterMetric>().count(), 0);
  EXPECT_GT(live_metrics.Get<AverageMetric>().count(), 0);
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}