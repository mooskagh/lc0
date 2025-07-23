#include <gtest/gtest.h>

#include <chrono>
#include <memory>
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

TEST_F(ExponentialAggregatorTest, UpdateLiveMetrics) {
  TestMetric metric;
  metric.GetMutable<CounterMetric>()->set_count(10);
  metric.GetMutable<AverageMetric>()->add_sample(5.0);

  // Update live metrics
  aggregator_->UpdateLiveMetrics(std::move(metric));

  // The original metric should be reset after move
  EXPECT_EQ(metric.Get<CounterMetric>().count(), 0);
  EXPECT_EQ(metric.Get<AverageMetric>().count(), 0);

  // Get live metrics to verify they were updated
  auto [live_metrics, age] =
      aggregator_->GetLiveMetricsAtLeast(TestAggregator::Duration::zero());
  EXPECT_EQ(live_metrics.Get<CounterMetric>().count(), 10);
  EXPECT_EQ(live_metrics.Get<AverageMetric>().average(), 5.0);
}

TEST_F(ExponentialAggregatorTest, MultipleUpdatesLiveMetrics) {
  // Update multiple times
  for (int i = 1; i <= 5; ++i) {
    TestMetric metric;
    metric.GetMutable<CounterMetric>()->set_count(i);
    metric.GetMutable<AverageMetric>()->add_sample(i * 2.0);
    aggregator_->UpdateLiveMetrics(std::move(metric));
  }

  // Get live metrics
  auto [live_metrics, age] =
      aggregator_->GetLiveMetricsAtLeast(TestAggregator::Duration::zero());
  EXPECT_EQ(live_metrics.Get<CounterMetric>().count(), 15);     // 1+2+3+4+5
  EXPECT_EQ(live_metrics.Get<AverageMetric>().average(), 6.0);  // (2+4+6+8+10)/5
}

TEST_F(ExponentialAggregatorTest, Advance) {
  // Add some live metrics
  TestMetric metric;
  metric.GetMutable<CounterMetric>()->set_count(10);
  aggregator_->UpdateLiveMetrics(std::move(metric));

  // Advance to move live metrics to buckets
  auto tick_time = start_time_ + std::chrono::milliseconds(16);
  auto period = aggregator_->Advance(tick_time);

  // Should return the base time period
  EXPECT_EQ(period, TimePeriod::k16Milliseconds);

  // Live metrics should be empty after tick
  auto [live_metrics, age] =
      aggregator_->GetLiveMetricsAtLeast(TestAggregator::Duration::zero(), tick_time);
  EXPECT_EQ(live_metrics.Get<CounterMetric>().count(), 0);
}

TEST_F(ExponentialAggregatorTest, MultipleAdvances) {
  // Add metrics and tick multiple times to test bucket management
  auto current_time = start_time_;
  for (int i = 0; i < 8; ++i) {
    TestMetric metric;
    metric.GetMutable<CounterMetric>()->set_count(1);
    aggregator_->UpdateLiveMetrics(std::move(metric));

    current_time += std::chrono::milliseconds(16);
    auto period = aggregator_->Advance(current_time);

    switch (i) {
      case 0:
        EXPECT_EQ(period, TimePeriod::k16Milliseconds);
        break;
      case 1:
        EXPECT_EQ(period, TimePeriod::k31Milliseconds);
        break;
      case 2:
        EXPECT_EQ(period, TimePeriod::k16Milliseconds);
        break;
      case 3:
        EXPECT_EQ(period, TimePeriod::k63Milliseconds);
        break;
      case 4:
        EXPECT_EQ(period, TimePeriod::k16Milliseconds);
        break;
      case 5:
        EXPECT_EQ(period, TimePeriod::k31Milliseconds);
        break;
      case 6:
        EXPECT_EQ(period, TimePeriod::k16Milliseconds);
        break;
      case 7:
        EXPECT_EQ(period, TimePeriod::k125Milliseconds);
        break;
    }
  }
}

TEST_F(ExponentialAggregatorTest, GetCompletedMetrics) {
  // Add some metrics and tick to create completed buckets
  TestMetric metric;
  metric.GetMutable<CounterMetric>()->set_count(5);
  aggregator_->UpdateLiveMetrics(std::move(metric));
  auto tick_time = start_time_ + std::chrono::milliseconds(16);
  aggregator_->Advance(tick_time);

  // Get completed metrics for base period with time
  auto current_time = tick_time + std::chrono::milliseconds(10);
  auto [metrics, age] = aggregator_->GetCompletedMetricsAndAge(
      TimePeriod::k16Milliseconds, current_time);

  // Age should be non-negative
  EXPECT_GE(age.count(), 0);

  // Get completed metrics without time (should exclude live time)
  auto [metrics_no_time, age_no_time] =
      aggregator_->GetCompletedMetricsAndAge(TimePeriod::k16Milliseconds,
                                             std::nullopt);

  // Age without live time should be less than or equal to age with live time
  EXPECT_LE(age_no_time, age);
}

TEST_F(ExponentialAggregatorTest, GetLiveMetricsOfAtLeast) {
  // Add some live metrics
  TestMetric metric;
  metric.GetMutable<CounterMetric>()->set_count(10);
  metric.GetMutable<AverageMetric>()->add_sample(20.0);
  aggregator_->UpdateLiveMetrics(std::move(metric));

  auto current_time = start_time_ + std::chrono::milliseconds(10);

  // Get live metrics requiring at least 0 seconds (should include current)
  auto [metrics, age] =
      aggregator_->GetLiveMetricsAtLeast(TestAggregator::Duration::zero(), current_time);

  EXPECT_EQ(metrics.Get<CounterMetric>().count(), 10);
  EXPECT_EQ(metrics.Get<AverageMetric>().average(), 20.0);
  EXPECT_GE(age.count(), 0);

  // Get live metrics without including live metrics (should be empty)
  auto [empty_metrics, empty_age] = aggregator_->GetLiveMetricsAtLeast(
      std::chrono::seconds(1000), std::nullopt);
  EXPECT_EQ(empty_metrics.Get<CounterMetric>().count(), 0);
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

    aggregator.UpdateLiveMetrics(std::move(metric));

    if (i % 3 == 2) {  // Advance every 3 updates
      current_time += std::chrono::milliseconds(16);
      aggregator.Advance(current_time);
    }
  }

  // Get final live metrics
  auto [live_metrics, age] =
      aggregator.GetLiveMetricsAtLeast(std::chrono::seconds(0), current_time);

  // Should have accumulated some metrics
  EXPECT_GT(live_metrics.Get<CounterMetric>().count(), 0);
  EXPECT_GT(live_metrics.Get<AverageMetric>().count(), 0);
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}