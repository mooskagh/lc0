#include "utils/stats_aggregator.h"

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <thread>

namespace lczero {

// Mock metric classes for testing
class CounterMetric {
 public:
  CounterMetric() : count_(0) {}
  CounterMetric(int count) : count_(count) {}

  void Reset() { count_ = 0; }

  void MergeFrom(const CounterMetric& other) { count_ += other.count_; }

  std::string_view name() const { return "counter"; }

  std::string ToString() const { return std::to_string(count_); }

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

  std::string_view name() const { return "average"; }

  std::string ToString() const {
    if (count_ == 0) return "0";
    return std::to_string(sum_ / count_);
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

  std::string ToString() const {
    return has_value_ ? std::to_string(max_value_) : "no_value";
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

TEST_F(MetricGroupTest, ToString) {
  // Set up data
  group_.GetMutable<CounterMetric>()->set_count(42);
  group_.GetMutable<AverageMetric>()->add_sample(10.0);
  group_.GetMutable<AverageMetric>()->add_sample(20.0);
  group_.GetMutable<MaxMetric>()->set_value(100.0);

  std::string result = group_.ToString();

  // Should contain all metric names and values
  EXPECT_NE(result.find("counter"), std::string::npos);
  EXPECT_NE(result.find("42"), std::string::npos);
  EXPECT_NE(result.find("average"), std::string::npos);
  EXPECT_NE(result.find("15"), std::string::npos);  // (10+20)/2
  EXPECT_NE(result.find("100"), std::string::npos);
}

// Test ExponentialAggregator functionality
class ExponentialAggregatorTest : public ::testing::Test {
 protected:
  using TestMetric = MetricGroup<CounterMetric, AverageMetric>;

  void SetUp() override {
    // Create a fresh aggregator for each test to avoid state contamination
    aggregator_ = std::make_unique<ExponentialAggregator<TestMetric>>();
  }

  std::unique_ptr<ExponentialAggregator<TestMetric>> aggregator_;
};

TEST_F(ExponentialAggregatorTest, UpdateLiveStats) {
  TestMetric metric;
  metric.GetMutable<CounterMetric>()->set_count(10);
  metric.GetMutable<AverageMetric>()->add_sample(5.0);

  // Update live stats
  aggregator_->UpdateLiveStats(std::move(metric));

  // The original metric should be reset after move
  EXPECT_EQ(metric.Get<CounterMetric>().count(), 0);
  EXPECT_EQ(metric.Get<AverageMetric>().count(), 0);

  // Get live stats to verify they were updated
  auto [live_stats, age] = aggregator_->GetLiveStatsOfAtLeast(0.0f, true);
  EXPECT_EQ(live_stats.Get<CounterMetric>().count(), 10);
  EXPECT_EQ(live_stats.Get<AverageMetric>().average(), 5.0);
}

TEST_F(ExponentialAggregatorTest, MultipleUpdatesLiveStats) {
  // Update multiple times
  for (int i = 1; i <= 5; ++i) {
    TestMetric metric;
    metric.GetMutable<CounterMetric>()->set_count(i);
    metric.GetMutable<AverageMetric>()->add_sample(i * 2.0);
    aggregator_->UpdateLiveStats(std::move(metric));
  }

  // Get live stats
  auto [live_stats, age] = aggregator_->GetLiveStatsOfAtLeast(0.0f, true);
  EXPECT_EQ(live_stats.Get<CounterMetric>().count(), 15);     // 1+2+3+4+5
  EXPECT_EQ(live_stats.Get<AverageMetric>().average(), 6.0);  // (2+4+6+8+10)/5
}

TEST_F(ExponentialAggregatorTest, Tick) {
  // Add some live stats
  TestMetric metric;
  metric.GetMutable<CounterMetric>()->set_count(10);
  aggregator_->UpdateLiveStats(std::move(metric));

  // Tick to move live stats to buckets
  auto period = aggregator_->Tick();

  // Should return the base time period
  EXPECT_EQ(period, ExponentialAggregator<TestMetric>::k16Milliseconds);

  // Live stats should be empty after tick
  auto [live_stats, age] = aggregator_->GetLiveStatsOfAtLeast(0.0f, true);
  EXPECT_EQ(live_stats.Get<CounterMetric>().count(), 0);
}

TEST_F(ExponentialAggregatorTest, MultipleTicks) {
  // Add stats and tick multiple times to test bucket management
  for (int i = 0; i < 8; ++i) {
    TestMetric metric;
    metric.GetMutable<CounterMetric>()->set_count(1);
    aggregator_->UpdateLiveStats(std::move(metric));

    auto period = aggregator_->Tick();

    switch (i) {
      case 0:
        EXPECT_EQ(period, ExponentialAggregator<TestMetric>::k16Milliseconds);
        break;
      case 1:
        EXPECT_EQ(period, ExponentialAggregator<TestMetric>::k31Milliseconds);
        break;
      case 2:
        EXPECT_EQ(period, ExponentialAggregator<TestMetric>::k16Milliseconds);
        break;
      case 3:
        EXPECT_EQ(period, ExponentialAggregator<TestMetric>::k63Milliseconds);
        break;
      case 4:
        EXPECT_EQ(period, ExponentialAggregator<TestMetric>::k16Milliseconds);
        break;
      case 5:
        EXPECT_EQ(period, ExponentialAggregator<TestMetric>::k31Milliseconds);
        break;
      case 6:
        EXPECT_EQ(period, ExponentialAggregator<TestMetric>::k16Milliseconds);
        break;
      case 7:
        EXPECT_EQ(period, ExponentialAggregator<TestMetric>::k125Milliseconds);
        break;
    }
  }
}

TEST_F(ExponentialAggregatorTest, GetCompletedStats) {
  // Add some stats and tick to create completed buckets
  TestMetric metric;
  metric.GetMutable<CounterMetric>()->set_count(5);
  aggregator_->UpdateLiveStats(std::move(metric));
  aggregator_->Tick();

  // Get completed stats for base period
  auto [stats, age] = aggregator_->GetCompletedStatsAndAgeSeconds(
      ExponentialAggregator<TestMetric>::k16Milliseconds, false);

  // Age should be non-negative
  EXPECT_GE(age, 0.0f);
}

TEST_F(ExponentialAggregatorTest, GetLiveStatsOfAtLeast) {
  // Add some live stats
  TestMetric metric;
  metric.GetMutable<CounterMetric>()->set_count(10);
  metric.GetMutable<AverageMetric>()->add_sample(20.0);
  aggregator_->UpdateLiveStats(std::move(metric));

  // Get live stats requiring at least 0 seconds (should include current)
  auto [stats, age] = aggregator_->GetLiveStatsOfAtLeast(0.0f, true);

  EXPECT_EQ(stats.Get<CounterMetric>().count(), 10);
  EXPECT_EQ(stats.Get<AverageMetric>().average(), 20.0);
  EXPECT_GE(age, 0.0f);

  // Get live stats requiring more time than available (should be empty)
  auto [empty_stats, empty_age] =
      aggregator_->GetLiveStatsOfAtLeast(1000.0f, false);
  EXPECT_EQ(empty_stats.Get<CounterMetric>().count(), 0);
}

// Test TimePeriod enum values
TEST(ExponentialAggregatorTimePeriodTest, TimePeriodValues) {
  // Test that time periods have expected relative values
  using TimePeriod = ExponentialAggregator<CounterMetric>::TimePeriod;

  EXPECT_EQ(static_cast<int>(TimePeriod::k16Milliseconds), -6);
  EXPECT_EQ(static_cast<int>(TimePeriod::k31Milliseconds), -5);
  EXPECT_EQ(static_cast<int>(TimePeriod::k1Second), 0);
  EXPECT_EQ(static_cast<int>(TimePeriod::k2Seconds), 1);
  EXPECT_EQ(static_cast<int>(TimePeriod::k1Minute), 6);
  EXPECT_EQ(static_cast<int>(TimePeriod::k1Hour), 12);
}

// Test edge cases and error conditions
class StatsAggregatorEdgeCasesTest : public ::testing::Test {};

TEST_F(StatsAggregatorEdgeCasesTest, EmptyMetricGroup) {
  MetricGroup<> empty_group;

  // Should not crash
  empty_group.Reset();
  empty_group.MergeFrom(MetricGroup<>{});
  std::string str = empty_group.ToString();
  EXPECT_TRUE(str.empty());
}

TEST_F(StatsAggregatorEdgeCasesTest, MetricWithoutToString) {
  class SimpleMetric {
   public:
    void Reset() { value_ = 0; }
    void MergeFrom(const SimpleMetric& other) { value_ += other.value_; }
    int value_ = 42;
  };

  MetricGroup<SimpleMetric> group;

  // Should not crash even without ToString method
  std::string str = group.ToString();
  EXPECT_TRUE(str.empty());  // No ToString method, so empty string
}

TEST_F(StatsAggregatorEdgeCasesTest, MetricWithoutName) {
  class UnnamedMetric {
   public:
    void Reset() { value_ = 0; }
    void MergeFrom(const UnnamedMetric& other) { value_ += other.value_; }
    std::string ToString() const { return std::to_string(value_); }
    int value_ = 42;
  };

  MetricGroup<UnnamedMetric> group;

  // Should work without name method
  std::string str = group.ToString();
  EXPECT_NE(str.find("42"), std::string::npos);
}

// Integration test
TEST_F(StatsAggregatorEdgeCasesTest, IntegrationTest) {
  using TestGroup = MetricGroup<CounterMetric, AverageMetric>;
  ExponentialAggregator<TestGroup> aggregator;

  // Simulate a realistic scenario
  for (int i = 0; i < 10; ++i) {
    TestGroup metric;
    metric.GetMutable<CounterMetric>()->set_count(i + 1);
    metric.GetMutable<AverageMetric>()->add_sample((i + 1) * 10.0);

    aggregator.UpdateLiveStats(std::move(metric));

    if (i % 3 == 2) {  // Tick every 3 updates
      aggregator.Tick();
    }
  }

  // Get final live stats
  auto [live_stats, age] = aggregator.GetLiveStatsOfAtLeast(0.0f, true);

  // Should have accumulated some stats
  EXPECT_GT(live_stats.Get<CounterMetric>().count(), 0);
  EXPECT_GT(live_stats.Get<AverageMetric>().count(), 0);
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
