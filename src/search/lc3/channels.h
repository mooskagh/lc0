#pragma once

#include <absl/synchronization/mutex.h>

#include "search/lc3/positions.h"
#include "third_party/moodycamel/blockingconcurrentqueue.h"
#include "third_party/moodycamel/concurrentqueue.h"
#include "utils/freelist.h"

namespace lczero {
namespace lc3 {

// TODO rename to NodeEvent or NodeMessage or something like that.
// TODO Also rename the file.
struct NodeEvent {
  enum class ResultType : uint8_t { kNormal, kTerminal, kCollisionRollback };

  NodeEvent(Variation variation, size_t num_visits)
      : variation(std::move(variation)), num_visits(num_visits) {}

  NodeEvent(Variation variation, size_t num_visits, ResultType result_type)
      : variation(std::move(variation)),
        num_visits(num_visits),
        result_type(result_type) {}

  NodeEvent(Variation variation, size_t num_visits, ResultType result_type,
            float v, float d, float m)
      : variation(std::move(variation)),
        num_visits(num_visits),
        result_type(result_type),
        v(v),
        d(d),
        m(m) {}

  // Input.
  Variation variation;
  size_t num_visits;

  // Result.
  ResultType result_type;
  float v;
  float d;
  float m;
  std::vector<Move> moves;
  std::vector<float> p;
};

using NodeEventPool = FreeListAllocator<NodeEvent, 1024>;

struct NodeEventSender {
 public:
  explicit NodeEventSender(
      moodycamel::BlockingConcurrentQueue<NodeEvent*>* queue)
      : queue_(queue), producer_token_(*queue) {}

  void Enqueue(NodeEvent* event) const {
    queue_->enqueue(producer_token_, event);
  }
  void EnqueueBulk(std::span<NodeEvent*> events) const {
    queue_->enqueue_bulk(producer_token_, events.data(), events.size());
  }

 private:
  moodycamel::BlockingConcurrentQueue<NodeEvent*>* const queue_;
  moodycamel::ProducerToken producer_token_;
};

class NodeEventReceiver {
 public:
  size_t Collect(std::span<NodeEvent*> events, bool block)
      REQUIRES(consumer_mutex_) {
    if (block && !draining_.load(std::memory_order_relaxed)) {
      return queue_.wait_dequeue_bulk(consumer_token_, events.data(),
                                      events.size());
    } else {
      return queue_.try_dequeue_bulk(consumer_token_, events.data(),
                                     events.size());
    }
  }

  absl::Mutex* GetConsumerMutex() { return &consumer_mutex_; }
  NodeEventSender MakeSender() { return NodeEventSender(&queue_); }
  size_t SizeApprox() const { return queue_.size_approx(); }
  void Drain() {
    draining_.store(true, std::memory_order_relaxed);
    queue_.enqueue(nullptr);  // Enqueue a sentinel to signal draining.
  }

 private:
  moodycamel::BlockingConcurrentQueue<NodeEvent*> queue_;
  moodycamel::ConsumerToken consumer_token_{queue_};
  absl::Mutex consumer_mutex_;
  std::atomic<bool> draining_{false};
};

}  // namespace lc3
}  // namespace lczero