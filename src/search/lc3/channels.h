#pragma once

#include "search/lc3/positions.h"
#include "third_party/moodycamel/blockingconcurrentqueue.h"
#include "third_party/moodycamel/concurrentqueue.h"

namespace lczero {
namespace lc3 {

// TODO rename to NodeEvent or NodeMessage or something like that.
// TODO Also rename the file.
struct EvalItem {
  enum class ResultType : uint8_t { kNormal, kTerminal, kCollisionRollback };

  EvalItem(Variation variation, size_t num_visits)
      : variation(std::move(variation)), num_visits(num_visits) {}

  EvalItem(Variation variation, size_t num_visits, ResultType result_type)
      : variation(std::move(variation)),
        num_visits(num_visits),
        result_type(result_type) {}

  EvalItem(Variation variation, size_t num_visits, ResultType result_type,
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

struct EvalItemSender {
 public:
  explicit EvalItemSender(moodycamel::BlockingConcurrentQueue<EvalItem*>* queue)
      : queue_(queue), producer_token_(*queue) {}

  void Enqueue(EvalItem* item) const { queue_->enqueue(producer_token_, item); }
  void EnqueueBulk(std::span<EvalItem*> items) const {
    queue_->enqueue_bulk(producer_token_, items.data(), items.size());
  }

 private:
  moodycamel::BlockingConcurrentQueue<EvalItem*>* const queue_;
  moodycamel::ProducerToken producer_token_;
};

class EvalItemReceiver {
 public:
  size_t Collect(std::span<EvalItem*> items, bool block)
      REQUIRES(consumer_mutex_) {
    if (block) {
      return queue_.wait_dequeue_bulk(consumer_token_, items.data(),
                                      items.size());
    } else {
      return queue_.try_dequeue_bulk(consumer_token_, items.data(),
                                     items.size());
    }
  }

  absl::Mutex* GetConsumerMutex() { return &consumer_mutex_; }
  EvalItemSender MakeSender() { return EvalItemSender(&queue_); }

 private:
  moodycamel::BlockingConcurrentQueue<EvalItem*> queue_;
  moodycamel::ConsumerToken consumer_token_{queue_};
  absl::Mutex consumer_mutex_;
};

}  // namespace lc3
}  // namespace lczero