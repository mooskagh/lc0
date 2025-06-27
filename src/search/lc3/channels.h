#pragma once

#include "search/lc3/positions.h"
#include "third_party/moodycamel/blockingconcurrentqueue.h"
#include "third_party/moodycamel/concurrentqueue.h"

namespace lczero {
namespace lc3 {

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

struct EvalItemSource {
 public:
  explicit EvalItemSource(moodycamel::BlockingConcurrentQueue<EvalItem*>* queue)
      : queue_(queue), producer_token_(*queue) {}

  void Enqueue(EvalItem* item) const { queue_->enqueue(producer_token_, item); }
  void EnqueueBulk(std::span<EvalItem*> items) const {
    queue_->enqueue_bulk(producer_token_, items.data(), items.size());
  }

 private:
  moodycamel::BlockingConcurrentQueue<EvalItem*>* const queue_;
  moodycamel::ProducerToken producer_token_;
};

class EvalItemSink {
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

 private:
  moodycamel::BlockingConcurrentQueue<EvalItem*> queue_;
  moodycamel::ConsumerToken consumer_token_{queue_};
  absl::Mutex consumer_mutex_;
  friend class SearchChannels;
};

struct GatherWorkerChannels {
  EvalItemSource evals;
  EvalItemSource backprop;
};

struct EvalWorkerChannels {
  EvalItemSink* const eval_sink;
  EvalItemSource backprop;
};

struct BackpropWorkerChannels {
  EvalItemSink* const backprop_sink;
};

class SearchChannels {
 public:
  GatherWorkerChannels MakeGatherWorkerChannels() {
    return GatherWorkerChannels{
        EvalItemSource(&eval_tasks_channel_.queue_),
        EvalItemSource(&backprop_tasks_channel_.queue_)};
  }

  EvalWorkerChannels MakeEvalWorkerChannels() {
    return EvalWorkerChannels(&eval_tasks_channel_,
                              EvalItemSource(&backprop_tasks_channel_.queue_));
  }

  BackpropWorkerChannels MakeBackpropWorkerChannels() {
    return BackpropWorkerChannels(&backprop_tasks_channel_);
  }

 private:
  EvalItemSink eval_tasks_channel_;
  EvalItemSink backprop_tasks_channel_;
};

}  // namespace lc3
}  // namespace lczero