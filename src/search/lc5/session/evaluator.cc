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

#include "search/lc5/session/evaluator.h"

#include <absl/synchronization/mutex.h>

#include <chrono>
#include <utility>

namespace lczero::lc5 {

Evaluator::Evaluator(const Context& context)
    : context_(context), producing_workers_(context.inboxes.size()) {}

void Evaluator::Start() {
  for (int i = 0; i < context_.settings.eval_threads; ++i) {
    threads_.emplace_back(&Evaluator::Run, this);
  }
}

void Evaluator::Join() {
  queued_jobs_.Close();
  for (auto& thread : threads_) {
    if (thread.joinable()) thread.join();
  }
}

void Evaluator::Push(std::unique_ptr<AsyncJob> job) {
  queued_jobs_.Push(std::move(job));
  RaiseHighWater(context_.metrics.ready_eval_high_water, queued_jobs_.Size());
}

void Evaluator::OnProducerActive() {
  // Only disables the starvation predicate, so it needs no handshake.
  producing_workers_.fetch_add(1);
}

void Evaluator::OnProducerBlocked() {
  if (producing_workers_.fetch_sub(1) == 1) {
    queued_jobs_.NotifyPredicateChanged();
  }
}

void Evaluator::Run() {
  std::unique_ptr<AsyncJob> job;
  while (queued_jobs_.Pop(&job)) {
    if (IsStopping()) {
      ReturnJobToWorker(std::move(job));
      continue;
    }
    jobs_in_collection_.fetch_add(1);
    EvaluateBatch(CollectBatch(std::move(job)));
  }
}

Evaluator::Batch Evaluator::CollectBatch(std::unique_ptr<AsyncJob> job) {
  const size_t target = context_.settings.minibatch_size;
  const auto deadline =
      context_.settings.max_batch_delay_ms == 0
          ? std::chrono::steady_clock::time_point::max()
          : job->queued_at +
                std::chrono::milliseconds(context_.settings.max_batch_delay_ms);
  Batch batch{.computation = context_.backend.CreateComputation()};
  for (;;) {
    AddJobToBatch(*batch.computation, *job);
    batch.jobs.push_back(std::move(job));
    const size_t used = batch.computation->UsedBatchSize();
    if (used >= target || IsStopping()) break;
    // A whole job's size bounds its uncached inputs and must fit before
    // AddInput. A job that doesn't fit stays queued and starts a later batch.
    const auto fits = [&](const std::unique_ptr<AsyncJob>& next) {
      return used + next->materializations.size() <= target;
    };
    // Consume already-ready jobs first, even past the deadline, rather than
    // fragmenting an accumulated backlog into single-job batches.
    if (!queued_jobs_.TryPopIf(&job, fits)) {
      // Immediate-only batches return their jobs instead of waiting on
      // producers that need those cache results. The oldest job's deadline is
      // absolute; zero disables it, but dependency starvation still flushes.
      if (used == 0 || IsStopping()) break;
      queued_jobs_.NotifyPredicateChanged();
      if (!queued_jobs_.PopUntil(
              &job, deadline, [&] { return IsStopping() || IsStarved(); },
              fits)) {
        batch.timed_out = std::chrono::steady_clock::now() >= deadline;
        break;
      }
    }
    jobs_in_collection_.fetch_add(1);
  }
  // Decreasing jobs_in_collection_ only disables the starvation predicate; the
  // increment before PopUntil above is the enabling handshake.
  jobs_in_collection_.fetch_sub(batch.jobs.size());
  return batch;
}

void Evaluator::AddJobToBatch(BackendComputation& computation, AsyncJob& job) {
  for (auto& materialization : job.materializations) {
    if (IsStopping()) break;
    auto& eval = materialization.eval;
    const auto result = computation.AddInput(
        EvalPosition{.pos = materialization.history.GetPositions(),
                     .legal_moves = eval.moves},
        EvalResultPtr{.q = &eval.leaf.q,
                      .d = &eval.leaf.d,
                      .m = &eval.leaf.m,
                      .p = eval.priors});
    if (result == BackendComputation::FETCHED_IMMEDIATELY) {
      materialization.state = Materialization::State::kEvaluated;
      context_.metrics.cache_hits.fetch_add(1);
    } else {
      materialization.state = Materialization::State::kQueued;
    }
  }
}

void Evaluator::EvaluateBatch(Batch batch) {
  const size_t used = batch.computation->UsedBatchSize();
  if (used > 0 && !IsStopping()) {
    if (used < static_cast<size_t>(context_.settings.minibatch_size)) {
      (batch.timed_out ? context_.metrics.partial_timeout_flushes
                       : context_.metrics.partial_starvation_flushes)
          .fetch_add(1);
    }
    context_.metrics.evaluation_batches.fetch_add(1);
    context_.metrics.nn_evaluations.fetch_add(used);
    batch.computation->ComputeBlocking();
    for (auto& job : batch.jobs) {
      for (auto& materialization : job->materializations) {
        if (materialization.state == Materialization::State::kQueued) {
          materialization.state = Materialization::State::kEvaluated;
        }
      }
    }
  }
  // Destroy the computation while the jobs it references are still alive.
  batch.computation.reset();
  for (auto& job : batch.jobs) ReturnJobToWorker(std::move(job));
}

bool Evaluator::IsStarved() const {
  // Executing, store or returned jobs can still unblock a producer. Only
  // predict starvation when every outstanding job is held by a collector and
  // no worker can send a job independently; the deadline bounds the
  // uncertainty.
  if (producing_workers_.load() != 0) return false;
  // Zero disables the timeout, not dependency flushing: with no deadline,
  // currently blocked producers cannot justify an unbounded wait.
  return context_.settings.max_batch_delay_ms == 0 ||
         context_.jobs_in_flight.load() <= jobs_in_collection_.load();
}

void Evaluator::ReturnJobToWorker(std::unique_ptr<AsyncJob> job) {
  context_.inboxes[job->worker].AddCompletion(std::move(job));
  context_.metrics.inbox_notifications.fetch_add(1);
}

}  // namespace lczero::lc5
