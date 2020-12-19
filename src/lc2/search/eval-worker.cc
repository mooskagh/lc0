/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors

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

#include "lc2/search/eval-worker.h"

#include "lc2/search/search.h"
#include "neural/encoder.h"
#include "utils/exception.h"
#include "utils/logging.h"

namespace lczero {
namespace lc2 {
namespace {

struct EvalPosition {
  std::unique_ptr<Message> msg;
  int transform;
};

void RunMoveGen(Message* msg) {
  msg->eval_result = Message::Eval{};
  msg->eval_result->edges =
      msg->position_history.Last().GetBoard().GenerateLegalMoves();
}

EvalPosition PrepareForEval(std::unique_ptr<Message> msg,
                            NetworkComputation* computation,
                            const NetworkCapabilities& caps) {
  assert(msg->arity == 1);
  EvalPosition res{std::move(msg), 0};

  auto planes =
      EncodePositionForNN(caps.input_format, res.msg->position_history, 8,
                          FillEmptyHistory::FEN_ONLY, &res.transform);
  computation->AddInput(std::move(planes));
  // It's better to run movegen immediately than later when NN eval is done, as
  // at this point there's a chance that batch is not ready yet so we can use
  // this time.
  RunMoveGen(res.msg.get());
  return res;
}
}  // namespace

EvalWorker::EvalWorker(Search* search, Network* network)
    : search_(search), network_(network) {}

void EvalWorker::RunBlocking() {
  // TODO add an exit condition.
  while (true) ProcessOneBatch();
}

void EvalWorker::ProcessOneBatch() {
  // TODO Move this to command line params.
  constexpr int kMinBatch = 2;

  auto computation = network_->NewComputation();
  const auto& caps = network_->GetCapabilities();

  // Number of nodes for which node workers gave up finding something for eval.
  int num_skip_nodes = 0;
  std::vector<EvalPosition> evals;

  // The batch is ready for the eval when there is at least kMinBatch nodes
  // including skip-nodes, and at least 1 real node for eval.
  do {
    auto new_msgs = channel_.DequeueEverything();
    for (auto& msg : new_msgs) {
      switch (msg->type) {
        case Message::kEvalEval:
          evals.push_back(
              PrepareForEval(std::move(msg), computation.get(), caps));
          break;
        case Message::kEvalSkip:
          num_skip_nodes += msg->arity;
          break;
        default:
          throw Exception("Unexpected message type " +
                          std::to_string(msg->type) + " in eval worker.");
      }
    }
  } while (num_skip_nodes + evals.size() < kMinBatch || evals.empty());

  // Now we have a batch ready for eval, so do a NN computation.
  computation->ComputeBlocking();

  // Sending the computation results.
  for (int i = 0; i < static_cast<int>(evals.size()); ++i) {
    auto msg = std::move(evals[i].msg);
    auto& eval = *msg->eval_result;
    eval.wdl = NT::WDLFromComputation(computation.get(), i);
    std::vector<int> move_indices(eval.edges.size());
    std::transform(eval.edges.begin(), eval.edges.end(), move_indices.begin(),
                   [transform = evals[i].transform](const Move& move) {
                     return move.as_nn_index(transform);
                   });
    eval.p_edge = NT::PFromComputation(computation.get(), i, move_indices);
    msg->type = Message::kNodeBackProp;
    assert(msg->arity == 1);
    search_->DispatchToNodes(std::move(msg));
  }

  // If there were any skip evals, free them up.
  if (num_skip_nodes > 0) {
    auto msg = std::make_unique<Message>();
    msg->arity = num_skip_nodes;
    msg->type = Message::kRootEvalSkipped;
    search_->DispatchToRoot(std::move(msg));
  }
}

}  // namespace lc2
}  // namespace lczero