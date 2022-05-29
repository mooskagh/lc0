/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2022 The LCZero Authors

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

#include "mcts/search-responder.h"

#include "chess/callbacks.h"
#include "mcts/search.h"

namespace lczero {
namespace {
// Maximum delay between outputting "uci info" when nothing interesting happens.
const int kUciInfoMinimumFrequencyMs = 5000;
}  // namespace

Search::Responder::Responder(const Search& search,
                             std::unique_ptr<UciResponder> responder)
    : search_(search), responder_(std::move(responder)) {}

int Search::Responder::Depth() const REQUIRES_SHARED(search_.nodes_mutex_) {
  return search_.cum_depth_ /
         (search_.total_playouts_ ? search_.total_playouts_ : 1);
}

void Search::Responder::MaybeOutputInfo() const
    REQUIRES_SHARED(search_.nodes_mutex_) REQUIRES(search_.counters_mutex_) {
  if (!search_.bestmove_is_sent_ && search_.current_best_edge_ &&
      (search_.current_best_edge_.edge() != previous_edge_ ||
       previous_info_.depth != Depth() ||
       previous_info_.seldepth != search_.max_depth_ ||
       previous_info_.time + kUciInfoMinimumFrequencyMs <
           search_.GetTimeSinceStart())) {
    SendUciInfo();
    if (search_.params_.GetLogLiveStats()) SendMovesStats();
    if (search_.stop_.load(std::memory_order_acquire) &&
        !search_.ok_to_respond_bestmove_) {
      OutputComment(
          "WARNING: Search has reached limit and does not make any progress.");
    }
  }
}

void Search::Responder::SendUciInfo() const REQUIRES(search_.nodes_mutex_)
    REQUIRES(search_.counters_mutex_) {
  const auto& params = search_.params_;
  const auto& root_node = search_.root_node_;
  const auto max_pv = params.GetMultiPv();
  const auto edges = search_.GetBestChildrenNoTemperature(root_node, max_pv, 0);
  const auto score_type = params.GetScoreType();
  const auto per_pv_counters = params.GetPerPvCounters();
  const auto display_cache_usage = params.GetDisplayCacheUsage();
  const auto draw_score = search_.GetDrawScore(false);

  std::vector<ThinkingInfo> uci_infos;

  // Info common for all multipv variants.
  ThinkingInfo common_info;
  common_info.depth = Depth();
  common_info.seldepth = search_.max_depth_;
  common_info.time = search_.GetTimeSinceStart();
  if (!per_pv_counters) {
    common_info.nodes = search_.total_playouts_ + search_.initial_visits_;
  }
  if (display_cache_usage) {
    common_info.hashfull = search_.cache_->GetSize() * 1000LL /
                           std::max(search_.cache_->GetCapacity(), 1);
  }
  if (search_.nps_start_time_) {
    const auto time_since_first_batch_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - *search_.nps_start_time_)
            .count();
    if (time_since_first_batch_ms > 0) {
      common_info.nps =
          search_.total_playouts_ * 1000 / time_since_first_batch_ms;
    }
  }
  common_info.tb_hits = search_.tb_hits_.load(std::memory_order_acquire);

  int multipv = 0;
  const auto default_q = -root_node->GetQ(-draw_score);
  const auto default_wl = -root_node->GetWL();
  const auto default_d = root_node->GetD();
  for (const auto& edge : edges) {
    ++multipv;
    uci_infos.emplace_back(common_info);
    auto& uci_info = uci_infos.back();
    const auto wl = edge.GetWL(default_wl);
    const auto floatD = edge.GetD(default_d);
    const auto q = edge.GetQ(default_q, draw_score);
    if (edge.IsTerminal() && wl != 0.0f) {
      uci_info.mate = std::copysign(
          std::round(edge.GetM(0.0f)) / 2 + (edge.IsTbTerminal() ? 101 : 1),
          wl);
    } else if (score_type == "centipawn_with_drawscore") {
      uci_info.score = 90 * tan(1.5637541897 * q);
    } else if (score_type == "centipawn") {
      uci_info.score = 90 * tan(1.5637541897 * wl);
    } else if (score_type == "centipawn_2019") {
      uci_info.score = 295 * wl / (1 - 0.976953126 * std::pow(wl, 14));
    } else if (score_type == "centipawn_2018") {
      uci_info.score = 290.680623072 * tan(1.548090806 * wl);
    } else if (score_type == "win_percentage") {
      uci_info.score = wl * 5000 + 5000;
    } else if (score_type == "Q") {
      uci_info.score = q * 10000;
    } else if (score_type == "W-L") {
      uci_info.score = wl * 10000;
    }

    auto w =
        std::max(0, static_cast<int>(std::round(500.0 * (1.0 + wl - floatD))));
    auto l =
        std::max(0, static_cast<int>(std::round(500.0 * (1.0 - wl - floatD))));
    // Using 1000-w-l so that W+D+L add up to 1000.0.
    auto d = 1000 - w - l;
    if (d < 0) {
      w = std::min(1000, std::max(0, w + d / 2));
      l = 1000 - w;
      d = 0;
    }
    uci_info.wdl = ThinkingInfo::WDL{w, d, l};
    if (search_.network_->GetCapabilities().has_mlh()) {
      uci_info.moves_left =
          static_cast<int>((1.0f + edge.GetM(1.0f + root_node->GetM())) / 2.0f);
    }
    if (max_pv > 1) uci_info.multipv = multipv;
    if (per_pv_counters) uci_info.nodes = edge.GetN();
    bool flip = search_.played_history_.IsBlackToMove();
    int depth = 0;
    for (auto iter = edge; iter;
         iter = search_.GetBestChildNoTemperature(iter.node(), depth),
              flip = !flip) {
      uci_info.pv.push_back(iter.GetMove(flip));
      if (!iter.node()) break;  // Last edge was dangling, cannot continue.
      depth += 1;
    }
  }

  if (!uci_infos.empty()) previous_info_ = uci_infos.front();
  if (search_.current_best_edge_ && !edges.empty()) {
    previous_edge_ = search_.current_best_edge_.edge();
  }

  responder_->OutputThinkingInfo(&uci_infos);
}

void Search::Responder::SendMovesStats() const
    REQUIRES(search_.counters_mutex_) {
  auto move_stats = GetVerboseStats(search_.root_node_);

  if (search_.params_.GetVerboseStats()) {
    std::vector<ThinkingInfo> infos;
    std::transform(move_stats.begin(), move_stats.end(),
                   std::back_inserter(infos), [](const std::string& line) {
                     ThinkingInfo info;
                     info.comment = line;
                     return info;
                   });
    responder_->OutputThinkingInfo(&infos);
  } else {
    LOGFILE << "=== Move stats:";
    for (const auto& line : move_stats) LOGFILE << line;
  }
  for (auto& edge : search_.root_node_->Edges()) {
    if (!(edge.GetMove(search_.played_history_.IsBlackToMove()) ==
          search_.final_bestmove_)) {
      continue;
    }
    if (edge.HasNode()) {
      LOGFILE << "--- Opponent moves after: "
              << search_.final_bestmove_.as_string();
      for (const auto& line : GetVerboseStats(edge.node())) {
        LOGFILE << line;
      }
    }
  }
}

}  // namespace lczero

/*
syntax = "proto2";

package pblczero;

// Messages are optimized to look nicer in JSON rather than protobuf.

message WDL {
    optional float w = 1;
    optional float d = 2;
    optional float l = 3;
}

message Evaluation {
   optional int32 cp = 1;
   optional float winprob = 2;
   optional WDL wdl = 3;
   optional int32 mate = 4;
}

message Lc0MoveInfo {
    optional uint64 n = 1;
    optional uint32 n_in_flight = 2;
    optional float p = 3;
    optional float wl = 4;
    optional float d = 5;
    optional float ml = 6;
    optional float q = 7;
    optional float v = 8;
    optional float u = 9;
    optional float s = 10;
    optional float visited_policy = 11;
}

message NodeInfo {
    repeated string position = 1;
    optional Evaluation eval = 2;
    optional uint32 depth = 3;
    optional uint32 seldepth = 4;
    optional uint64 nodes = 5;
    optional uint32 nps = 6;
    repeated string pv = 7;
    optional uint64 tbhits = 8;
    optional string comment = 9;
    optional Lc0MoveInfo lc0_info = 10;
}

message JsonInfo {
    optional NodeInfo posinfo = 1;
    repeated NodeInfo moves = 2;
    // Time management, cache usage.
}
*/

/*
std::vector<std::string> Search::GetVerboseStats(Node* node) const {
  assert(node == root_node_ || node->GetParent() == root_node_);
  const bool is_root = (node == root_node_);
  const bool is_odd_depth = !is_root;
  const bool is_black_to_move = (played_history_.IsBlackToMove() == is_root);
  const float draw_score = GetDrawScore(is_odd_depth);
  const float fpu = GetFpu(params_, node, is_root, draw_score);
  const float cpuct = ComputeCpuct(params_, node->GetN(), is_root);
  const float U_coeff =
      cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
  std::vector<EdgeAndNode> edges;
  for (const auto& edge : node->Edges()) edges.push_back(edge);

  std::sort(edges.begin(), edges.end(),
            [&fpu, &U_coeff, &draw_score](EdgeAndNode a, EdgeAndNode b) {
              return std::forward_as_tuple(
                         a.GetN(), a.GetQ(fpu, draw_score) + a.GetU(U_coeff)) <
                     std::forward_as_tuple(
                         b.GetN(), b.GetQ(fpu, draw_score) + b.GetU(U_coeff));
            });

  auto print = [](auto* oss, auto pre, auto v, auto post, auto w, int p = 0) {
    *oss << pre << std::setw(w) << std::setprecision(p) << v << post;
  };
  auto print_head = [&](auto* oss, auto label, int i, auto n, auto f, auto p) {
    *oss << std::fixed;
    print(oss, "", label, " ", 5);
    print(oss, "(", i, ") ", 4);
    *oss << std::right;
    print(oss, "N: ", n, " ", 7);
    print(oss, "(+", f, ") ", 2);
    print(oss, "(P: ", p * 100, "%) ", 5, p >= 0.99995f ? 1 : 2);
  };
  auto print_stats = [&](auto* oss, const auto* n) {
    const auto sign = n == node ? -1 : 1;
    if (n) {
      print(oss, "(WL: ", sign * n->GetWL(), ") ", 8, 5);
      print(oss, "(D: ", n->GetD(), ") ", 5, 3);
      print(oss, "(M: ", n->GetM(), ") ", 4, 1);
    } else {
      *oss << "(WL:  -.-----) (D: -.---) (M:  -.-) ";
    }
    print(oss, "(Q: ", n ? sign * n->GetQ(sign * draw_score) : fpu, ") ", 8, 5);
  };
  auto print_tail = [&](auto* oss, const auto* n) {
    const auto sign = n == node ? -1 : 1;
    std::optional<float> v;
    if (n && n->IsTerminal()) {
      v = n->GetQ(sign * draw_score);
    } else {
      NNCacheLock nneval = GetCachedNNEval(n);
      if (nneval) v = -nneval->q;
    }
    if (v) {
      print(oss, "(V: ", sign * *v, ") ", 7, 4);
    } else {
      *oss << "(V:  -.----) ";
    }

    if (n) {
      auto [lo, up] = n->GetBounds();
      if (sign == -1) {
        lo = -lo;
        up = -up;
        std::swap(lo, up);
      }
      *oss << (lo == up                                                ? "(T) "
               : lo == GameResult::DRAW && up == GameResult::WHITE_WON ? "(W) "
               : lo == GameResult::BLACK_WON && up == GameResult::DRAW ? "(L) "
                                                                       : "");
    }
  };

  std::vector<std::string> infos;
  const auto m_evaluator = network_->GetCapabilities().has_mlh()
                               ? MEvaluator(params_, node)
                               : MEvaluator();
  for (const auto& edge : edges) {
    float Q = edge.GetQ(fpu, draw_score);
    float M = m_evaluator.GetM(edge, Q);
    std::ostringstream oss;
    oss << std::left;
    // TODO: should this be displaying transformed index?
    print_head(&oss, edge.GetMove(is_black_to_move).as_string(),
               edge.GetMove().as_nn_index(0), edge.GetN(), edge.GetNInFlight(),
               edge.GetP());
    print_stats(&oss, edge.node());
    print(&oss, "(U: ", edge.GetU(U_coeff), ") ", 6, 5);
    print(&oss, "(S: ", Q + edge.GetU(U_coeff) + M, ") ", 8, 5);
    print_tail(&oss, edge.node());
    infos.emplace_back(oss.str());
  }

  // Include stats about the node in similar format to its children above.
  std::ostringstream oss;
  print_head(&oss, "node ", node->GetNumEdges(), node->GetN(),
             node->GetNInFlight(), node->GetVisitedPolicy());
  print_stats(&oss, node);
  print_tail(&oss, node);
  infos.emplace_back(oss.str());
  return infos;
}



*/