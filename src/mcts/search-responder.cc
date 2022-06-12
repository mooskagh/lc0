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
#include "proto/jsondata.pb.h"

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

void Search::Responder::OutputBestMove(Move bestmove, Move pondermove) const {
  BestMoveInfo info(bestmove, pondermove);
  responder_->OutputBestMove(&info);
}

namespace {
ThinkingInfo CommonToThinking(const pblczero::CommonInfo& info) {
  ThinkingInfo result;
  result.depth = info.depth();
  result.seldepth = info.seldepth();
  result.time = info.time();
  result.hashfull = info.hashfull();
  result.nodes = info.nodes();
  result.nps = info.nps();
  result.tb_hits = info.tbhits();
  return result;
}
}  // namespace

void Search::Responder::SendUciInfo() const REQUIRES(search_.nodes_mutex_)
    REQUIRES(search_.counters_mutex_) {
  const auto& params = search_.params_;
  const auto& root_node = search_.root_node_;
  const auto max_pv = params.GetMultiPv();
  const auto edges = search_.GetBestChildrenNoTemperature(root_node, max_pv, 0);
  const auto score_type = params.GetScoreType();
  const auto per_pv_counters = params.GetPerPvCounters();
  const auto draw_score = search_.GetDrawScore(false);

  std::vector<ThinkingInfo> uci_infos;

  auto common_info = GetCommonInfo();

  int multipv = 0;
  const auto default_q = -root_node->GetQ(-draw_score);
  const auto default_wl = -root_node->GetWL();
  const auto default_d = root_node->GetD();
  for (const auto& edge : edges) {
    ++multipv;
    uci_infos.emplace_back(CommonToThinking(common_info));
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
    if (per_pv_counters) {
      uci_info.nodes = edge.GetN();
    } else {
      uci_info.nodes = search_.total_playouts_ + search_.initial_visits_;
    }
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

void Search::Responder::OutputComment(std::string_view text) const {
  std::vector<ThinkingInfo> infos;
  infos.emplace_back().comment = text;
  responder_->OutputThinkingInfo(&infos);
}

std::vector<std::string> Search::Responder::GetVerboseStats(Node* node) const
    REQUIRES(search_.counters_mutex_) {
  assert(node == search_.root_node_ || node->GetParent() == search_.root_node_);
  const bool is_root = (node == search_.root_node_);
  const bool is_odd_depth = !is_root;
  const bool is_black_to_move =
      (search_.played_history_.IsBlackToMove() == is_root);
  const float draw_score = search_.GetDrawScore(is_odd_depth);
  const float fpu = GetFpu(search_.params_, node, is_root, draw_score);
  const float cpuct = ComputeCpuct(search_.params_, node->GetN(), is_root);
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
      NNCacheLock nneval = search_.GetCachedNNEval(n);
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
  const auto m_evaluator = search_.network_->GetCapabilities().has_mlh()
                               ? MEvaluator(search_.params_, node)
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

pblczero::CommonInfo Search::Responder::GetCommonInfo() const {
  pblczero::CommonInfo info;
  info.set_depth(Depth());
  info.set_seldepth(search_.max_depth_);
  info.set_time(search_.GetTimeSinceStart());
  if (search_.params_.GetDisplayCacheUsage()) {
    info.set_hashfull(search_.cache_->GetSize() * 1000LL /
                      std::max(search_.cache_->GetCapacity(), 1));
  }
  info.set_nodes(search_.root_node_->GetN());
  if (search_.nps_start_time_) {
    const auto time_since_first_batch_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - *search_.nps_start_time_)
            .count();
    if (time_since_first_batch_ms > 0) {
      info.set_nps(search_.total_playouts_ * 1000 / time_since_first_batch_ms);
    }
  }
  info.set_tbhits(search_.tb_hits_.load(std::memory_order_acquire));
  return info;
}

namespace {
pblczero::Evaluation GetEvaluationInfo(const EdgeAndNode& edge_and_node,
                                       float wl, float d, float draw_score,
                                       const std::string& score_type) {
  pblczero::Evaluation eval;
  const auto q = wl + draw_score * d;
  if (edge_and_node.IsTerminal() && wl != 0.0f) {
    eval.set_mate(std::copysign(std::round(edge_and_node.GetM(0.0f)) / 2 +
                                    (edge_and_node.IsTbTerminal() ? 101 : 1),
                                wl));
  } else if (score_type == "centipawn_with_drawscore") {
    eval.set_cp(90 * tan(1.5637541897 * q));
  } else if (score_type == "centipawn") {
    eval.set_cp(90 * tan(1.5637541897 * wl));
  } else if (score_type == "centipawn_2019") {
    eval.set_cp(295 * wl / (1 - 0.976953126 * std::pow(wl, 14)));
  } else if (score_type == "centipawn_2018") {
    eval.set_cp(290.680623072 * tan(1.548090806 * wl));
  } else if (score_type == "win_percentage") {
    eval.set_cp(wl * 5000 + 5000);
  } else if (score_type == "Q") {
    eval.set_cp(q * 10000);
  } else if (score_type == "W-L") {
    eval.set_cp(wl * 10000);
  }
  float w = std::max(0.0f, 0.5f * (1.0f + wl - d));
  float l = std::max(0.0f, 0.5f * (1.0f - wl - d));
  d = 1.0f - w - l;
  if (d < 0) {
    w = std::min(1.0f, std::max(0.0f, w + 0.5f * d));
    l = 1.0f - w;
    d = 0.0f;
  }
  pblczero::WDL* wdl = eval.mutable_wdl();
  wdl->set_w(w);
  wdl->set_d(d);
  wdl->set_l(l);
  eval.set_expected_score(wl * 0.5f + 0.5f);
  return eval;
}
}  //  namespace

pblczero::MoveInfo Search::Responder::GetMoveInfo(
    const EdgeAndNode& edge_and_node,
    const pblczero::NodeInfo& parent_node_info, bool is_odd_depth) const {
  pblczero::MoveInfo info;

  const auto draw_score = search_.GetDrawScore(is_odd_depth);
  if (edge_and_node.HasNode()) {
    *info.mutable_node_info() = GetNodeInfo(edge_and_node.node());
  }
  const auto& node_info = info.node_info();
  const auto wl = node_info.has_wl() ? node_info.wl() : -parent_node_info.wl();
  const auto d = node_info.has_d() ? node_info.d() : parent_node_info.d();
  *info.mutable_eval() = GetEvaluationInfo(edge_and_node, wl, d, draw_score,
                                           search_.params_.GetScoreType());
  info.set_nodes(edge_and_node.GetN());

  bool flip = search_.played_history_.IsBlackToMove() ^ is_odd_depth;
  int depth = 0;
  for (auto iter = edge_and_node; iter;
       iter = search_.GetBestChildNoTemperature(iter.node(), depth),
            flip = !flip) {
    info.add_pv(iter.GetMove(flip).as_string());
    if (!iter.node()) break;  // Last edge was dangling, cannot continue.
    depth += 1;
  }
  return info;
}
}  // namespace lczero