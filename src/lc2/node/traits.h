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

#pragma once

#include <cstdint>

namespace lczero {
namespace lc2 {

struct WdlNodeTraits {
  //////////////////////////////////////
  // Values stored in nodes or edges.
  //////////////////////////////////////
  using N = uint32_t;

  // Q, as stored in node (WDL, or whatever we want to propagate back to root).
  struct WDL {
    double q;
    double d;
  };

  // P, a prior stored in edge and returned from NN.
  using P = double;

  // Q as used in Q+U formula. Note that it also is stored in edge.
  using Q = double;

  ///////////////////////////////
  // Values not stored in nodes.
  ///////////////////////////////

  // Q, U and Q+U, as used in Q+U formula.
  using U = double;
  using QPlusU = double;

  // Type used for CPuct constant and for policy visits factor.
  using QFactor = double;

  // First play urgency.
  using FPU = double;

  static QFactor ComputeQFactor(QFactor cpuct, N total_n);
  static FPU ComputeFPU();
  static Q ComputeQ(QFactor, Q);
  static U ComputeU(P, FPU, N n_edge, N total_n);
};

}  // namespace lc2
}  // namespace lczero