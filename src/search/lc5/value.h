#pragma once

#include <cstdint>

namespace lczero::lc5 {

struct ValueStats {
  uint64_t visits = 0;
  double q_sum = 0.0;
  double d_sum = 0.0;
  double m_sum = 0.0;

  float Q() const { return visits ? static_cast<float>(q_sum / visits) : 0.0f; }
  float D() const { return visits ? static_cast<float>(d_sum / visits) : 0.0f; }
  float M() const { return visits ? static_cast<float>(m_sum / visits) : 0.0f; }
  void Add(float q, float d, float m) {
    ++visits;
    q_sum += q;
    d_sum += d;
    m_sum += m;
  }
};

struct SearchValue {
  float q = 0.0f;
  float d = 0.0f;
  float m = 0.0f;
  SearchValue Parent() const { return {-q, d, m + 1.0f}; }
};

}  // namespace lczero::lc5
