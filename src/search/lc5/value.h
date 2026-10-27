#pragma once

#include <cstdint>

namespace lczero::lc5 {

// A value sample from the perspective of the side to move at its node.
struct SearchValue {
  float q = 0.0f;
  float d = 0.0f;
  float m = 0.0f;
  SearchValue Parent() const { return {-q, d, m + 1.0f}; }
};

struct ValueStats {
  uint64_t visits = 0;
  double q_sum = 0.0;
  double d_sum = 0.0;
  double m_sum = 0.0;

  float Q() const { return Mean(q_sum); }
  float D() const { return Mean(d_sum); }
  float M() const { return Mean(m_sum); }
  void Add(const SearchValue& value) {
    ++visits;
    q_sum += value.q;
    d_sum += value.d;
    m_sum += value.m;
  }

 private:
  float Mean(double sum) const {
    return visits ? static_cast<float>(sum / visits) : 0.0f;
  }
};

}  // namespace lczero::lc5
