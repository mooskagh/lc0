#pragma once
#include <cstdint>

namespace lczero {
namespace lc3 {

class NodeKey {
 public:
  explicit NodeKey(uint64_t hash) : hash_(hash) {}
  auto operator<=>(const NodeKey& other) const = default;

  uint64_t raw_hash() const { return hash_; }

  template <typename H>
  friend H AbslHashValue(H h, const NodeKey& key) {
    return H::combine(std::move(h), key.hash_);
  }

 private:
  uint64_t hash_;
};

}  // namespace lc3
}  // namespace lczero