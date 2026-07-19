#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "chess/types.h"
#include "search/lc5/key.h"

namespace lczero::lc5 {

enum class TerminalKind : uint8_t {
  kNonTerminal,
  kCheckmate,
  kStalemate,
  kRule50,
  kRepetition,
  kInsufficientMaterial,
};

struct ExpansionPayload {
  TerminalKind terminal = TerminalKind::kNonTerminal;
  float leaf_q = 0.0f;
  float leaf_d = 0.0f;
  float leaf_m = 0.0f;
  std::vector<Move> moves;
  std::vector<float> priors;
};

struct StoredExpansion {
  NodeKey key;
  ExpansionPayload payload;
};

class NodeStore {
 public:
  virtual ~NodeStore() = default;
  virtual std::vector<std::optional<ExpansionPayload>> LoadBatch(
      std::span<const NodeKey> keys) = 0;
  virtual void StoreBatch(std::span<const StoredExpansion> entries) = 0;
  virtual void Clear() = 0;
};

}  // namespace lczero::lc5
