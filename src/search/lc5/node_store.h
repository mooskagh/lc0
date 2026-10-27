#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "chess/types.h"
#include "search/lc5/key.h"
#include "search/lc5/value.h"

namespace lczero::lc5 {

enum class TerminalKind : uint8_t {
  kNonTerminal,
  kCheckmate,
  kStalemate,
  kRule50,
  kRepetition,
  kInsufficientMaterial,
};

// Value of a terminal position for its side to move.
inline SearchValue TerminalValue(TerminalKind terminal) {
  return terminal == TerminalKind::kCheckmate ? SearchValue{-1.0f, 0.0f, 0.0f}
                                              : SearchValue{0.0f, 1.0f, 0.0f};
}

// Immutable result of expanding a node: either a terminal kind, or legal moves
// with priors and the network's leaf value.
struct ExpansionPayload {
  TerminalKind terminal = TerminalKind::kNonTerminal;
  SearchValue leaf{};
  std::vector<Move> moves;
  std::vector<float> priors;
};

struct StoredExpansion {
  NodeKey key;
  ExpansionPayload payload;
};

// Optional batched backing store for expansion payloads.
class NodeStore {
 public:
  virtual ~NodeStore() = default;
  virtual std::vector<std::optional<ExpansionPayload>> LoadBatch(
      std::span<const NodeKey> keys) = 0;
  virtual void StoreBatch(std::span<const StoredExpansion> entries) = 0;
  virtual void Clear() = 0;
};

}  // namespace lczero::lc5
