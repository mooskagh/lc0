#pragma once

#include <cstdint>
#include <optional>
#include <span>

#include "chess/types.h"
#include "utils/exception.h"

namespace lczero {

struct NodeHash {
  uint64_t hash;
};

struct NodeCreate {};

struct NodeUpdate {
  bool HasVisits() const { NotImplemented(); }
  bool IsTerminal() const { NotImplemented(); }
  uint64_t IncrementN(int64_t n) { NotImplemented(); }
  size_t FetchNumMoves() const { NotImplemented(); }
  size_t FetchNumMovesWithVisits() const { NotImplemented(); }

  struct EdgeDataRequest {
    std::span<Move> moves = {};
    std::span<float> p = {};
    std::span<float> q = {};
    std::span<uint64_t> n = {};
  };

  void FetchEdgeData(EdgeDataRequest request) const { NotImplemented(); }
  void UpdateEdgeN(std::span<const uint64_t> n) const { NotImplemented(); }
};

// While this lock is held, no hashmap rehashing will occur.
struct UpdateLock {
  // Returns nullopt if the node is not found.
  std::optional<NodeUpdate> Fetch(NodeHash node) { NotImplemented(); }
};

struct CreateLock {
  static CreateLock FromUpdateLock(UpdateLock&& lock) { NotImplemented(); }

  std::optional<NodeCreate> Create(NodeHash node) { NotImplemented(); }
};

class Storage {
 public:
  UpdateLock GetUpdateLock() { NotImplemented(); }
};

}  // namespace lczero