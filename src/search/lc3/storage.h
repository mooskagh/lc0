#pragma once

#include <cstdint>
#include <optional>
#include <span>

#include "chess/types.h"

namespace lczero {

struct NodeHash {
  uint64_t hash;
};

struct NodeCreate {};

struct NodeUpdate {
  bool HasVisits() const;
  bool IsTerminal() const;
  uint64_t IncrementN(int64_t n);
  size_t FetchNumMoves() const;
  size_t FetchNumMovesWithVisits() const;

  struct EdgeDataRequest {
    std::span<Move> moves = {};
    std::span<float> p = {};
    std::span<float> q = {};
    std::span<uint64_t> n = {};
  };

  void FetchEdgeData(EdgeDataRequest request) const;
  void UpdateEdgeN(std::span<const uint64_t> n) const;
};

// While this lock is held, no hashmap rehashing will occur.
struct UpdateLock {
  // Returns nullopt if the node is not found.
  std::optional<NodeUpdate> Fetch(NodeHash node);
};

struct CreateLock {
  static CreateLock FromUpdateLock(UpdateLock&& lock);

  std::optional<NodeCreate> Create(NodeHash node);
};

class Storage {
 public:
  UpdateLock GetUpdateLock();
};

}  // namespace lczero