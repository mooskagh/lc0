
#pragma once

#include <array>
#include <cstdint>
#include <span>

class BlockStorage {
 public:
  using Block = std::array<uint8_t, 64>;
  using Hash = uint64_t;

  // Return status core for the FetchOrCreate method.
  enum class Status {
    kFetched,  // The node existed and was fetched.
    kCreated,  // The node didn't exist and was created empty.
    // kCollision,  // The node existed, but empty, being created in other
    // place.
  };

  void Discard(std::span<const Hash> hashes);
  void Fetch(std::span<const Hash> hashes, std::span<Block> dst_blocks,
             std::span<Status> statii);
  void Store(std::span<const Hash> hashes, std::span<const Block> blocks);
  template <typename F>
  void Update(std::span<const Hash> hashes, std::span<Block*> blocks,
              std::span<Status> statii, F&&);
};

class BufferStorage {
 public:
  class Ptr {
    operator void*() const;
    uint32_t ptr_;
    ~Ptr();  // Deallocates or submits to GC.
  };

  static std::pair<Ptr, void*> Allocate(uint32_t size);
};

class NodeStorage {
 public:
};

// --------------------------

#include <cstddef>  // For size_t
#include <cstdint>  // For specific integer types

// Consider placing these types and the constant in a dedicated namespace or
// header
using NodeIndex = uint32_t;
using PayloadIndex =
    uint64_t;  // Assuming payload might exceed 4 billion elements total

// Represents an invalid or non-existent node index (e.g., parent of the root)
constexpr NodeIndex kInvalidNodeIndex = static_cast<NodeIndex>(-1);

struct PayloadRange {
  PayloadIndex start;  // Inclusive start index
  PayloadIndex end;    // Exclusive end index (start + size)

  // Helper function
  size_t size() const { return static_cast<size_t>(end - start); }
};

struct CreateNodeResult {
  NodeIndex node_idx;
  PayloadRange payload_range;
};

class IActiveNodeIndexManager {
 public:
  virtual ~IActiveNodeIndexManager() = default;

  // Creates a new node associated with the given parent.
  // Returns the index of the newly created node and its allocated payload
  // range. `parent_idx` should be `kInvalidNodeIndex` for the root node.
  // `payload_size` is the number of elements requested for this node's payload.
  virtual CreateNodeResult CreateNode(NodeIndex parent_idx,
                                      size_t payload_size) = 0;

  // Deletes the specified node.
  // IMPORTANT PRECONDITION: This function must only be called with the most
  // recently created `node_idx` that has not yet been deleted (LIFO/stack
  // discipline).
  virtual void DeleteNode(NodeIndex node_idx) = 0;

  // Retrieves the payload range associated with a given node index.
  virtual PayloadRange GetPayloadRange(NodeIndex node_idx) const = 0;

  // Retrieves the parent index of a given node index.
  // Returns `kInvalidNodeIndex` if the node is the root.
  virtual NodeIndex GetParent(NodeIndex node_idx) const = 0;

  // Finds the node index that owns the given payload index.
  // Behavior is undefined if `payload_idx` does not fall within any allocated
  // range.
  virtual NodeIndex PayloadIdxToNodeIdx(PayloadIndex payload_idx) const = 0;

  // Optional: Query current state
  virtual NodeIndex GetCurrentNodeCount() const = 0;
  virtual PayloadIndex GetCurrentPayloadOffset()
      const = 0;  // Represents the next available payload index
};