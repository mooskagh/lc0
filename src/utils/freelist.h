#pragma once

#include <cstddef>
#include <cstdint>

namespace lczero {

template <typename T, size_t BlockSize>
class FreeList {
 public:
  FreeList() = default;
  ~FreeList() = default;
  FreeList(const FreeList&) = delete;
  FreeList& operator=(const FreeList&) = delete;
  FreeList(FreeList&&) = delete;
  FreeList& operator=(FreeList&&) = delete;

  template <typename... Args>
  [[nodiscard]] T* Allocate(Args&&... args) {
    Node* node = Pop();
    if (!node) {
      node = TryAllocateFromCurrentBlock();
      if (!node) node = AllocateNewBlock();
    }
    T* obj_ptr = reinterpret_cast<T*>(node);
    ::new (obj_ptr) T(std::forward<Args>(args)...);
    return obj_ptr;
  }

  void Release(T* p) noexcept(std::is_nothrow_destructible_v<T>) {
    if constexpr (!std::is_trivially_destructible_v<T>) p->~T();
    Node* node = reinterpret_cast<Node*>(p);
    Push(node);
  }

 private:
  union Node {
    static constexpr size_t StorageSize = std::max(sizeof(T), sizeof(Node*));
    static constexpr size_t StorageAlign = std::max(alignof(T), alignof(Node*));

    alignas(StorageAlign) std::byte storage[StorageSize];
    Node* next;
  };

  struct alignas(alignof(Node)) Block {
    Node nodes[BlockSize];
    std::atomic<size_t> next_node_index_{0};
  };

  Node* Pop() noexcept {
    Node* current_head = head_.load(std::memory_order_acquire);
    while (current_head) {
      Node* next_node = current_head->next;
      if (head_.compare_exchange_weak(current_head, next_node,
                                      std::memory_order_release,
                                      std::memory_order_acquire)) {
        return current_head;
      }
    }
    return nullptr;
  }

  void Push(Node* node) noexcept {
    Node* current_head = head_.load(std::memory_order_acquire);
    do {
      node->next = current_head;
    } while (!head_.compare_exchange_weak(current_head, node,
                                          std::memory_order_release,
                                          std::memory_order_acquire));
  }

  Node* TryAllocateFromCurrentBlock() noexcept {
    Block* current_block =
        current_allocation_block_.load(std::memory_order_acquire);
    if (current_block) {
      size_t index = current_block->block_local_next_node_index_.fetch_add(
          1, std::memory_order_acq_rel);
      if (index < BlockSize) return current_block->nodes[index];
    }
    return nullptr;
  }

  Node* AllocateNewBlock() {
    std::lock_guard<std::mutex> lock(block_mutex_);
    Block* current_block =
        current_allocation_block_.load(std::memory_order_relaxed);
    if (current_block) {
      size_t index = current_block->block_local_next_node_index_.fetch_add(
          1, std::memory_order_relaxed);
      if (index < BlockSize) return current_block->nodes[index];
    }
    auto new_block_ptr = std::make_unique<Block>();
    Block* new_block_raw = new_block_ptr.get();
    buffers_.push_back(std::move(new_block_ptr));
    new_block_raw->block_local_next_node_index_.store(
        1, std::memory_order_relaxed);
    current_allocation_block_.store(new_block_raw, std::memory_order_release);
    return &new_block_raw->nodes[0];
  }

  std::atomic<Node*> head_{nullptr};
  std::atomic<Block*> current_allocation_block_{nullptr};
  std::vector<std::unique_ptr<Block>> buffers_;  // absl REQUIRES(block_mutex_);
  std::mutex block_mutex_;
};

}  // namespace lczero