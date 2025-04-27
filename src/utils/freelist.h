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
  [[nodiscard]] T* New(Args&&... args) {
    Node* node = Pop();
    while (!node) {
      AllocateNewBlock();
      node = Pop();
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

  void AllocateNewBlock() {
    auto new_block_ptr = std::make_unique<Block>();
    Block* block_raw_ptr = new_block_ptr.get();

    {
      std::lock_guard<std::mutex> lock(block_mutex_);
      if (head_.load(std::memory_order_relaxed) != nullptr) return;
      buffers_.push_back(std::move(new_block_ptr));
    }

    Node* block_head = &block_raw_ptr->nodes[0];
    Node* block_tail = &block_raw_ptr->nodes[BlockSize - 1];

    for (size_t i = 0; i < BlockSize - 1; ++i) {
      block_raw_ptr->nodes[i].next = &block_raw_ptr->nodes[i + 1];
    }
    block_raw_ptr->nodes[BlockSize - 1].next = nullptr;

    Node* current_head = head_.load(std::memory_order_acquire);
    do {
      block_tail->next = current_head;
    } while (!head_.compare_exchange_weak(current_head, block_head,
                                          std::memory_order_release,
                                          std::memory_order_acquire));
  }

  union Node {
    static constexpr size_t StorageSize = std::max(sizeof(T), sizeof(Node*));
    static constexpr size_t StorageAlign = std::max(alignof(T), alignof(Node*));

    alignas(StorageAlign) std::byte storage[StorageSize];
    Node* next;
  };

  struct alignas(alignof(Node)) Block {
    Node nodes[BlockSize];
  };

  std::atomic<Node*> head_{nullptr};
  std::vector<std::unique_ptr<Block>> buffers_;  // absl REQUIRES(block_mutex_);
  std::mutex block_mutex_;
};

}  // namespace lczero