#pragma once

#include <cstddef>
#include <cstdint>

namespace lczero {

template <typename T, size_t BlockSize>
class FreeList {
 public:
  FreeList() { AddNewBlock(); }
  FreeList(const FreeList&) = delete;
  FreeList& operator=(const FreeList&) = delete;
  FreeList(FreeList&&) = delete;
  FreeList& operator=(FreeList&&) = delete;

  template <typename... Args>
  [[nodiscard]] T* New(Args&&... args) {
    if (!head_) [[unlikely]] {
      AddNewBlock();
    }
    Node* node = head_;
    head_ = node->next;
    return ::new (static_cast<void*>(node->storage))
        T(std::forward<Args>(args)...);
  }

  void Release(T* p) noexcept(std::is_nothrow_destructible_v<T>) {
    p->~T();
    Node* node = reinterpret_cast<Node*>(p);
    node->next = head_;
    head_ = node;
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
  };

  std::vector<std::unique_ptr<Block>> buffers_;
  Node* head_ = nullptr;

  void AddNewBlock() {
    buffers_.push_back(std::make_unique<Block>());
    for (size_t i = 0; i < BlockSize; ++i) {
      Node* current = std::addressof(new_block->nodes[BlockSize - 1 - i]);
      current->next = head_;
      head_ = current;
    }
  }
};

}  // namespace lczero