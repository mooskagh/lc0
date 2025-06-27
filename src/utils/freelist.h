#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace lczero {

template <typename T, size_t BlockSize = 4096>
class FreeListAllocator {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using propagate_on_container_move_assignment = std::true_type;
  using is_always_equal = std::false_type;

  class Deleter;
  using Ptr = std::unique_ptr<T, Deleter>;

  template <typename U>
  struct rebind {
    using other = FreeListAllocator<U, BlockSize>;
  };

  FreeListAllocator() noexcept = default;
  template <typename U>
  FreeListAllocator(const FreeListAllocator<U, BlockSize>&) noexcept {}
  ~FreeListAllocator() = default;
  [[nodiscard]] T* allocate(std::size_t n) {
    if (n != 1) [[unlikely]] {
      throw Exception(
          "FreeListAllocator only supports single object allocation");
    }
    Node* node = Pop();
    if (!node) {
      node = TryAllocateFromCurrentBlock();
      if (!node) node = AllocateNewBlock();
    }
    return reinterpret_cast<T*>(node);
  }

  void deallocate(T* p, std::size_t n) noexcept {
    assert(n == 1 &&
           "FreeListAllocator only supports single object deallocation");
    (void)n;

    Node* node = reinterpret_cast<Node*>(p);
    Push(node);
  }

  template <typename U, typename... Args>
  void construct(U* p, Args&&... args) {
    ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
  }

  template <typename U>
  void destroy(U* p) noexcept {
    p->~U();
  }

  [[nodiscard]] size_type max_size() const noexcept {
    return std::numeric_limits<size_type>::max() / sizeof(T);
  }

  template <typename U>
  bool operator==(const FreeListAllocator<U, BlockSize>& other) const noexcept {
    return this == &other;
  }

  template <typename U>
  bool operator!=(const FreeListAllocator<U, BlockSize>& other) const noexcept {
    return !(*this == other);
  }

  template <typename... Args>
  [[nodiscard]] Ptr Make(Args&&... args) {
    T* raw_ptr = this->allocate(1);
    this->construct(raw_ptr, std::forward<Args>(args)...);
    return Ptr(raw_ptr, Deleter(this));
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
      size_t index = current_block->next_node_index_.fetch_add(
          1, std::memory_order_acq_rel);
      if (index < BlockSize) return &current_block->nodes[index];
    }
    return nullptr;
  }

  Node* AllocateNewBlock() {
    std::lock_guard<std::mutex> lock(block_mutex_);
    Block* current_block =
        current_allocation_block_.load(std::memory_order_relaxed);
    if (current_block) {
      size_t index = current_block->next_node_index_.fetch_add(
          1, std::memory_order_relaxed);
      if (index < BlockSize) return &current_block->nodes[index];
    }
    auto new_block_ptr = std::make_unique<Block>();
    Block* new_block_raw = new_block_ptr.get();
    buffers_.push_back(std::move(new_block_ptr));
    new_block_raw->next_node_index_.store(1, std::memory_order_relaxed);
    current_allocation_block_.store(new_block_raw, std::memory_order_release);
    return &new_block_raw->nodes[0];
  }

  std::atomic<Node*> head_{nullptr};
  std::atomic<Block*> current_allocation_block_{nullptr};
  std::vector<std::unique_ptr<Block>> buffers_;
  std::mutex block_mutex_;
};

template <typename T, size_t BlockSize>
class FreeListAllocator<T, BlockSize>::Deleter {
 public:
  explicit Deleter(FreeListAllocator* allocator) noexcept
      : allocator_ptr_(allocator) {}

  void operator()(T* p) const noexcept {
    if (p) {
      p->~T();
      allocator_ptr_->deallocate(p, 1);
    }
  }

  bool operator==(const Deleter& other) const noexcept {
    return allocator_ptr_ == other.allocator_ptr_;
  }

  bool operator!=(const Deleter& other) const noexcept {
    return !(*this == other);
  }

 private:
  FreeListAllocator* allocator_ptr_;
};

}  // namespace lczero