#pragma once

#include <memory>

namespace lczero {

template <typename T, typename Allocator = std::allocator<T>>
class Tree {
 public:
  class node_handle;

  template <typename... Args>
  Tree(Args&&... args) {
    Node* new_node = allocator_.allocate(1);
    ::new (new_node) Node(nullptr, this, std::forward<Args>(args)...);
    root_ = node_handle(new_node);
  }
  ~Tree() { root_.reset(); }
  node_handle root() { return root_; }

 private:
  struct Node {
    T data;
    Node* parent;
    // TODO having pointer to tree in each node is not great, but let's see how
    // it goes.
    Tree* tree;
    std::atomic<size_t> ref_count_;

    Node() = delete;
    template <typename... Args>
    Node(Node* parent, Tree* tree, Args&&... args)
        : data(std::forward<Args>(args)...),
          parent(parent),
          tree(tree),
          ref_count_(0) {}
  };

  Node* allocate_node() {
    return std::allocator_traits<NodeAllocator>::allocate(allocator_, 1);
  }
  void deallocate_node(Node* p) {
    std::allocator_traits<NodeAllocator>::deallocate(allocator_, p, 1);
  }

  using NodeAllocator =
      typename std::allocator_traits<Allocator>::template rebind_alloc<Node>;
  NodeAllocator allocator_;
  node_handle root_;
  friend class node_handle;
};

template <typename T, typename Allocator>
class Tree<T, Allocator>::node_handle {
 public:
  node_handle() : node_(nullptr) {}
  node_handle(const node_handle& other) : node_handle(other.node_) {}
  node_handle& operator=(const node_handle& other) {
    if (this != &other) {
      reset();
      node_ = other.node_;
      if (node_) node_->ref_count_.fetch_add(1, std::memory_order_relaxed);
    }
    return *this;
  }
  node_handle(node_handle&& other) : node_(other.node_) {
    other.node_ = nullptr;
  }
  node_handle& operator=(node_handle&& other) {
    if (this != &other) {
      reset();
      node_ = other.node_;
      other.node_ = nullptr;
    }
    return *this;
  }

  operator bool() const { return node_ != nullptr; }
  bool has_parent() const { return node_->parent != nullptr; }
  node_handle parent() { return node_handle(node_->parent); }
  T* operator->() const { return &node_->data; }
  T& operator*() const { return node_->data; }

  template <typename... Args>
  node_handle make_child(Args&&... args) {
    node_->ref_count_.fetch_add(1, std::memory_order_relaxed);
    Node* new_node = node_->tree->allocator_.allocate(1);
    ::new (new_node) Node(node_, node_->tree, std::forward<Args>(args)...);
    return node_handle(new_node);
  }

  ~node_handle() { reset(); }
  void reset() {
    if (!node_) return;
    Node* node = node_;
    while (node &&
           node->ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      Tree* tree = node->tree;
      Node* parent = node->parent;
      node->~Node();
      tree->allocator_.deallocate(node, 1);
      node = parent;
    }
    node_ = nullptr;
  }

 private:
  node_handle(Node* node) : node_(node) {
    if (node_) node_->ref_count_.fetch_add(1, std::memory_order_relaxed);
  }
  Node* node_;
  friend class Tree;
};

}  // namespace lczero