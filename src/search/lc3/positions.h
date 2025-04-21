#pragma once

#include <vector>

#include "chess/gamestate.h"
#include "chess/position.h"
#include "search/lc3/storage.h"

namespace lczero {
namespace lc3 {

struct PositionChain {
  static PositionChain FromStartpos(const Position& pos);
  static PositionChain FromMove(const PositionChain* prev, Move move);

  NodeHash hash{};
  Position position{};
  const PositionChain* prev = nullptr;

  int GetRepetitionCount() const;

 private:
  PositionChain(NodeHash hash, Position position, const PositionChain* prev)
      : hash(hash), position(position), prev(prev) {}
};

// TODO Move this function somewhere else.
[[nodiscard]] inline size_t UnpackPositionsBackwards(
    const PositionChain& pos_chain, std::span<Position> positions) {
  const PositionChain* cur_node = &pos_chain;
  auto iter = positions.rbegin();
  const auto end = positions.rend();

  while (iter != end && cur_node != nullptr) {
    *iter = cur_node->position;
    cur_node = cur_node->prev;
    ++iter;
  }

  return std::distance(positions.rbegin(), iter);
}

// TODO Move this too, maybe
inline int PositionChain::GetRepetitionCount() const {
  if (position.GetRule50Ply() < 4) return 0;
  auto skip = [](const PositionChain* node, size_t count) {
    for (; count > 0 && node; --count) node = node->prev;
    return node;
  };
  int num_reps = 0;
  for (const PositionChain* node = skip(this, 4); node; node = skip(node, 2)) {
    if (node->position.GetBoard() == position.GetBoard()) ++num_reps;
    if (node->position.GetRule50Ply() < 2) break;
  }
  return num_reps;
};

std::vector<PositionChain> GameStateToPositionChain(
    const GameState& game_state);

}  // namespace lc3
}  // namespace lczero