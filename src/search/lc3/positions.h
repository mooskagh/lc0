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

  int GetRepetitionCount() const { NotImplemented(); }

 private:
  PositionChain(NodeHash hash, Position position, const PositionChain* prev)
      : hash(hash), position(position), prev(prev) {}
};

size_t UnpackPositionsBackwards(const PositionChain& pos_chain,
                                std::span<Position> positions);

std::vector<PositionChain> GameStateToPositionChain(
    const GameState& game_state);

}  // namespace lc3
}  // namespace lczero