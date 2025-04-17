#include "search/lc3/positions.h"

#include "utils/hashcat.h"

namespace lczero {
namespace lc3 {

PositionChain PositionChain::FromStartpos(const Position& pos) {
  return PositionChain{NodeHash{pos.Hash()}, pos, nullptr};
}

PositionChain PositionChain::FromMove(const PositionChain* prev, Move move) {
  return PositionChain{
      NodeHash{NodeHash{HashCat(prev->hash.hash, move.raw_data())}},
      Position(prev->position, move), prev};
}

std::vector<PositionChain> GameStateToPositionChain(
    const GameState& game_state) {
  std::vector<PositionChain> positions;
  positions.reserve(game_state.moves.size() + 1);
  positions.push_back(PositionChain::FromStartpos(game_state.startpos));
  for (const auto& move : game_state.moves) {
    positions.push_back(PositionChain::FromMove(&positions.back(), move));
  }
  return positions;
}

}  // namespace lc3
}  // namespace lczero