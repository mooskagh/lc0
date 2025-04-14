
#include "chess/gamestate.h"


#include <queue>
#include <utility>  

namespace lczero {

struct GameStateHash {};

GameStateHash GameStateToHash(const GameState& state);

//

struct NodeStorage{};

struct HodeHandle{};


struct WorkTree {
    size_t head = 0;
    size_t tail = 0;
};

void SearchLoop(const GameState& state, size_t max_visits) {
    NodeStorage node_storage;


  GameStateHash root_hash = GameStateToHash(state);

  std::queue<std::pair<GameStateHash, size_t>> queue;
  queue.push({root_hash, max_visits});





};

}  // namespace lczero