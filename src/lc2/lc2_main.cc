#include "lc2/chess/position-key.h"
#include "lc2/mcts/batches.h"
#include "lc2/mcts/params.h"
#include "lc2/storage/storage.h"

namespace lc2 {
void test() {
  NodeStorage storage;
  const auto& board = lczero::ChessBoard::kStartposBoard;
  const auto key = PositionKey(board.Hash());
  Params params;
  Batch batch(params);
  batch.EnqueuePosition(board, key, 100);
  batch.Gather(&storage);
}
}  // namespace lc2

int main() { lc2::test(); }