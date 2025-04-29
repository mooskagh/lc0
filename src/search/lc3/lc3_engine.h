#pragma once

#include "search/lc3/session.h"
#include "search/search.h"

namespace lczero {
namespace lc3 {

class Lc3Engine : public SearchBase {
 public:
  using SearchBase::SearchBase;

  void SetPosition(const GameState&) override;
  void StartSearch(const GoParams&) override;
  void StartClock() override { TODO("Start clock"); }
  void StopSearch() override { NotImplemented(); }
  void AbortSearch() override;
  void WaitSearch() override;

 private:
  void EnsureSearchStopped();

  std::unique_ptr<SearchSession> search_;
  NodeStorage storage_;
  GameState game_state_;
};

}  // namespace lc3
}  // namespace lczero