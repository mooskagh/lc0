#pragma once

#include "search/lc3/session.h"
#include "search/search.h"

namespace lczero {
namespace lc3 {

class Lc3Engine : public SearchBase {
 public:
  Lc3Engine(UciResponder* responder, const OptionsDict* options)
      : SearchBase(responder), options_(options) {}

  void SetPosition(const GameState&) override;
  void StartSearch(const GoParams&) override;
  void StartClock() override { TODO("Start clock"); }
  void StopSearch() override;
  void AbortSearch() override;
  void WaitSearch() override;

 private:
  void EnsureSearchStopped();

  std::unique_ptr<SearchSession> search_;
  NodeRepository node_repository_;
  GameState game_state_;
  ThreadPool thread_pool_{ThreadPoolOptions{.grow_automatically = true}};
  const OptionsDict* options_;
};

}  // namespace lc3
}  // namespace lczero