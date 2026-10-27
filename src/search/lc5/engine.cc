#include "search/lc5/engine.h"

#include "chess/callbacks.h"
#include "chess/uciloop.h"
#include "search/lc5/settings.h"
#include "search/register.h"
#include "utils/exception.h"

namespace lczero::lc5 {

Lc5Engine::~Lc5Engine() { AbortAndWait(); }

void Lc5Engine::AbortAndWait() {
  if (!run_) return;
  run_->Abort();
  run_->Wait();
}

void Lc5Engine::SetBackend(Backend* backend) {
  if (backend_ == backend) return;
  AbortAndWait();
  backend_ = backend;
  graph_.Clear();
}

void Lc5Engine::SetPosition(const GameState& state) {
  AbortAndWait();
  // The graph survives position changes within the same game.
  if (game_start_ != state.startpos) graph_.Clear();
  game_start_ = state.startpos;
  root_history_.emplace();
  root_history_->Reset(state.startpos);
  for (Move move : state.moves) root_history_->Append(move);
}

void Lc5Engine::NewGame() {
  AbortAndWait();
  run_.reset();
  graph_.Clear();
  game_start_.reset();
  root_history_.reset();
  clock_start_.reset();
}

void Lc5Engine::StartClock() {
  clock_start_ = std::chrono::steady_clock::now();
}

void Lc5Engine::StartSearch(const GoParams& params) {
  AbortAndWait();
  run_.reset();
  if (!backend_) throw Exception("Lc5 requires a backend");
  if (!root_history_) throw Exception("Lc5 position was not set");

  const Settings settings(*options_);
  VisitOrigin root{
      .key = MakeNodeKey(*root_history_, settings.history_key_length()),
      .history = *root_history_,
      .backup_prefix = {}};
  const auto start = clock_start_.value_or(std::chrono::steady_clock::now());
  clock_start_.reset();
  run_ = std::make_unique<SearchRun>(
      &graph_, /*store=*/nullptr, backend_, uci_responder_,
      settings.Resolve(backend_->GetAttributes()), settings.time_management(),
      std::move(root), params, start);
  run_->Start();
}

void Lc5Engine::StopSearch() {
  if (run_) run_->Stop();
}

void Lc5Engine::AbortSearch() {
  if (run_) run_->Abort();
}

void Lc5Engine::WaitSearch() {
  if (run_) run_->Wait();
}

namespace {
class Lc5Factory final : public SearchFactory {
 public:
  std::string_view GetName() const override { return "lc5"; }
  void PopulateParams(OptionsParser* options) const override {
    Settings::Populate(options);
  }
  std::unique_ptr<SearchBase> CreateSearch(
      UciResponder* responder, const OptionsDict* options) const override {
    return std::make_unique<Lc5Engine>(responder, options);
  }
};

REGISTER_SEARCH(Lc5Factory)
}  // namespace

}  // namespace lczero::lc5
