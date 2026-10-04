#include "search/lc5/engine.h"

#include <sstream>

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

void Lc5Engine::ClearGame() { graph_.Clear(); }

void Lc5Engine::SetBackend(Backend* backend) {
  if (backend_ == backend) return;
  AbortAndWait();
  backend_ = backend;
  ClearGame();
}

void Lc5Engine::SetPosition(const GameState& state) {
  AbortAndWait();
  if (!game_start_ || *game_start_ != state.startpos) ClearGame();
  game_start_ = state.startpos;
  PositionHistory history;
  history.Reset(state.startpos);
  for (Move move : state.moves) history.Append(move);
  root_history_ = std::move(history);
  Settings settings(*options_);
  root_key_ = MakeNodeKey(*root_history_, settings.history_key_length());
}

void Lc5Engine::NewGame() {
  AbortAndWait();
  run_.reset();
  ClearGame();
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

  Settings settings(*options_);
  root_key_ = MakeNodeKey(*root_history_, settings.history_key_length());
  const auto resolved = settings.Resolve(backend_->GetAttributes());
  const bool clock_was_missing = !clock_start_.has_value();
  const auto start = clock_start_.value_or(std::chrono::steady_clock::now());
  clock_start_.reset();

  GoParams effective_params = params;
  if (!effective_params.movetime) {
    effective_params.movetime =
        settings.GetTimeBudget(params, root_history_->Last().IsBlackToMove());
  }

  if (clock_was_missing) {
    std::vector<ThinkingInfo> infos{{
        .comment = "lc5 StartClock was not called; using StartSearch time"}};
    uci_responder_->OutputThinkingInfo(&infos);
  }

  std::vector<std::string> ignored;
  if (params.depth) ignored.push_back("depth");
  if (params.mate) ignored.push_back("mate");
  if (!params.searchmoves.empty()) ignored.push_back("searchmoves");
  if (!ignored.empty()) {
    std::ostringstream warning;
    warning << "lc5 ignored unsupported go fields:";
    for (const auto& field : ignored) warning << ' ' << field;
    std::vector<ThinkingInfo> infos{{.comment = warning.str()}};
    uci_responder_->OutputThinkingInfo(&infos);
  }

  VisitOrigin root{.key = root_key_,
                   .history = *root_history_,
                   .backup_prefix = {}};
  run_ = std::make_unique<SearchRun>(&graph_, nullptr, backend_,
                                     uci_responder_, resolved, std::move(root),
                                     effective_params, start);
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
