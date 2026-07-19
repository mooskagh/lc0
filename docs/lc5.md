# Lc5 search backend

Lc5 is an experimental, game-scoped DAG search selected with `lc0 lc5`.  It
uses key/generation paths, exact per-edge in-flight reservations, a bounded
epoch-tagged VisitPool, deduplicated materialization tickets, and a
value-semantic NodeStore.  Mutable search statistics never leave the hot
graph; immutable expansion payloads may be rehydrated after a forced erase.

The initial NodeStore is memory-backed. Store loads and neural evaluations run
away from visit workers, and the ready-evaluation queue is intentionally
allowed to contain more than one backend batch. A non-owner visit reaching a
pending node suspends and, after a nonterminal expansion, resumes selection
below that node. Stop and abort cancel every surviving reservation and retain
epoch checks on late completions.

Supported v1 limits are `nodes`, `movetime`, `infinite`, and ponder as
infinite. Fields belonging to the classic time manager are reported as
ignored. The hot graph and payload store survive same-game `position` changes;
`ucinewgame`, an incompatible starting position, and backend replacement clear
them.

## Verification performed during implementation

The following checks were run on the implementation host:

```text
meson setup build/debug --reconfigure -Dlc5=true -Dgtest=true
ninja -C build/debug
meson test -C build/debug --print-errorlogs

meson configure build/debug -Dlc3=false -Dlc5=true
ninja -C build/debug

meson configure build/debug -Dlc3=true -Dlc5=false
ninja -C build/debug
```

All 13 configured debug tests passed. A trivial-backend UCI smoke run with two
visit workers, one evaluator, batch target four, and 16 active visits completed
`go nodes 20` with exactly 20 completed visits. Interactive checks also covered
repeated go commands, same-game re-rooting, infinite plus stop, and immediate
stop.

Hardware-dependent GPU, sanitizer, scaling, and fastchess results are not
claimed here. Use the benchmark harness and the workloads in
`Lc5_IMPLEMENTATION_PLAN.md` before making performance or strength claims.
