# Lc5 search backend

For a source-oriented review, see [Reading the lc5 search](lc5-code-walkthrough.md).

Lc5 is an experimental, game-scoped DAG search selected with `lc0 lc5`.  It
uses key/generation paths, exact per-edge in-flight reservations, a bounded
epoch-tagged VisitPool, deduplicated materialization tickets, and a
value-semantic optional NodeStore. Mutable search statistics never leave the hot
graph; immutable expansion payloads may be rehydrated after a forced erase.

The default engine retains its GameGraph without eviction across same-game
searches and position changes. `GameGraph::Erase` has no production callers;
previously its memory-backed payload store was cleared with the graph, so no
stored entry could serve a graph miss. The engine therefore passes no store to
SearchRun. Explicit non-null stores still support batched loads and immutable
payload persistence. Without a store, materialization uses the same queues,
tickets, terminal detection, and neural evaluation path, but does not build
store batches or record store metrics.

## Worker-owned batched scheduler

Each visit has one worker owner for selection, suspension/resumption, backup,
and cancellation. Workers own their runnable/active visits and pending load,
evaluation, and persistence requests; the graph and materialization tickets
remain shared and synchronized. Cross-worker continuations are grouped into
mailboxes. A ticket's owner publishes its expansion and wakes waiters; a
non-owner visit resumes selection below a nonterminal expansion rather than
backing up the owner's leaf value. Epoch and generation checks reject stale
continuations and graph updates.

Workers submit whole I/O jobs and consume whole-job completions through their
mailboxes. Evaluators only add inputs and run backend computations; store
workers only call `LoadBatch`/`StoreBatch`. Publication, store-miss preparation,
and visit transitions run on the owning visit worker, not on I/O threads.
`EvalThreads` sets evaluator concurrency and, with a store, store-worker
concurrency. Evaluators combine jobs across owners toward `MinibatchSize`,
accounting for immediate/cache results without exceeding the batch target;
store calls use each submitted job's batch.

Per worker, submitted I/O is bounded by two minibatches of item credits and
16 whole-job credits, shared by loads, evaluations, and persistence. Jobs
contain at most `ceil(MinibatchSize / 4)` items; persistence is submitted first,
then loads, then evaluations. Credits cover queued, executing, and
returned-but-unconsumed jobs and are released only when the owner consumes a
completion. The evaluation queue may span multiple backend batches, but is
not unbounded. `MaxActiveVisits` separately bounds the epoch-tagged VisitPool.

Workers refill their own visits in chunks of at most 32, targeting a rounded-up
share of the pool under the global admission/node-limit guard. They process
mailboxes, advance runnable visits, refill, and submit jobs; pending persistence
of at least one minibatch pauses refill/selection so writes can catch up.
The controller checks time/node limits, emits periodic info, and coordinates
shutdown/output; it does not drive visit refill or individual I/O completions.

`MaxBatchDelayMs` defaults to 2. A positive value gives a partial computation a
deadline measured from its first job's enqueue time. Dependency-starvation
flushing requires no worker able to produce more work and no outstanding job
outside evaluator collection. With `0`, the timeout is disabled, but starvation
flushing still occurs when no worker can produce, even if another I/O job is
outstanding. Both modes return immediate-only jobs without waiting for neural
work and flush partial batches on stop/abort. Metrics distinguish partial
starvation, timeout, and drain flushes; mailbox notifications, I/O-job high
water, and rejected publications expose scheduler activity. `ready_eval` and
its high-water metric count queued jobs, not individual evaluation items.

Stop and abort close admission and make owners cancel surviving visit
reservations, while tickets, continuations, and whole-job completions drain.
Accepted immutable payloads still drain to persistence after cancellation;
rejected publications are not stored. Blocking backend/store calls are not
interrupted, so shutdown waits for them and owner-side completion consumption.
Only after visits, tickets, jobs, pending requests, and mailboxes drain does the
controller join visit workers, close I/O queues, and join I/O workers. Stop emits
final info/bestmove; abort suppresses it unless output was already committed.

For issue #1734, selection (`GameGraph::SelectAndReserve`) and propagation
(`SearchRun::Backup` / `GameGraph::BackupNode`) remain separate named policy
seams. They currently implement concrete selection/backup rules, not a
pluggable policy API; changing those rules is independent of worker ownership,
mailboxes, or I/O thread counts.

## Limits and game lifetime

Supported limits include `nodes`, `movetime`, `infinite`, and ponder as infinite.
Without explicit `movetime`, the engine derives a budget from the side-to-move's
`wtime`/`btime` and `winc`/`binc`. It reserves `MoveOverheadMs` and uses
`AlphaZeroTimePct` as the clock fraction, raised to `1 / movestogo` when a
positive `movestogo` implies a shorter horizon. The budget is
`usable_clock * fraction + increment * (1 - fraction)`, capped at the usable
clock with a 1 ms floor when usable time remains. Clocks at or below overhead
receive a zero budget; infinite/ponder suppress clock-derived budgets.
Only `depth`, `mate`, and `searchmoves` are reported as ignored.
Thinking output includes a centipawn score converted from root Q, from the
side-to-move's perspective, alongside WDL when root value visits are available.
The default engine's hot graph survives same-game `position` changes;
`ucinewgame`, an incompatible starting position, and backend replacement clear
it.

## Verification performed during implementation

The following historical checks were run on the implementation host; they are
not a fresh validation of the current uncommitted scheduler:

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
