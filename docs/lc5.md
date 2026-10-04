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
work. Stop/abort abandons unauthorized partial batches instead of flushing them.
Metrics distinguish partial starvation and timeout flushes; mailbox notifications,
I/O-job high water, and rejected publications expose scheduler activity. `ready_eval` and
its high-water metric count queued jobs, not individual evaluation items.

Stop and abort atomically close admission under the same mutex used to authorize
NN computations, then promptly notify the controller. On responding stop, the
controller builds one value-owned final info/PV and bestmove before inspecting
any drain predicate or synchronizing wakeups with individual worker mailboxes.
The root is snapshotted once, so final score and the first PV move use the same
snapshot. Final info may report nonzero active work and queued evaluation jobs.
Abort suppresses output before commitment under the admission mutex; after
commitment it cannot retract the response. The stop phase must run even if all
work has already drained when an external stop arrives.

If no completed root edge exists, including an unexpanded root, bestmove uses
the deterministic legal `FallbackMove` (or the empty move if no legal move
exists). There is no bootstrap exception, protected visit, or required evaluation
before responding. Completed graph statistics are retained without fabricated
value samples.

Every nonempty backend computation needs final authorization under the admission
mutex, which is released before entering the backend. No authorization is
possible after stop. Each serial evaluator therefore has at most one outstanding
authorized NN computation at stop, including one already executing or about to
enter the backend, not an additional shutdown computation on top. Whole jobs
must conservatively fit the batch target before `AddInput`, preventing hidden
batch-split computations. Worker-local requests are cancelled without preparation
or submission; queued loads are skipped, an in-flight hit may publish, and a
returned miss after stop cannot evaluate. Immediate cache results and successful
authorized results remain valid; abandoned outputs never publish default values.

Owners cancel surviving visit reservations separately from request retirement.
An abandoned owner request retires its ticket and key mapping under the ticket
mutex and conditionally erases only the matching materializing placeholder
under one graph-shard mutex. Expanded or replacement nodes are never removed.
This also covers suspension racing with stop; whole-run cancellation does not
need to resume ticket waiters. Computations are destroyed while their referenced
jobs remain alive, then owners consume returned jobs and release credits once.

Accepted immutable payloads still drain to persistence after cancellation;
rejected publications are not stored. Blocking backend/store calls are not
interrupted. Only after visits, tickets, jobs, pending requests, and mailboxes
drain does the controller join visit workers, close I/O queues, and join I/O
workers. `Wait`, `Finished`, and destruction remain full drain/join barriers;
receiving bestmove alone does not make the retained graph reusable yet.

For issue #1734, selection (`GameGraph::SelectAndReserve`) and propagation
(`SearchRun::Backup` / `GameGraph::BackupNode`) remain separate named policy
seams. They currently implement concrete selection/backup rules, not a
pluggable policy API; changing those rules is independent of worker ownership,
mailboxes, or I/O thread counts.

## Limits and game lifetime

Each `SearchRun` owns a value-semantic `TimeManager`. The engine passes the
original UCI `GoParams`, `Settings::time_management()` configuration (separate
from backend-dependent `Settings::Resolved`), and the existing start timestamp;
it does not rewrite clock allocation into `movetime`. The manager uses the root
side-to-move and stores only an optional fixed deadline. It neither reads the
clock internally nor owns workers, node accounting, or output.

Limit precedence is unchanged:

- Ponder suppresses all time and node limits, including explicit limits. The
  outer engine handles `ponderhit` by aborting/draining and starting a fresh run
  with ponder cleared and a new clock origin.
- Otherwise, explicit `movetime` wins over clock allocation and is used
  literally, without subtracting `MoveOverheadMs`. Zero or negative values are
  already due.
- Otherwise, `infinite` suppresses clock allocation. Explicit `movetime` and
  node limits still apply under `infinite` and can produce bestmove.
- Otherwise, allocation requires the active side's `wtime`/`btime`; a missing
  active-side clock means no time limit, even if an increment is supplied.

Clock allocation reserves `MoveOverheadMs` (default 200 ms) and uses
`AlphaZeroTimePct` (default 3%) as the fraction, raised to `1 / movestogo` when a
positive `movestogo` implies a shorter horizon. With
`usable_clock = remaining_clock - overhead`, the budget is
`usable_clock * fraction + increment * (1 - fraction)`. Only the active side's
`winc`/`binc` is used; absent or negative increments contribute zero. Calculation
uses long-double precision, clamps to a 1 ms floor and the usable-clock cap,
then truncates to whole milliseconds. Clocks at or below overhead receive a
zero budget.

Nonnegative `nodes` limits remain exact admission caps enforced by search,
independently of the manager; negative node limits are ignored. UCI nodes count
completed visits, not backend evaluations, and stopping early may complete fewer
visits than the cap.

The deadline is relative to the existing `StartClock()` steady-clock origin, so
preparation after that origin is charged before search initialization. The outer
engine retains its position/go timing rules; if no origin was set, `StartSearch`
uses its current time. On each running controller iteration,
`TimeManager::Evaluate(now)` requests stop at/after the deadline; before it, the
manager returns the deadline as `next_check`. With no time limit it requests
neither stop nor a timed check. The controller calls the existing `Stop()` when
requested and waits until the earlier of periodic info and `next_check`, or an
enabling notification. During stop/abort drain it uses lifecycle notifications,
not an expired policy wakeup.

A time decision requests stopping, not a hard bestmove response deadline:
final output does not wait for outstanding backend/store work to drain, but
controller scheduling and snapshot/output work still take time. Visit workers
start before the controller, so an already-expired budget can race with initial
admission and does not guarantee zero visits or evaluations. Only a computation
authorized before admission closes may run after stop.

This is a concrete policy seam, not the classical time manager or an adaptive
algorithm. A future policy could extend `Evaluate` with a small
controller-supplied statistics snapshot and request earlier reevaluation; today
allocation stays fixed, with no extra threads or persistent cross-move state.

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
