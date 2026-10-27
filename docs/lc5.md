# Lc5 search

Lc5 is an experimental search selected with `lc0 lc5`. It searches a
game-scoped DAG keyed by recent position history, runs many independent
*visits* concurrently on visit-worker threads, and decouples neural evaluation
(and an optional payload store) behind whole-job I/O queues. Sources live in
`src/search/lc5/`; this document follows the code from the outside in.

## Engine and lifetime

`Lc5Engine` (`engine.cc`) implements `SearchBase`. It keeps one `GameGraph`
for the whole game: same-game `position` commands keep it (the new root is just
a history and a key; nothing is promoted or pruned), while `ucinewgame`, a
different starting position, or a different backend clear it. There is no
eviction, so the engine passes no `NodeStore`. `MaxActiveVisits` bounds live
visits, not graph memory.

Each `go` creates a `SearchRun` (`search.h`) with resolved `Settings`, the
original `GoParams`, a `TimeManager`, and the `StartClock()` origin (or the
current time if none was set). Preparation after that origin is charged to
the move. `Wait`, `Finished` and destruction are full drain/join barriers;
receiving `bestmove` alone does not make the graph reusable yet.

## Data model

**Keys.** `MakeNodeKey` (`key.h`) hashes the last position plus
`HistoryKeyLength` (default 7) preceding ones via `HashLast`, so rule-50 and
repetition state are included. Shorter windows merge more histories. Hash
collisions are not verified.

**Graph.** `GameGraph` (`graph.h`) is a 1024-shard map from key to
`NodeState`: generation, lifecycle (`kMaterializing`/`kExpanded`), ticket,
terminal kind, value stats and outgoing `EdgeState`s (move, prior, `in_flight`
reservations, value stats). Edges hold no child pointers; a visit derives the
child key from its own history. Transpositions share a node and its outgoing
edges, while incoming edges from different parents keep separate statistics.
Every mutation names the generation it expects, so updates computed against
an erased and recreated node are rejected (`Erase` is currently test-only).

**Values.** `SearchValue` (`value.h`) is from the side to move at its node;
`Parent()` negates Q, keeps D and adds one to M. Sums are doubles, means
floats.

**Visits.** `VisitPool` (`visit.h`) is a fixed pool of slots. A `VisitId` packs
slot index and epoch, so continuations for a finished visit are ignored even
after the slot is reused. A visit owns its history, current key, and a path of
`(key, generation, selected move)` steps that identify the reservations to
complete or release. `VisitOrigin::backup_prefix` lets callers start visits
below the root.

## Visit lifecycle

**Admission.** `AdmitMore` and `Admit` allocate under `admission_mutex_`,
which also serializes stop requests. Admission consumes the node budget, so a
`go nodes N` run never starts more than N visits. Each worker refills root
visits in chunks of up to 32, up to a rounded-up share of the pool. When the
pool is full, a worker registers as waiting and is woken when another worker
releases capacity.

**Worker loop.** `VisitWorker` processes its mailbox (continuations and job
completions), advances up to 32 runnable visits, refills, submits I/O jobs,
delivers cross-worker continuations, and sleeps only when it has nothing left
to do locally. Only the owning worker ever mutates a visit.

**Selection.** `AdvanceVisit` walks down the graph. At each node it takes a
metadata snapshot. A missing or materializing node suspends the visit, a
terminal node backs up its fixed value, and an expanded node goes through
`GameGraph::SelectAndReserve`: PUCT with logarithmic CPuct growth, started
counts (`visits + in_flight`) in the exploration term, absolute or reduction
FPU for unvisited edges, and ties broken by prior and then move encoding.
Reserving an edge adds exploration pressure but no virtual-loss sample.

**Materialization.** `SuspendForMaterialization` locks slot → tickets → graph
shard. The first visit at a key creates a ticket and a `Request`; later visits
join as waiters. Without a store the request waits in the worker's
`unprepared` queue. With one it goes to `loads` first, and store misses are
prepared afterwards. Preparation (`DetectTerminal`) runs outside slot locks:
no legal moves, insufficient material, rule 50 and repetition are terminal and
publish immediately. Otherwise the legal moves are kept for the network.

**Publication.** `CompleteMaterialization` installs the payload only if the
node is still materializing under the same ticket (`InstallResult::accepted`).
On acceptance the owner backs up the leaf value. Waiters select below the new
node instead, except at terminal nodes, where they back up the terminal value.
A rejected publication resumes everyone in selection and is neither backed up
nor persisted. Accepted fresh payloads are queued for persistence when a store
exists.

**Backup and cancel.** `Backup` walks the path from the leaf, adding the value
to each node and completing each ancestor's reserved edge, with the
perspective flipped at every step. A stale node, missing edge or reservation
underflow is skipped and counted in metrics without undoing the other updates.
`Cancel` releases reservations without adding samples.

## I/O scheduling

Requests travel in whole `Job`s (`kPersist`, `kLoad`, `kEval`) of at most
`ceil(MinibatchSize / 4)` items, submitted in that priority order. Per worker,
in-flight I/O is bounded by 16 jobs and two minibatches of items. Credits cover
queued, executing and returned-but-unconsumed jobs and are released only when
the owner consumes the completion. A full minibatch of pending writes pauses
selection until persistence catches up.

`EvaluatorWorker` combines jobs from all owners into one backend computation
(`CollectBatch`), adding a job only if its size still fits the batch target.
A job that doesn't fit is kept for the next batch. Cache hits
(`FETCHED_IMMEDIATELY`) complete without computing. A partial batch is flushed
when:

- the deadline passes: `MaxBatchDelayMs` (default 2) after the first job was
  enqueued; `0` disables the deadline;
- the batch is *starved*: no worker can produce more work, and (with a
  positive delay) every outstanding job is already held by a collector;
- it holds only cache hits, which return at once instead of waiting.

Store threads only call `LoadBatch`/`StoreBatch`. Evaluators only add inputs
and compute. All results return to the owning worker's mailbox, and that worker
publishes, prepares store misses and queues persistence. Each `Request` carries
one `State` (pending, store hit/miss, queued, evaluated), so an abandoned
output is never mistaken for a result. A computation is destroyed before its
jobs are returned.

## Stop, abort and output

`Stop` and `Abort` set the stop mode under `admission_mutex_`, closing
admission atomically, and wake the controller. On stop, the controller builds
the final info and `bestmove` from a single root snapshot, without waiting for
outstanding work, and commits it under the same mutex. An `Abort` that arrives
before that commit suppresses the output; after it, the output cannot be
retracted. Only then are workers and collectors woken to drain.

Every nonempty backend computation is authorized under `admission_mutex_` just
before it runs, so after a stop each evaluator has at most the one computation
it had already authorized. Workers cancel their visits and pending requests.
Authorized results, store hits and accepted payloads still publish and persist.
Blocking backend and store calls are not interrupted. Once no visits, tickets,
jobs or mailbox items remain and every worker has acknowledged that it is
idle, the controller joins the visit workers, closes the I/O queues and joins
the I/O threads.

UCI `nodes`/`nps` count completed visits in this run, and `eps` counts backend
evaluations, excluding cache hits. Score (centipawns from root Q) and WDL come
from the retained root statistics. The PV follows completed edges by visits,
then Q, then the tie-break. `Metrics` (`metrics.h`) holds diagnostic counters,
and a subset goes into the info `comment`.

## Time management

`TimeManager` (`time_manager.h`) turns `GoParams` into an optional fixed
deadline. The controller calls `Evaluate(now)`, which either requests a stop or
returns the deadline as the next time to check. Precedence:

1. `ponder` disables all time and node limits. `ponderhit` restarts the search
   without ponder and with a new clock origin.
2. An explicit `movetime` is used as is, without subtracting overhead.
3. `infinite` disables clock allocation, but `movetime` and `nodes` still
   apply.
4. Otherwise, the active side's clock is required; without it there is no
   time limit.

The allocation is
`(clock - MoveOverheadMs) * f + increment * (1 - f)`, where
`f = max(AlphaZeroTimePct / 100, 1 / movestogo)`. Defaults are 200 ms of
overhead and 3%. The result is clamped to [1 ms, usable clock], and a clock at
or below the overhead gets zero. Nonnegative `nodes` limits are exact
admission caps.

## Options

| Option | Default | Meaning |
|---|---|---|
| `Threads` | auto | Visit workers. |
| `EvalThreads` | auto | Concurrent evaluators (and store workers). |
| `MinibatchSize` | backend | Neural batch target. |
| `MaxActiveVisits` | auto | Live visit bound. |
| `HistoryKeyLength` | 7 | Positions beyond the current one in node keys. |
| `MaxBatchDelayMs` | 2 | Partial batch deadline; 0 disables it. |
| `CPuct`, `CPuctBase`, `CPuctFactor` | 1.745, 38739, 3.894 | PUCT. |
| `FpuStrategy`, `FpuValue` | reduction, 0.33 | First-play urgency. |
| `MoveOverheadMs`, `AlphaZeroTimePct` | 200, 3 | Time allocation. |

## Known limitations

- If no root edge has a completed visit (e.g. `go nodes 1` on a fresh root, or
  an exhausted clock), `bestmove` is the legal move with the lowest encoding,
  not the highest-prior one, even when the root has already been evaluated.
- Not implemented: classical time management, Syzygy, MultiPV, training data,
  tree pruning, mate bounds and eviction. Selection (`SelectAndReserve`) and
  backup (`Backup`/`BackupNode`) are concrete functions, not a pluggable policy
  API.
- `scripts/benchmark_lc5.py` is the benchmark harness. No strength or
  performance claims are made here.
