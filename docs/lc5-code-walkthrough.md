# Reading the lc5 search

This is a code-reading guide for someone familiar with classical lc0. It follows
one logical visit from admission through selection, expansion and backup, then
covers reporting and shutdown. Links name the files to open; function names are
the navigation points. The subject is the current implementation, not the
proposed architecture in the implementation plan.

## 1. Establish the position and run

Start in [`engine.cc`](../src/search/lc5/engine.cc), with `SetPosition` and
`StartSearch`; [`engine.h`](../src/search/lc5/engine.h) shows the lifetimes.
`Lc5Engine` implements `SearchBase` and registers the `lc5` search factory.

The engine owns a `GameGraph` across searches. `SetPosition` drains the previous
run, rebuilds `PositionHistory`, and retains the graph if the supplied starting
position is unchanged. There is no subtree promotion or pruning: the new root
is a history and a key. `NewGame`, a changed starting position, or a changed
backend pointer clears the graph. The default engine has no store and no graph
eviction; `MaxActiveVisits` is not a graph-memory limit.

`StartSearch` resolves [`settings.cc`](../src/search/lc5/settings.cc) against
backend attributes and constructs a fresh `SearchRun`.
[`settings.h`](../src/search/lc5/settings.h) declares `Settings::Resolved` and
`FpuStrategy`. Thread counts, batch size
and visit capacity have backend-dependent defaults; explicit oversized batches
are rejected. PUCT/FPU settings are resolved here too. Time management is a
percentage of remaining time after overhead, not the classical stopper/time
manager. Increment, moves-to-go, depth, mate and searchmoves are ignored with
warnings. Ponder suppresses node/time limits; infinite suppresses clock-derived
time allocation, not explicit node or movetime limits. The outer engine handles
ponderhit by aborting and starting another run.

In [`search.cc`](../src/search/lc5/search.cc), the constructor records the limits
and creates worker state. `Start` launches evaluators, optional store workers,
visit workers and the controller. The graph, backend and responder are borrowed;
visits, tickets, queues, counters and threads belong to this run.

## 2. Know what is shared

Read [`graph.h`](../src/search/lc5/graph.h),
[`key.h`](../src/search/lc5/key.h), and
[`value.h`](../src/search/lc5/value.h) before following selection.

The graph is a sharded map, not a tree of owning node pointers. Nodes contain
aggregate values and outgoing edges; edges contain moves, priors, completed
statistics and `in_flight` reservations. They do not contain child pointers or
keys. A visit obtains the next key by appending the selected move to its own
history. Transpositions share node and outgoing-edge statistics; incoming edges
from different parents remain independent. Backup updates only the traversed
path, not every ancestor of a shared node.

`MakeNodeKey` hashes the current position and a configurable recent-history
window, including repetition and rule-50 information through `HashLast`.
`HistoryKeyLength` defaults to seven preceding positions. This is neither full
history equality nor verified position equality: short windows merge more
histories, and hash collisions have no secondary check. There is no mechanism
that enforces acyclic topology despite the DAG terminology.

Keep three kinds of reuse separate:

- The **graph** retains mutable search statistics and expansion edges across runs.
- A **NodeStore** retains expansion payloads, not statistics or reservations.
- The ordinary **backend NN cache** can satisfy evaluation immediately. Its
  current key is `Position::Hash`, not lc5's recent-history/rule-50 graph key.

Values are side-to-move at each node. `SearchValue::Parent` negates Q, preserves draw
probability, and adds one to moves-left. Outgoing-edge samples use their parent
node's perspective. Aggregates use double sums and float means.

## 3. Admit a logical visit

Open [`visit.h`](../src/search/lc5/visit.h),
[`search.h`](../src/search/lc5/search.h), then
[`search_visits.cc`](../src/search/lc5/search_visits.cc).

A `Visit` owns its history, current key, path, waiting ticket and backup value.
Each path step records a key, graph generation and optional selected move.
These records replace pointer ancestry and identify reservations to release.

`VisitPool` bounds concurrent visits. A `VisitId` combines slot index and epoch;
reused slots must not accept old continuations. Lookup is preliminary: callers
recheck the epoch/state under the slot mutex. Allocation reserves pool capacity
before initializing a slot and drops the pool lock before waiting for that slot.
Release runs slot-to-pool. This prevents a busy slot from blocking unrelated
allocation and avoids the reverse lock order.

`Admit` and `AdmitMore` hold `admission_mutex_` while calling `Allocate`,
serializing admission with stop. Successful
admissions consume the node budget, so a node-limited run cannot overshoot by
starting extra visits. Normal completion produces the requested count unless
time or explicit stop intervenes. Workers admit root visits toward a rounded-up
share of pool capacity; `Admit` also accepts arbitrary origins, round-robin.
An origin's optional backup prefix is caller-supplied ancestry, not reconstructed
by the search. It must already describe valid reservations if moves are present.

## 4. Run the worker loop and select

Follow `VisitWorker`, then `AdvanceVisit` in `search_visits.cc`.

A worker consumes mailbox continuations and job completions, admits visits,
advances up to 32 runnable visits, submits pending jobs, and delivers grouped
cross-worker continuations. The chunk bounds dispatches, not depth: one advance
can traverse several nodes. Workers own visit transitions; I/O threads do not
select, back up or cancel visits. The controller does not drive this loop.

`AdvanceVisit` locks the slot and checks stop/state. At each node it takes a
metadata snapshot. Missing or materializing nodes require suspension; terminal
nodes provide a fixed value. For expanded nonterminals, enter
`GameGraph::SelectAndReserve` in [`graph.cc`](../src/search/lc5/graph.cc).

Selection combines completed edge visits and `in_flight` into a started count.
That count enters the exploration denominator and visited-policy mass; the
exploration scale uses the sum of started counts. Q comes from completed edge
samples, or absolute/reduction FPU when none exist. CPuct grows logarithmically
with node visits. Ties use prior, then raw move encoding.

Selection and reservation increment happen under the same shard lock. A
reservation changes exploration pressure, but does not add a virtual loss or
completed value sample. The visit records the selected move, updates its history,
and continues. Generation checks handle disappearance/recreation between
snapshot and selection; no graph pointer is retained across steps.

## 5. Suspend for one expansion

Continue in
[`search_materialization.cc`](../src/search/lc5/search_materialization.cc),
starting at `SuspendForMaterialization`.

The first visit at a missing key creates a materialization ticket and request;
others attach as waiters. Suspension takes slot, ticket-registry, then graph-shard
locks. If the node became expanded in the meantime, selection retries instead.
A request copies history so its work can outlive the originating visit. Cancelling
the ticket owner does not cancel the expansion or discard other waiters.

Without a store, the request goes directly to worker-side `PrepareEvaluation`.
`DetectTerminal` checks no legal moves, mating material, rule-50 and repetition,
in that order. Mate is Q=-1; draws have D=1. Terminal publication does not call
the backend. Nonterminal requests retain the generated legal moves for NN input.

With a store, lookup precedes this preparation. See
[`node_store.h`](../src/search/lc5/node_store.h) and
[`memory_node_store.cc`](../src/search/lc5/memory_node_store.cc), with its layout in
[`memory_node_store.h`](../src/search/lc5/memory_node_store.h). Payloads contain
terminal kind, original leaf Q/D/M, moves and priors. Loads copy payloads; stores
replace entries. The memory implementation uses 64 shards, has no capacity bound,
and treats a batch as individual operations rather than one transaction. A hit
bypasses fresh terminal detection. Store compatibility and invalidation are the
caller's responsibility. The default engine passes `nullptr`; tests exercise
this optional path with injected stores.

## 6. Submit jobs and collect a backend batch

Follow `SubmitJobs`, `EvaluatorWorker`, `StoreWorker`, then `ConsumeJob`.

Each worker has 16 whole-job credits and two minibatches of item credits, shared
by loads, evaluations and persistence. A job contains at most a quarter minibatch
rounded up. Persistence is submitted first, then loads, then evaluation. A pending
minibatch of writes pauses new selection/refill so persistence can catch up.
Credits include queued, executing and returned-but-unconsumed work; only owner
consumption releases them. The FIFO queues themselves have no capacity bound.

Evaluators combine whole jobs across owners. `AddInput` receives history, legal
moves and pointers into payload storage. Heap-owned jobs remain intact through
computation destruction and completion consumption because backends may retain
those pointers, including mixed immediate/queued results. `UsedBatchSize`, not
request count, measures actual NN occupancy. Immediate cache hits can therefore
make total requests exceed the neural batch target; conservative whole-job
fitting prevents queued NN work from exceeding it.

A partial batch waits until its target, the first job's enqueue-time deadline,
dependency starvation, or stop. It also flushes when the next whole job cannot
fit conservatively; that job is retained for the next computation. Ready jobs
are consumed before waiting. An
immediate-only computation returns without waiting for neural work.
`MaxBatchDelayMs=0` disables the timeout, not starvation flushing. With positive
delay, starvation requires no producing worker and all outstanding jobs held
by collectors; with zero delay, no producing worker suffices. `producing` is a
conservative worker signal, not an exact queue length. These checks prevent
visits suspended on evaluation from waiting for inputs they cannot generate.

Store threads only perform blocking batch calls. Evaluators only execute backend
computations. Both return jobs to the originating worker's mailbox. That worker
performs publication, store-miss preparation and persistence enqueueing before
accounting the job as consumed.

## 7. Publish and resume

Read `PublishEvaluation`, `CompleteMaterialization` and `InstallPayload` together.

NN priors are sorted by descending prior and raw move tie-break; lc5 does not
apply another normalization. Publication accepts only the matching materializing
ticket, or a missing node. An already expanded node is not overwritten. The
explicit `accepted` result matters: a returned generation alone is not evidence
that this payload was installed.

Publication installs edges but does not seed node value sums from the payload.
On acceptance, the owner resumes for leaf backup. Nonterminal waiters resume
selection from the newly expanded node; they do not reuse the owner's evaluation
as additional samples. Terminal waiters do back up the terminal value. Rejected
publication resumes selection without backing up or persisting the rejected
payload. Fresh accepted payloads are persisted when a store exists; store hits
are not written back.

Registry/shard locks are dropped before resuming slots or delivering mailboxes.
Epoch checks reject continuations for released visits. Consequently, expansion
may finish after its owner was cancelled, leaving an expanded node with zero
completed visits. That is separate from successful publication.

## 8. Back up or cancel

Return to `Backup`, `Cancel`, and `FinishVisit` in `search_visits.cc`, and inspect
`BackupNode` and `CancelEdge` in `graph.cc`.

Backup walks the recorded path backward, updating the node and, where recorded,
completing its selected edge. Perspective changes between path steps. Node/edge
updates share one shard lock; an edge completion decrements `in_flight` and adds
one sample. A missing/stale generation skips the node update. A missing edge or
reservation underflow does not undo an otherwise valid node update. These are
separate outcomes, recorded in metrics.

Cancellation releases recorded edge reservations without adding samples.
Both paths release the pool slot under its mutex; capacity notifications occur
after dropping slot locks. A logical
visit counts as completed even if some graph updates were rejected. Stop is
checked before backup, not between every step of an already-started backup.

Generations protect paths against erased/recreated nodes; epochs protect visits
against reused slots; ticket IDs identify expansions. None substitutes for the
others. `Erase` is currently test-only, and generation checks are not evidence
of a production eviction policy.

## 9. Report, stop and retain

Return to `Controller`, `BuildPv`, `OutputInfo` and `RequestStop` in `search.cc`.

The controller checks limits and periodically reports. UCI nodes/NPS count
completed visits in this run; EPS counts backend-used NN items, excluding
immediate hits. Root WDL and PV use retained graph statistics. Their totals need
not match current-run counters. Depth metrics count path nodes, not conventionally
just edges. PV follows completed edges, ranking visits, Q, prior and move encoding;
it recomputes keys from history, handles output orientation, detects repeated
keys and caps length. If no visited root edge exists, bestmove falls back to the
lowest raw-encoded legal move, not the highest prior. No legal move yields null.

[`metrics.h`](../src/search/lc5/metrics.h) lists accounting and diagnostic counters;
[`metrics.cc`](../src/search/lc5/metrics.cc) formats a subset. `ready_eval` counts
queued jobs, not requests. Eviction accounting exists but has no production writer.

Stop closes admission and asks owners to cancel remaining visits. Already-created
expansion and persistence work still drains; blocking backend/store calls are
not interrupted. Drain requires no visits, tickets or outstanding jobs, empty
mailboxes, and worker acknowledgements covering their pending local requests.
Then the controller joins visit workers, closes executor queues and joins I/O
threads. `Wait` joins this controller.

Stop requests final info/bestmove; abort can override it until output commitment
under the admission mutex. `Finished` is published after output. Queue/mailbox
mutex handshakes and the controller notification generation avoid lost wakeups;
see [`search_internal.h`](../src/search/lc5/search_internal.h) for queue mechanics.
No graph/slot/registry lock spans backend or store calls. The retained graph is
available to the next run only after this drain.

## 10. Review tests and boundaries

Finish with the four test files:

- [`graph_test.cc`](../src/search/lc5/graph_test.cc): publication, generations,
  reservations, perspective and node/edge update outcomes.
- [`visit_test.cc`](../src/search/lc5/visit_test.cc): reuse, stale IDs, pool/slot
  lock ordering and concurrent allocation/release.
- [`search_test.cc`](../src/search/lc5/search_test.cc): exact budgets, owner/waiter
  behavior, terminal bypass, backend-buffer lifetime, batching/credits,
  blocked-I/O drains, store rehydration and game retention.
- [`settings_test.cc`](../src/search/lc5/settings_test.cc): backend-dependent
  resolution, batch rejection and clock allocation.

`ExpectDrained` in the search tests checks visit accounting, absent tickets and
reservations, and ordinary-run rejection/underflow counters. Rehydration tests
use a fresh graph, not live-search eviction. Nonempty origin prefixes, history-key
merging and live eviction are not comprehensively exercised here.

[`meson.build`](../meson.build) wires production sources under `lc5` and the four
test targets under `gtest` plus `lc5`. The concrete selection and backup functions
are policy seams, not a pluggable policy API. This implementation does not include
classical search's full time management, Syzygy, MultiPV, training output, pruning
or mate-bound machinery. Keep those omissions distinct from scheduler mechanics
when reviewing behavior.
