# Lc5 search

Lc5 is an experimental search, started with `lc0 lc5`. PUCT, FPU and their
defaults match classic (except that in-flight visits also count in the
parent's totals, see "Following one visit"); what differs is how the work is
organized. Code is in `src/search/lc5/`.

## Differences from classic

- **No gather/evaluate/backup loop.** Many independent **visits** are in
  flight at once. Each visit walks root→leaf, waits for the network if
  needed, backs up and finishes. Visit workers advance visits; evaluator
  threads batch whatever positions are pending.
- **A graph, not a tree.** Nodes live in a hash map keyed by the position
  plus the 7 before it, so transpositions with the same recent history share
  a node. Shorter keys
  (`HistoryKeyLength`) merge more transpositions.
- **Kept for the whole game.** The graph survives from move to move. On a new
  `position` the root is simply looked up again; nothing is promoted or
  pruned. It is cleared on `ucinewgame`, a different start position, or a
  backend change. By default nothing is evicted, so memory grows over the
  game; `Lc5GraphNodeLimit` caps the graph (see "Node store").
- **Statistics on edges.** Each edge records its own visits and value. When
  two parents lead into the same node, each parent's edge keeps its own
  numbers, so a transposition does not distort the other parent's view.
  (With `Lc5Policy=puct-dag` an edge caches the child's value instead, so
  every parent sees the shared node's full value; see "Policies".)
  Every legal move keeps only its move and prior (stored as fp16, 4 bytes in
  all); statistics exist only for the edges that have been tried, which
  always form a prefix of the prior-sorted edges.
- **Collisions wait instead of being discarded.** If a visit reaches a node
  that is already waiting for the network, it waits too. When the evaluation
  arrives, it carries on selecting below that node (see "Waiting" below).

## Following one visit

Code: `session/search_session.cc`, `policies/policy_engine.h`, `policies/puct.cc`.

1. **Admission.** A worker lets a new visit in at the root, if limits allow
   (see "How many visits are in flight"). Under `go nodes N`, at most N
   visits are ever admitted.

2. **Selection.** At each node the visit picks a move by PUCT
   (`GameGraph::SelectAndReserve`) over the edges tried so far plus the best
   untried one, and reserves the edge (`in_flight`). In-flight visits count
   in the edge's own U denominator and in the parent's sqrt(N) and FPU
   visited policy (counting only completed visits there lost about 50 Elo).
   With `MovesLeft` (needs a network with a moves-left head), in nodes
   whose |Q| exceeds `MovesLeftThreshold` a tried edge's Q also gets a small
   bonus for shortening a win or lengthening a loss, from the edge's M
   against the node's. With `StickyEndgames`, a proven edge (see Backup)
   scores its exact value and an unproven one's Q is kept within its
   bounds. Each step goes into the visit's `path` for
   backup or cancel.

3. **Reaching a leaf.** Selection ends in one of five ways:
   - The node is **terminal** (mate, stalemate, insufficient material,
     rule 50, threefold): back up its fixed value.
   - With `StickyEndgames`, the node is **proven** and is not the search
     root: back up its exact value without descending.
   - The position is a **two-fold repetition** whose cycle lies within this
     search (at least 4 plies from the root): back up a draw. This depends on
     the root, so the graph node itself is not marked terminal. With
     `TwoFoldDraws=contained`, the draw goes up only as far as the repeated
     position's first occurrence, which takes it; positions above receive
     that node's current value instead, so nodes kept for later searches do
     not hold draws that were only valid from this root.
   - The node **doesn't exist yet**: create it in a "materializing" state,
     become its owner and queue the position for the network.
   - The node is **already materializing**: become a waiter on it.

4. **Evaluation.** Evaluator threads gather queued positions from all workers
   into one backend call. When results return, the owning worker installs the
   node: its legal moves, priors and value.

5. **Waiting.** When a node is installed, its owner backs up the network's
   value before its waiters resume. Waiters do not back anything up. By default
   (`WaiterResume=reselect`) a waiter reruns PUCT at the node's parent (whose
   edge now holds the owner's backed-up value) and switches to another edge if
   that one is now better (about +7 Elo over `select`).
   With `WaiterResume=select` it continues selecting *below* the new node
   and ends up at some other leaf; with `WaiterResume=restart` it restarts
   from the root instead (this was the `RestartWaiters` option; in earlier
   testing it was worse).

6. **Backup.** Along the path, add the value to each node and edge and
   release the reservations. Node values are from the side to move at that
   node; edge values from the parent's side to move. Each node's update
   returns the value its parent receives: the averaged value with the
   perspective flipped, and the node's win/draw/loss bounds. A terminal's
   bounds are exact, except for repetitions, which depend on the history
   before the node (the node key covers checkmate, stalemate, insufficient
   material and rule 50). Each edge keeps the last bounds its child sent,
   and a node proves its own bounds from its edges: a win if any edge wins
   (the shortest), a loss if all edges lose (the longest), and an untried
   edge is unknown. Bounds are always tracked; only `StickyEndgames` uses
   them.

If the search stops, a visit that hasn't finished is **cancelled**: its
reservations are released and nothing is added.

## How many visits are in flight

This matters a lot for strength. At 10+0.1 a move is ~5k visits; with 1024 in
flight lc5 was ~100 Elo behind classic.

Two limits apply:

- `MaxActiveVisits`: a hard cap. By default
  `max(256, 1.5 × EvalThreads × MinibatchSize)`, enough to keep the GPU fed.
  It is at most 65535, because each edge counts its in-flight visits in
  16 bits.
- `ActiveVisitsRatio` (default 0.25): at most `16 + 0.25 × N` visits are live,
  where N is the root's visit count. A fresh search starts with only a few
  visits in flight, and more are allowed as the tree grows. Classic search
  scales its collision limit with tree size in the same way.

## Threads

- **Visit workers** (`Threads`) move visits forward. Each visit belongs to
  one worker, and only that worker touches it. Workers talk to each other
  through inboxes.
- **Evaluators** (`EvalThreads`) build and run backend batches.
- **The controller** (one thread per search) watches the clock, prints `info`
  once a second, and handles stopping.

A worker's loop: read its inbox (evaluation results, visits handed over by
other workers), advance up to 32 visits, admit new ones, send queued
positions to the evaluators, and sleep only when there is nothing to do.

## Batching

Workers hand positions to evaluators in small **jobs** (about a quarter of a
minibatch each). An evaluator packs jobs into one backend call until the
batch target (`MinibatchSize`) is reached. It also sends a partial batch
early when:

- `MaxBatchDelayMs` (default 2 ms) has passed since the first job arrived,
- no worker can produce more work, so waiting longer is pointless, or
- every position was a cache hit, so no GPU work is needed.

The NN cache is keyed on the last position only, so transpositions with a
different history (separate lc5 nodes) show up as cache hits.

### Node store

The **node store** is the graph's cold tier. It holds the search state of
nodes that left the graph: terminal kind, node statistics, legal edges with
priors, and the statistics of tried edges (`graph/node_record.h`).
Reservations, generations and tickets are not stored. A record is tagged with
the policy and a format version, and a record that does not match is treated
as a miss, so one policy never reads another's statistics. The store sees
records only as bytes; `MemoryNodeStore` keeps them in RAM.

With `Lc5GraphNodeLimit` > 0, the engine gives the `VisitEngine` a store,
which it owns and clears with the graph, and keeps the graph near that size:

- **Eviction.** When the graph grows past the limit (the controller checks
  every 10 ms, and a worker wakes it when it sees the graph over the limit),
  the controller queues an eviction down to 90% of the limit, and a store
  thread runs `VisitEngine::EvictToStore`. It picks nodes with the fewest visits,
  never the root, a materializing node, or a node with a reserved edge.
  Eviction has two phases, so a miss can never race ahead of the write: the
  records are taken under the shard lock along with each node's generation
  and version, written to the store, and then each node is erased only if its
  version is unchanged. A node that changed stays in the
  graph. Its stale record does no harm because the graph is always checked
  first.
- **Restore.** With a store, a graph miss loads from the store before
  evaluating. On a hit the node comes back with all its statistics under a
  new generation, so in-flight updates meant for the evicted node are
  rejected. Nothing is backed up: the owner and waiters all continue
  selecting below the node.

Fresh evaluations are not stored; that is the NN cache's job. The store is
cleared with the graph. Saving the whole graph to a file and loading it later
to continue a search is the planned next step.

## Time management

Under a clock, a move gets
`(time left − MoveOverheadMs) × 3% + increment × 97%`, plus any time saved
earlier (below). `movetime` minus `MoveOverheadMs` is used, as in classic;
`infinite` and `ponder` have no deadline. `AlphaZeroTimePct` sets the 3%.

**Early stop.** Classic-style smart pruning, checked every 10 ms against the
session's visit rate (`SmartPruningFactor`, 1.33), for clock budgets only. A root
with a single legal move, or with `StickyEndgames` a proven root, also stops
the search (checked every 10 ms, also under `go nodes` and `movetime`, but not
in `infinite` or `ponder`). Whatever stops a clock-budget search, the rest of
its budget goes to the next move; a search without a clock budget (ponder,
infinite, movetime) passes the saved time on untouched.

## Stopping and output

On a stop (deadline, node limit or UCI `stop`), the controller takes one
snapshot of the root and prints the final `info` and `bestmove` immediately.
It does not wait for in-flight work. After that, workers cancel their visits
and the session drains; any batch already sent to the GPU still finishes and is
installed. Only when everything has drained is the graph safe for the next
search.

`abort` (e.g. a new `position` during search) suppresses `bestmove` if it
arrives before the move was printed.

Reading the `info` line:

- `nodes`/`nps`: visits completed in this search.
- `eps`: network evaluations, excluding cache hits.
- `depth` = `seldepth` = the longest path. Not comparable to classic.
- `score mate N`: with `StickyEndgames`, a root proven won (N > 0, moves to
  mate) or lost (N < 0). Then `bestmove` is the shortest proven win, and a
  proven losing move is played only if all moves lose (the longest one).
- `string lc5 ...`: debug counters. `cache` counts NN cache hits,
  `suspended` counts visits that had to wait for the network, `proven`
  counts visits that backed up a proven node's value, `twofold` counts
  visits that ended at a two-fold repetition, `active` is the number of
  live visits, and `graph` is the node count.

## Code map

Four subdirectories: `graph/` is the node data model, `store/` is the
graph's cold tier, `session/` is the `SearchSession` machinery, and `policies/` holds
the search policies. `store/` sees only node keys and opaque records; the
graph does not include it. The root
has the engine, configuration, and the seam between `session/` and `policies/`
(`visit_engine.h`, `visit.h`, and `metrics.h`, which both sides bump).

| File | What's there |
|---|---|
| `engine.cc` | UCI glue: keeps the visit engine (and its graph), starts a `SearchSession` per `go`, carries saved time. |
| `visit_engine.h` | `VisitEngine` interface: descent, backup, cancel, report. |
| `visit.h` | Visit state passed to the visit engine. |
| `settings.cc` | Machinery options and their automatic values. |
| `metrics.h/.cc` | Debug counters, bumped by both `session/` and the policies. |
| `time_manager.cc` | Per-move deadline, smart pruning's visit estimate, saved time. |
| `session/search_session.h/.cc` | `SearchSession`: setup, controller, early stop, `info`/`bestmove`; admission, the worker loop and each visit's steps; creating nodes, terminal detection, jobs, evaluator batching. |
| `session/messages.h` | What `SearchSession`'s threads pass to each other: visit wakeups, tickets, materializations, jobs, batches, worker inboxes. |
| `session/visit_pool.h/.cc` | `VisitPool`: fixed visit slots addressed by epoch-tagged `VisitId`s. |
| `session/work_queue.h` | Closable MPMC queue for jobs. |
| `graph/graph.h` | `GameGraph<P>`: sharded node map, reservations, per-node updates, tried-edge stats; node and edge stats belong to the policy. |
| `graph/key.h`, `graph/value.h`, `graph/prior.h` | Node keys; value sums and perspective flips; fp16 priors. |
| `graph/node.h`, `graph/node_eval.h` | Node, edge and tried-edge state; node evaluations and terminal kinds. |
| `graph/node_record.h` | The node record format: encoding and decoding a node's statistics and edges for the policy. |
| `store/node_store.h`, `store/memory_node_store.cc` | The node store (the graph's cold tier) and its in-RAM implementation. |
| `policies/policy_engine.h` | `PolicyEngine<P>` implements `VisitEngine` on a `GameGraph<P>` it owns: descent, two-fold check, backup, PV walk. |
| `policies/puct.h/.cc` | `PuctPolicy<Self>`: PUCT selection, PV edge, bounds, score/WDL and the PUCT options (`PuctParams`), shared by the policies; `PuctAveraging` (`puct`) adds the averaging backup. |
| `policies/puct_dag.h/.cc` | `PuctDag` (`puct-dag`): edges cache the child's value; see "Policies". |
| `policies/registry.h/.cc` | `PolicyRegistry` and the `Lc5Policy` option. A policy registers itself with `REGISTER_LC5_POLICY(P, priority)` in its own .cc; the highest priority (`puct`) is the default. |

Things that look odd but are deliberate:

- **Generations.** Every node has a generation number, and updates name the
  generation they expect. An update meant for a node that was erased and
  recreated is then ignored instead of corrupting the new one.
- **Tickets.** A materializing node has a ticket listing its owner and
  waiters. A result is only installed if the ticket still matches.
- **Visit IDs.** A visit ID includes an epoch, so a late message for a
  finished visit is ignored even after its slot is reused.

## Options

| Option | Default | Meaning |
|---|---|---|
| `Threads` | auto | Visit worker threads. |
| `EvalThreads` | auto | Evaluator threads. |
| `MinibatchSize` | backend | Target batch size. |
| `MaxActiveVisits` | auto | Hard cap on live visits (at most 65535). |
| `ActiveVisitsRatio` | 0.25 | Live visits ≤ 16 + ratio × root visits; 0 disables. |
| `HistoryKeyLength` | 7 | Earlier positions included in node keys. |
| `MaxBatchDelayMs` | 2 | Max wait before sending a partial batch; 0 disables. |
| `MoveOverheadMs` | 200 | As classic. |
| `AlphaZeroTimePct` | 3 | Share of the clock spent per move. |
| `SmartPruningFactor` | 1.33 | Early stop aggressiveness; 0 disables. |
| `Lc5Policy` | puct | Search policy: `puct` or `puct-dag` (see "Policies"). |
| `Lc5GraphNodeLimit` | 0 | Graph nodes kept in RAM; extra nodes are evicted to the node store with their statistics. 0 means unlimited. |

Options of the `puct` and `puct-dag` policies (they share them):

| Option | Default | Meaning |
|---|---|---|
| `CPuct`, `CPuctBase`, `CPuctFactor` | 1.745, 38739, 3.894 | As classic. |
| `FpuStrategy`, `FpuValue` | reduction, 0.33 | As classic. |
| `WaiterResume` | reselect | After waiting for a node's evaluation: select below it, reselect at its parent, or restart from the root. |
| `MovesLeft` | true | Prefer shorter wins and longer losses in clearly decided nodes. Off without a moves-left head. |
| `MovesLeftMaxEffect`, `MovesLeftThreshold`, `MovesLeftSlope` | 0.0345, 0.8, 0.0027 | Largest bonus, node \|Q\| above which it applies, bonus per ply of M difference. |
| `MovesLeftConstantFactor`, `MovesLeftScaledFactor`, `MovesLeftQuadraticFactor` | 0, 1.6521, −0.6521 | Bonus scaling: a + b·x + c·x², x = edge \|Q\| above the threshold, rescaled to [0, 1]. |
| `StickyEndgames` | false | Use proven wins, draws and losses in selection, the bestmove, early stopping and mate scores. Repetition draws are not proven. |
| `TwoFoldDraws` | root | How far a two-fold repetition inside the search is backed up as a draw: to the search root, or contained at the repeated position's first occurrence, which sends its own value further up. |

## Policies

Both policies select by PUCT with the options above; they differ in what an
edge and a node remember.

- **`puct`.** An edge averages the values of the visits that went through
  it, and a node those of all visits to it. A transposed node pools its
  visits from all parents, but each parent's edge sees only its own share.
- **`puct-dag`.** An edge caches its child's current value from the last
  visit through it, and its visit count is its weight. A node's value is the
  visit-weighted mean of its own evaluation and its edges' cached values, and
  each update sends the node's current value up (not the leaf's). So a parent
  sees what the child learned through other parents too, from its next visit
  through that edge. Without `StickyEndgames`, a node's visits are always
  1 + its edges' visits. A two-fold draw is cached by the edge that reached
  it and goes further up only through the nodes' values, so `TwoFoldDraws`
  makes no difference.

With a short `HistoryKeyLength` more positions share a node, which matters
most for `puct-dag`. The key still includes the fifty-move counter, so the
graph has no cycles. A three-fold repetition found at a node is stored on it,
so with short keys another history reaching that node also sees the draw,
whatever its own repetitions (both policies).

## Not done yet

- Proving repetition draws, Syzygy, MultiPV, training data output, a
  memory-based limit (the limit counts nodes), an on-disk node store, and
  saving and reloading a graph.

`scripts/benchmark_lc5.py` is a speed benchmark harness.
