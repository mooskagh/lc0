# Lc5 search

Lc5 is an experimental search, started with `lc0 lc5`. PUCT, FPU and their
defaults match classic (except that by default in-flight visits also count in
the parent's totals, see `PuctTotals`); what differs is how the work is
organized. Code is in `src/search/lc5/`.

## Differences from classic

- **No gather/evaluate/backup loop.** Many independent **visits** are in
  flight at once. Each visit walks root→leaf, waits for the network if
  needed, backs up and finishes. Worker threads advance visits; evaluator
  threads batch whatever positions are pending.
- **A graph, not a tree.** Nodes live in a hash map keyed by the position
  plus the 7 before it, so transpositions with the same recent history share
  a node. Shorter keys
  (`HistoryKeyLength`) merge more transpositions.
- **Kept for the whole game.** The graph survives from move to move. On a new
  `position` the root is simply looked up again; nothing is promoted or
  pruned. It is cleared on `ucinewgame`, a different start position, or a
  backend change. There is no eviction, so memory grows over the game.
- **Statistics on edges.** Each edge records its own visits and value. When
  two parents lead into the same node, each parent's edge keeps its own
  numbers, so a transposition does not distort the other parent's view.
  Every legal move keeps only its move and prior (stored as fp16, 4 bytes in
  all); statistics exist only for the edges that have been tried, which
  always form a prefix of the prior-sorted edges.
- **Collisions wait instead of being discarded.** If a visit reaches a node
  that is already waiting for the network, it waits too. When the evaluation
  arrives, it carries on selecting below that node (see "Waiting" below).

## Following one visit

Code: `search_visits.cc`, `policy_engine.h`, `puct.cc`, `search_materialization.cc`.

1. **Admission.** A worker lets a new visit in at the root, if limits allow
   (see "How many visits are in flight"). Under `go nodes N`, at most N
   visits are ever admitted.

2. **Selection.** At each node the visit picks a move by PUCT
   (`GameGraph::SelectAndReserve`) over the edges tried so far plus the best
   untried one, and reserves the edge (`in_flight`). In-flight visits count
   in the edge's own U denominator; by default they also count in the
   parent's sqrt(N) and FPU visited policy, and with `PuctTotals=completed`
   they don't, as in classic. With `MovesLeft` (needs a network with a
   moves-left head), in nodes whose |Q| exceeds `MovesLeftThreshold` a tried
   edge's Q also gets a small bonus for shortening a win or lengthening a
   loss, from the edge's M against the node's. With `StickyEndgames`, a
   proven edge (see Backup) scores its exact value and an unproven one's Q
   is kept within its bounds. Each step goes into the visit's `path` for
   backup or cancel.

3. **Reaching a leaf.** Selection ends in one of five ways:
   - The node is **terminal** (mate, stalemate, insufficient material,
     rule 50, threefold): back up its fixed value.
   - With `StickyEndgames`, the node is **proven** and is not the search
     root: back up its exact value without descending.
   - The position is a **two-fold repetition** whose cycle lies within this
     search (at least 4 plies from the root): back up a draw. This depends on
     the root, so the graph node itself is not marked terminal.
   - The node **doesn't exist yet**: create it in a "materializing" state,
     become its owner and queue the position for the network.
   - The node is **already materializing**: become a waiter on it.

4. **Evaluation.** Evaluator threads gather queued positions from all workers
   into one backend call. When results return, the owning worker installs the
   node: its legal moves, priors and value.

5. **Waiting.** When a node is installed, its owner backs up the network's
   value. Its waiters do not back anything up; by default they continue
   selecting *below* the new node and end up at some other leaf. With
   `WaiterResume=reselect` a waiter first reruns PUCT at the node's parent,
   counting its own visit as completed with the node's value, and switches
   to another edge if that one is now better. With `WaiterResume=restart`
   it restarts from the root instead (this was the `RestartWaiters` option;
   in earlier testing it was worse).

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
  through mailboxes.
- **Evaluators** (`EvalThreads`) build and run backend batches.
- **The controller** (one thread per search) watches the clock, prints `info`
  once a second, and handles stopping.

A worker's loop: read its mailbox (evaluation results, visits handed over by
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

There is also an optional **node store** for persisting evaluated nodes, with
its own load/store jobs. The engine doesn't use it yet; it exists for future
eviction.

## Time management

Under a clock, a move gets
`(time left − MoveOverheadMs) × 3% + increment × 97%`, plus any time saved
earlier (below). `movetime` minus `MoveOverheadMs` is used, as in classic;
`infinite` and `ponder` have no deadline. `AlphaZeroTimePct` sets the 3%.

**Early stop.** Classic-style smart pruning, checked every 10 ms against the
run's visit rate (`SmartPruningFactor`, 1.33). Unused time goes to the next
move. With `StickyEndgames`, a search whose root is proven also stops
(checked every 10 ms, also under `go nodes` and `movetime`, but not in
`infinite` or `ponder`).

## Stopping and output

On a stop (deadline, node limit or UCI `stop`), the controller takes one
snapshot of the root and prints the final `info` and `bestmove` immediately.
It does not wait for in-flight work. After that, workers cancel their visits
and the run drains; any batch already sent to the GPU still finishes and is
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
  counts visits that backed up a proven node's value, `active` is the
  number of live visits, and `graph` is the node count.

## Code map

| File | What's there |
|---|---|
| `engine.cc` | UCI glue: keeps the visit engine (and its graph), starts a `SearchRun` per `go`, carries saved time. |
| `search.cc` | `SearchRun`: setup, controller, early stop, `info`/`bestmove`. |
| `search_visits.cc` | Admission, worker loop, driving each visit through the engine. |
| `visit_engine.h` | `VisitEngine` interface: descent, backup, cancel, report. |
| `policy_engine.h` | `PolicyEngine<P>` implements it on a `GameGraph<P>` it owns: descent, two-fold check, backup, PV walk. |
| `puct.h/.cc` | `PuctAveraging`, the policy: PUCT selection, averaging backup, its options, PV edge, score/WDL. |
| `policies.cc` | Policy table and the `Lc5Policy` option; a new policy is one row. |
| `search_materialization.cc` | Creating nodes, terminal detection, jobs, evaluator batching. |
| `graph.h` | `GameGraph<P>`: sharded node map, reservations, per-node updates, tried-edge stats; node and edge stats belong to the policy. |
| `visit.h` | Visit state and the fixed pool of visit slots. |
| `key.h`, `value.h` | Node keys; value sums and perspective flips. |
| `time_manager.cc` | Per-move deadline. |
| `settings.cc` | Machinery options and their automatic values. |
| `node_store.h`, `memory_node_store.cc` | Optional persistence (unused by the engine). |
| `metrics.h` | Debug counters. |

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
| `Lc5Policy` | puct | Search policy; only `puct` for now. |

Options of the `puct` policy:

| Option | Default | Meaning |
|---|---|---|
| `CPuct`, `CPuctBase`, `CPuctFactor` | 1.745, 38739, 3.894 | As classic. |
| `FpuStrategy`, `FpuValue` | reduction, 0.33 | As classic. |
| `PuctTotals` | started | Child visits counted in sqrt(N) and the FPU visited policy: started (with in-flight) or completed (as classic). |
| `WaiterResume` | select | After waiting for a node's evaluation: select below it, reselect at its parent, or restart from the root. |
| `MovesLeft` | false | Prefer shorter wins and longer losses in clearly decided nodes. Off without a moves-left head. |
| `MovesLeftMaxEffect`, `MovesLeftThreshold`, `MovesLeftSlope` | 0.0345, 0.8, 0.0027 | Largest bonus, node \|Q\| above which it applies, bonus per ply of M difference. |
| `MovesLeftConstantFactor`, `MovesLeftScaledFactor`, `MovesLeftQuadraticFactor` | 0, 1.6521, −0.6521 | Bonus scaling: a + b·x + c·x², x = edge \|Q\| above the threshold, rescaled to [0, 1]. |
| `StickyEndgames` | false | Use proven wins, draws and losses in selection, the bestmove, early stopping and mate scores. Repetition draws are not proven. |

## Not done yet

- Proving repetition draws, Syzygy, MultiPV, training data output, memory limits
  and eviction.

`scripts/benchmark_lc5.py` is a speed benchmark harness.
