# Lc5 search

Lc5 is an experimental search, started with `lc0 lc5`. PUCT, FPU and their
defaults match classic; what differs is how the work is organized. Code is in
`src/search/lc5/`.

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
- **Collisions wait instead of being discarded.** If a visit reaches a node
  that is already waiting for the network, it waits too. When the evaluation
  arrives, it carries on selecting below that node (see "Waiting" below).

## Following one visit

Code: `search_visits.cc`, `search_materialization.cc`.

1. **Admission.** A worker lets a new visit in at the root, if limits allow
   (see "How many visits are in flight"). Under `go nodes N`, at most N
   visits are ever admitted.

2. **Selection.** At each node the visit picks a move by PUCT
   (`GameGraph::SelectAndReserve`) and reserves the edge (`in_flight`, which
   counts toward started visits, as in classic). Each step goes into the
   visit's `path` for backup or cancel.

3. **Reaching a leaf.** Selection ends in one of four ways:
   - The node is **terminal** (mate, stalemate, insufficient material,
     rule 50, threefold): back up its fixed value.
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
   value. Its waiters do not back anything up; they continue selecting
   *below* the new node and end up at some other leaf. (With
   `RestartWaiters` they restart from the root instead; in testing that was
   worse, so it is off.)

6. **Backup.** Along the path, add the value to each node and edge and
   release the reservations. Node values are from the side to move at that
   node; edge values from the parent's side to move.

If the search stops, a visit that hasn't finished is **cancelled**: its
reservations are released and nothing is added.

## How many visits are in flight

This matters a lot for strength. At 10+0.1 a move is ~5k visits; with 1024 in
flight lc5 was ~100 Elo behind classic.

Two limits apply:

- `MaxActiveVisits`: a hard cap. By default
  `max(256, 1.5 × EvalThreads × MinibatchSize)`, enough to keep the GPU fed.
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

lc5 uses classic's time managers and stoppers, with the same options
(`TimeManager`, default `legacy`; `MoveOverheadMs`, `SmartPruningFactor`,
`Slowmover`, `RamLimitMb` and so on). The engine keeps one time manager per
game, so `legacy` carries its first-move bonus and its bank of saved time
from move to move. It is created at the first `go` after `ucinewgame` or a
new start position, so changed options take effect there.

For each `go`, the time manager builds a chain of stoppers: the clock budget,
`movetime` minus `MoveOverheadMs`, `go depth`, the RAM limit and smart
pruning. The controller asks the chain whether to stop as soon as the root
has been evaluated, and then at least every 10 ms. It passes the stoppers
classic's statistics: root visits per move, visits and batches in this run,
and average depth. lc5 has no proven results yet, so stoppers never see a
mate or a won position. When `bestmove` is sent, the chain is told the search
is done, which is how `legacy` banks unused time; an aborted search is not
reported.

`go nodes` is not passed to the stoppers. It stays exact: the run completes
exactly that many visits, smart pruning does not stop it early, and visits
reused from earlier searches do not count.

## Stopping and output

On a stop (a stopper, the node limit or UCI `stop`), the controller takes one
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
- `string lc5 ...`: debug counters. `cache` counts NN cache hits,
  `suspended` counts visits that had to wait for the network, `active` is
  the number of live visits, and `graph` is the node count.

## Code map

| File | What's there |
|---|---|
| `engine.cc` | UCI glue: keeps the graph and the time manager, starts a `SearchRun` per `go`. |
| `search.cc` | `SearchRun`: setup, controller, stopper checks, `info`/`bestmove`. |
| `search_visits.cc` | Admission, worker loop, selection, two-fold check, backup. |
| `search_materialization.cc` | Creating nodes, terminal detection, jobs, evaluator batching. |
| `graph.h/.cc` | `GameGraph`: sharded node map, PUCT selection, per-node updates. |
| `visit.h` | Visit state and the fixed pool of visit slots. |
| `key.h`, `value.h` | Node keys; value sums and perspective flips. |
| `settings.cc` | Options and their automatic values. |
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
| `MaxActiveVisits` | auto | Hard cap on live visits. |
| `ActiveVisitsRatio` | 0.25 | Live visits ≤ 16 + ratio × root visits; 0 disables. |
| `RestartWaiters` | false | Waiters restart from the root instead of continuing. |
| `HistoryKeyLength` | 7 | Earlier positions included in node keys. |
| `MaxBatchDelayMs` | 2 | Max wait before sending a partial batch; 0 disables. |
| `CPuct`, `CPuctBase`, `CPuctFactor` | 1.745, 38739, 3.894 | As classic. |
| `FpuStrategy`, `FpuValue` | reduction, 0.33 | As classic. |
| `TimeManager`, `MoveOverheadMs`, `SmartPruningFactor`, … | legacy, 200, 1.33 | Classic's time management options. |

## Not done yet

- Moves-left utility, mate propagation, Syzygy, MultiPV, training data
  output, memory limits and eviction.
- If no root move has a finished visit (e.g. `go nodes 1`), `bestmove` is an
  arbitrary legal move rather than the top policy move.

`scripts/benchmark_lc5.py` is a speed benchmark harness.
