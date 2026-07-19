# Lc5 Search Backend: Detailed Implementation Plan

Status: approved design, not yet implemented  
Repository baseline: commit 1c92f516 on branch experiment-lc3-bounded-pipeline  
Intended audience: an engineer or coding agent starting without the design
conversation that produced this document

## 1. Executive summary

Lc5 is a new experimental search backend. It must keep the useful ideas behind
LC3—key-addressed nodes, data-oriented traversal, decoupled neural evaluation,
and eventual non-memory storage—without inheriting LC3's fixed speculative
waves, collision rollbacks, root-only assumptions, or repository handles that
cannot survive eviction.

The implementation must have these defining properties:

1. It is a real session DAG. Transposed histories with the same configured node
   key share one node and its outgoing search state. Different parents keep
   independent incoming-edge statistics.
2. Every logical visit is explicitly owned from admission through completion or
   cancellation. There is no anonymous speculative visit count and no
   gather-exactly-N requirement.
3. Reaching an evaluation already in progress suspends the visit. After the
   node expands, the visit continues below it instead of becoming a collision
   rollback.
4. Ready neural evaluations may exceed the next backend batch. The evaluator
   consumes only the next batch and leaves the surplus queued with intact visit
   ownership.
5. Search-side parallelism is first-class. Independent visit workers mutate a
   short-lock, sharded graph. There is no single graph-mutation owner.
6. Paths contain keys, node generations, and moves—never persistent Node
   pointers. Eviction may lose mutable statistics, but it cannot cause a
   dangling access, counter underflow, deadlock, or crash.
7. Reusable immutable expansion payloads are accessed through a batched
   NodeStore interface. The first implementation is memory-backed, while the
   caller already treats storage latency as asynchronous work.
8. The first release is a research UCI engine. It supports node limits,
   movetime, infinite search, stop/abort, useful thinking output, whole-game
   graph reuse, and the core PUCT/FPU settings. It is not yet a self-play,
   training-data, tablebase, or full time-management replacement.

Lc5 must be implemented in a new src/search/lc5 directory. Do not copy LC3 and
rename it. Reuse generic repository facilities where they are genuinely useful,
but keep the new search's ownership and lifecycle model independent.

## 2. Evidence and assessment that motivate the design

### 2.1 Static assessment of the current search implementations

At the baseline commit:

- Classic's core node/search/wrapper implementation is approximately 4,232
  lines, excluding its stopper/time-management subsystem.
- DAG-preview's corresponding core is approximately 5,220 lines.
- LC3 is approximately 2,511 non-test lines.

The LC3 number is not an apples-to-apples simplicity win. LC3 currently omits
or hardcodes substantial behavior:

- SearchPolicy::MakeNodeKey hashes parent key plus move, making the repository
  a tree rather than a DAG.
- StartClock and GoParams are TODOs; node/time/depth limits are ignored.
- PUCT, FPU, root behavior, draw scoring, tablebases, searchmoves, tree
  lifecycle, and most output behavior are absent or hardcoded.
- NodeRepository is a sharded in-memory flat hash map whose NodeHandle retains
  a lock and direct pointer. That API cannot naturally support remote access,
  disk latency, or arbitrary eviction.
- Backpropagation asserts that every path node still exists.
- The current bounded-wave patch added a 261-line search_pipeline.h and
  substantial cross-stage token plumbing to contain a positive feedback loop
  introduced by the gather/eval/backprop decomposition.

The direction is therefore partially correct, but the current architecture is
not the right foundation:

- Correct direction: key-addressed state, data-oriented bulk work, decoupled
  evaluation, explicit lifecycle states, metrics.
- Incorrect foundation: fixed 2,560-visit waves, approximate bulk visit
  distribution, rollback-oriented collisions, hard-coded memory storage,
  root-bound scheduling, and stage-specific worker ownership.

### 2.2 Measured backend and search baselines

Measurements were taken with:

- NVIDIA GeForce RTX 5090
- BT4-1024x15x32h-swa-6147500.pb.gz
- cuda-auto/cuda-fp16
- neural cache disabled for search comparisons

Backend-only throughput:

| Batch | Approximate eval/s |
|------:|-------------------:|
| 128   | 13,332 |
| 256   | 12,010 |
| 384   | 11,266 |
| 512   | 10,333 |
| 1024  | 10,681 |

Search observations:

- Classic with the backend-recommended/default minibatch: about 11.5k eval/s.
- Classic with minibatch 128: about 12.9k eval/s.
- LC3 with its default 2/2/2 stage workers: about 12.0k eval/s.

This means the current machine/network is not limited by classic's inability to
gather batches larger than roughly 1,000. The backend reaches peak throughput at
a much smaller batch. Improving the backend batch recommendation is more
important for this exact workload than producing giant batches.

The large-batch problem remains real for other backend/network/hardware
combinations. Lc5 must solve it without making giant batches mandatory.

CPU-only/trivial-backend observations:

- Classic in the sampled configuration: roughly 0.8M nodes/s.
- LC3 2/2/2: roughly 1.3M nodes/s.
- LC3 1 gather / 1 eval / 1 backprop: roughly 0.7M nodes/s.
- LC3 1 gather / 8 eval / 1 backprop remained around 0.68M nodes/s.
- LC3 8/8/8 reached roughly 1.38M nodes/s on a 16-core/32-thread Ryzen 9950X.

These figures are not strength comparisons because the algorithms and thread
budgets differ. They do show two things relevant to Lc5:

1. Data-oriented search execution can materially reduce CPU overhead.
2. A single gather or mutation owner becomes a ceiling at high backend
   throughput. Lc5 must be capable of parallel selection and backup for
   projected systems such as 8xB200 with hundreds of CPU cores.

### 2.3 Required benchmark discipline

All future Lc5 performance claims must distinguish:

- completed logical visits per second (NPS);
- actual neural evaluations per second (EPS);
- backend batch-size distribution;
- cache/store hits;
- terminal visits;
- active and suspended visit counts;
- total CPU thread budget;
- queue high-water marks.

Do not compare Lc5's completed visits with LC3's spawned speculative visits.
Do not call a higher NPS result a win if it uses a materially different total
CPU thread count without reporting that difference.

## 3. Goals, non-goals, and success criteria

### 3.1 Goals

Lc5 v1 must:

- register and run as the UCI search mode named lc5;
- perform exact one-visit-at-a-time PUCT decisions, including in-flight
  reservations;
- keep a mutable DAG for the duration of a game and re-root it across searches;
- share a node across transpositions according to a configurable history key;
- retain independent edge statistics for different transposition parents;
- keep enough ready work to overlap selection, storage, and neural evaluation;
- permit more ready evaluations than one backend batch;
- suspend and resume pending-node visits without collision rollback work;
- scale selection and backup over configurable visit workers;
- cleanly drain or cancel every visit and evaluation during stop/abort;
- support node-limited, movetime, infinite, and ponder-as-infinite UCI searches;
- produce legal bestmove/PV output and useful performance diagnostics;
- expose an internal arbitrary-origin visit primitive;
- remain safe when any hot graph node is erased;
- isolate immutable reusable expansion payloads behind NodeStore;
- meet the performance and boundedness gates in Section 15.

### 3.2 Non-goals for v1

Do not implement these as part of the initial Lc5 change:

- replacing classic as the default search;
- deleting or refactoring LC3 or DAG-preview;
- training-data artifacts or self-play integration;
- Syzygy probing;
- sticky endgame bounds or mate-distance propagation;
- contempt, moves-left utility, draw-score asymmetry, Dirichlet noise, or smart
  pruning;
- the complete classic time-manager and stopper matrix;
- MultiPV;
- an actual disk or network NodeStore;
- automatic production eviction policy;
- exact preservation of mutable MCTS statistics across eviction;
- runtime batch-size autotuning;
- an external UCI command for arbitrary-origin visits;
- a hard Elo claim based only on the implementation benchmarks.

### 3.3 Definition of successful v1

V1 is complete only when:

- all functional, lifecycle, DAG, eviction, and concurrency tests in this
  document pass;
- sanitizer runs do not report races, use-after-free, or counter errors in the
  CPU/trivial configuration;
- long searches remain inside configured active-work bounds;
- GPU throughput is within 5% of tuned classic on the reference workload;
- CPU/trivial throughput matches or exceeds classic at an equal documented
  thread budget;
- an 8-worker Lc5 configuration materially outperforms 1 worker on a
  deliberately search-CPU-limited benchmark;
- fixed-node smoke games produce no illegal moves, hangs, protocol violations,
  or catastrophic deterministic search errors.

## 4. Terminology and state model

Use the following terms consistently in code and comments:

- NodeKey: 64-bit identity derived from PositionHistory.
- Generation: monotonically increasing identity of a particular hot graph
  incarnation of a NodeKey.
- Hot graph: mutable, in-memory, game-scoped DAG containing search statistics.
- ExpansionPayload: immutable terminal or NN evaluation result, legal moves,
  and policy priors.
- NodeStore: reusable payload store, separate from mutable MCTS statistics.
- Visit: one logical MCTS sample admitted at an origin and completed exactly
  once by backup or cancellation.
- VisitOrigin: node/history at which a visit starts, plus an optional backup
  prefix.
- PathStep: key/generation/move record owned by a Visit.
- Reservation: an in-flight increment on one selected edge.
- MaterializationTicket: ownership record for loading/evaluating one NodeKey.
- Evaluation owner: the one Visit that causes an unexpanded node to be
  evaluated and backs up that leaf evaluation.
- Waiter: another Visit that reaches the same pending node and suspends.
- Ready visit: a Visit that can continue selection.
- Ready evaluation: a store miss with legal moves and encoded history waiting
  for a backend batch.
- SearchRun: threads, queues, counters, and limits for one go command.
- GameGraph: hot graph and payload store retained across SearchRuns in one
  game.

## 5. Architectural overview

The major data flow is:

~~~text
                              +-----------------------+
                              |    Search controller  |
                              | limits/info/stop/PV   |
                              +-----------+-----------+
                                          |
                                  admit/recycle VisitId
                                          |
                                          v
 +----------------+     node step     +-----------------------+
 | Visit workers  | <---------------> | Sharded mutable DAG   |
 | selection and  |                   | 1024 short-lock shards|
 | backup         |                   +-----------------------+
 +---+---------+--+
     |         ^
     | miss /  | materialized node, owner result, resumed waiters
     | pending |
     v         |
 +-------------------------+       batched load       +------------------+
 | Materialization tickets | -----------------------> | NodeStore workers|
 | own owner + waiters      | <----------------------- | memory store v1  |
 +------------+------------+        hit / miss        +------------------+
              |
              | store miss, nonterminal
              v
 +-------------------------+        blocking batch    +------------------+
 | Ready evaluation queue  | -----------------------> | Backend evaluators|
 | surplus remains queued  | <----------------------- | 1+ computations  |
 +-------------------------+       evaluation result  +------------------+
~~~

There is no gather wave and no separate backprop stage:

- Visit workers advance individual Visits.
- A Visit that completes immediately backs itself up on a visit worker.
- A Visit that suspends is retained in the fixed-capacity VisitPool.
- Evaluation completion makes the owner ready to back up and the waiters ready
  to continue.

The control thread does not mutate graph nodes. It handles limits, stop state,
periodic output, and thread lifecycle. Therefore it is not a search-throughput
bottleneck.

## 6. Node identity and DAG semantics

### 6.1 Constructing PositionHistory

Do not use GameState::GetPositions for search identity because it constructs
Positions directly and does not populate repetition counts.

Build the root history exactly as follows:

1. PositionHistory::Reset(game_state.startpos).
2. For each game_state.moves entry, call PositionHistory::Append(move).
3. Use this history for terminal detection, NN input, root identity, PV
   reconstruction, and child histories.

Each Visit owns a PositionHistory for its current path. The first
implementation may copy histories when creating Visits; profile before adding
custom persistent-history storage.

### 6.2 NodeKey

Define:

~~~cpp
struct NodeKey {
  uint64_t hash;
  auto operator<=>(const NodeKey&) const = default;
};
~~~

Compute it using:

~~~text
history.HashLast(settings.history_key_length + 1)
~~~

The setting range is 0 through 7:

- 7 is the default and includes the current position plus all seven historical
  positions used by the NN encoder.
- 0 keys by the current Position hash plus rule-50 count, matching the existing
  low-history cache approximation and producing denser transpositions.
- Intermediate values make the strength/storage tradeoff explicit.

With a shortened history key, the first ExpansionPayload materialized for a key
is canonical for that game. Later histories with the same key reuse it. This is
an intentional approximation and must be visible in the option help.

### 6.3 Shared and non-shared state

One NodeKey maps to one hot NodeState and therefore shares:

- terminal/expanded lifecycle;
- legal moves and priors;
- node-level completed visit/value aggregates;
- outgoing edges and their statistics;
- all selection knowledge below the transposed position.

Each incoming parent edge is an outgoing EdgeState in a different parent node.
It therefore has its own:

- completed visit count;
- in-flight reservation count;
- value/draw/moves-left aggregates for visits that traversed that edge.

PUCT uses edge-local Q and N. This avoids locking child nodes while comparing
parent edges and avoids broadcasting every child update to all transposition
parents. Node-level aggregates remain shared for FPU, root output, arbitrary
origins, and diagnostics.

### 6.4 Graph lifetime

GameGraph persists across go commands when SetPosition uses the same starting
Position:

- re-root by computing the new history and NodeKey;
- retain all graph branches, not only the selected subtree;
- retain mutable statistics and payloads.

Clear the GameGraph when:

- NewGame is called;
- SetPosition supplies a different starting Position;
- SetBackend supplies a different backend instance;
- the engine object is destroyed.

Clearing on NewGame avoids stale payloads after backend/policy-temperature
configuration changes.

## 7. Exact data structures

Names may be adjusted only to match local style; ownership and fields must not
change without updating this plan.

### 7.1 Value aggregates

Use double for accumulated q and float or double for d/m sums. Do not update
running averages in a way that becomes order-sensitive under different worker
interleavings.

~~~cpp
struct ValueStats {
  uint64_t visits = 0;
  double q_sum = 0.0;
  double d_sum = 0.0;
  double m_sum = 0.0;

  float Q() const;
  float D() const;
  float M() const;
  void Add(float q, float d, float m);
};
~~~

Q/D/M return zero when visits is zero. Selection uses FPU instead of Q for an
unvisited edge.

### 7.2 Expansion payload

~~~cpp
enum class TerminalKind : uint8_t {
  kNonTerminal,
  kCheckmate,
  kStalemate,
  kRule50,
  kRepetition,
  kInsufficientMaterial,
};

struct ExpansionPayload {
  TerminalKind terminal = TerminalKind::kNonTerminal;
  float leaf_q = 0.0f;
  float leaf_d = 0.0f;
  float leaf_m = 0.0f;
  std::vector<Move> moves;
  std::vector<float> priors;
};
~~~

Invariants:

- moves.size equals priors.size;
- terminal payloads have empty moves/priors;
- nonterminal moves are sorted by descending prior, then by Move raw data for
  deterministic ties;
- priors are the post-softmax values returned through Backend;
- leaf q/d/m are from the current side-to-move perspective.

### 7.3 Mutable node and edge state

~~~cpp
enum class NodeLifecycle : uint8_t {
  kMaterializing,
  kExpanded,
};

struct EdgeState {
  Move move;
  float prior;
  uint64_t visits = 0;
  uint32_t in_flight = 0;
  double q_sum = 0.0;
  double d_sum = 0.0;
  double m_sum = 0.0;
};

struct NodeState {
  uint64_t generation;
  NodeLifecycle lifecycle;
  MaterializationTicketId ticket;
  TerminalKind terminal;
  ValueStats value;
  std::vector<EdgeState> edges;
  uint64_t last_access_epoch;
};
~~~

Do not store parent pointers, child pointers, waiter lists, Visit pointers, or
backend result pointers in NodeState.

The materialization registry—not NodeState—is authoritative for ticket
ownership and waiters. This allows a pending node to be evicted.

### 7.4 Graph shards

GameGraph contains exactly 1,024 power-of-two shards:

~~~cpp
struct GraphShard {
  std::mutex mutex;
  absl::flat_hash_map<NodeKey, NodeState> nodes;
};
~~~

Select the shard with the same multiplicative hash approach used elsewhere in
the repository, using the high ten bits.

All APIs operate by key and execute a bounded callback or operation while the
shard lock is held. Never return NodeState pointers or references.

Required graph operations:

- FindOrCreateMaterializing(key, ticket) -> generation and whether created.
- InstallPayload(key, ticket, payload) -> current generation.
- SelectAndReserve(key, expected generation, policy inputs) -> selected move
  and current generation/status.
- UpdateNodeValue(key, expected generation, value delta) -> applied/skipped.
- CompleteEdge(key, expected generation, move, value) -> decrements in-flight
  and commits one edge visit.
- CancelEdge(key, expected generation, move) -> decrements in-flight only.
- SnapshotNode(key) -> immutable copy used outside the lock.
- Erase(key) -> whether erased.
- Clear().
- Size() and shard-level metrics.

No graph operation may lock a second graph shard or call NodeStore/backend code.

### 7.5 Generation allocation

GameGraph owns an atomic uint64 generation counter:

- increment for every newly inserted NodeState;
- never reuse a generation during one GameGraph lifetime;
- generation zero is invalid;
- overflow may assert because it is practically unreachable.

InstallPayload may recreate a missing node and therefore return a different
generation from the one held by suspended Visits.

### 7.6 Visit and path state

~~~cpp
struct PathStep {
  NodeKey key;
  uint64_t generation;
  std::optional<Move> selected_move;
};

struct VisitOrigin {
  NodeKey key;
  PositionHistory history;
  std::vector<PathStep> backup_prefix;
};

enum class VisitState : uint8_t {
  kFree,
  kReadySelect,
  kWaitingMaterialization,
  kReadyBackup,
  kBackingUp,
  kCancelling,
};

struct Visit {
  VisitId id;
  VisitState state;
  VisitOrigin origin;
  PositionHistory history;
  NodeKey current_key;
  std::vector<PathStep> path;
  MaterializationTicketId waiting_ticket;
  float result_q;
  float result_d;
  float result_m;
};
~~~

VisitPool allocates a fixed number of Visit slots at SearchRun construction.
Queues carry VisitId, never Visit pointers. Slot reuse increments a slot epoch
or embeds an epoch in VisitId so a late message cannot target a reused slot.

The active-work bound is therefore hard:

~~~text
number of non-free Visit slots <= MaxActiveVisits
~~~

Every queued, waiting, evaluating, backing-up, or cancelling Visit occupies one
of these slots.

### 7.7 Materialization ticket

~~~cpp
enum class TicketState : uint8_t {
  kLoadingStore,
  kWaitingForEval,
  kEvaluating,
  kCompleting,
};

struct MaterializationTicket {
  MaterializationTicketId id;
  NodeKey key;
  TicketState state;
  VisitId owner;
  std::vector<VisitId> waiters;
  PositionHistory owner_history;
  std::chrono::steady_clock::time_point created_at;
};
~~~

Use a separately sharded MaterializationRegistry keyed by NodeKey. Registry
locks must never nest with graph locks.

The ticket owns owner/waiter relationships even when the hot NodeState is
erased. Exactly one ticket may exist for a NodeKey.

## 8. Search algorithm

### 8.1 Admission

SearchRun initially admits up to MaxActiveVisits logical Visits from the root
origin, constrained by the node limit:

~~~text
admitted visits <= requested new visits
~~~

For infinite/movetime searches, a completed Visit slot is recycled into a new
root Visit while the run is accepting work.

For go nodes N:

- count N as new completed logical visits for this SearchRun, independent of
  visits retained in GameGraph;
- never admit more than N;
- do not count a suspended, in-flight, cancelled, cache probe, or NN request as
  a completed visit;
- after all N admitted visits complete, automatically produce bestmove and
  stop.

### 8.2 Entering a node

A Visit worker repeatedly advances one Visit:

1. Check the SearchRun stop token.
2. Ensure the Visit has a PathStep for current_key.
3. Inspect the current graph entry under its shard lock.
4. Handle one of:
   - missing node;
   - materializing node;
   - expanded terminal node;
   - expanded nonterminal node.
5. Release the shard lock before queueing or registry operations.

Missing:

- atomically create or join a MaterializationTicket;
- create a materializing NodeState if still absent;
- update the Visit's current PathStep generation;
- owner enters kWaitingMaterialization and submits a store load;
- joiners enter kWaitingMaterialization and append to ticket.waiters.

Materializing:

- join the authoritative ticket;
- do not create an independent collision or rollback event;
- suspend the Visit.

Expanded terminal:

- use the exact terminal q/d/m;
- transition the Visit to kReadyBackup.

Expanded nonterminal:

- compute PUCT;
- increment the chosen edge.in_flight;
- record the move in the current PathStep;
- append the move to Visit.history;
- compute child NodeKey;
- set Visit.current_key to the child;
- continue selection.

### 8.3 PUCT and FPU formulas

Match classic's core formula where applicable, without moves-left utility,
draw-score asymmetry, root-special parameters, smart pruning, or bounds.

For a parent node:

~~~text
started(edge) = edge.visits + edge.in_flight
children_started = sum(started(edge))
cpuct = CPuct + CPuctFactor * log((node.value.visits + CPuctBase) / CPuctBase)
u(edge) = cpuct * prior(edge) * sqrt(max(children_started, 1))
          / (1 + started(edge))
~~~

For a visited edge:

~~~text
q(edge) = edge.q_sum / edge.visits
~~~

For an unvisited edge:

~~~text
visited_policy = sum(prior(edge) for edges where started(edge) > 0)
parent_q = node.value.Q() when node.value.visits > 0, otherwise 0

if FpuStrategy == absolute:
    q(edge) = FpuValue
else:
    q(edge) = parent_q - FpuValue * sqrt(visited_policy)
~~~

Score:

~~~text
score(edge) = q(edge) + u(edge)
~~~

Tie-break in this order:

1. higher score;
2. higher prior;
3. lower Move raw representation.

PUCT is computed for exactly one logical Visit at a time. The shard lock
serializes choices at the same node, so each subsequent worker sees the prior
worker's in-flight reservation.

### 8.4 Terminal detection

On NodeStore miss, before NN evaluation:

1. call PositionHistory::ComputeGameResult;
2. distinguish checkmate, stalemate, repetition, rule-50, and insufficient
   material for metrics where possible;
3. construct terminal q/d/m from the current side-to-move perspective;
4. install/store a terminal ExpansionPayload without invoking Backend.

Terminal values:

- draw: q=0, d=1, m=0;
- current side wins: q=1, d=0, m=0;
- current side loses: q=-1, d=0, m=0.

Every waiter on an exact terminal represents a valid logical visit. Therefore
the owner and all waiters transition to backup with the terminal value. They do
not resume below the node.

### 8.5 Materialization and suspension

The first Visit to an unmaterialized node becomes the evaluation owner.

For a nonterminal payload:

- only the owner backs up the leaf q/d/m;
- waiters refresh their leaf PathStep with the installed generation, clear its
  selected_move, and return to kReadySelect at the expanded node;
- resumed waiters select below the node using the newly available policy.

This rule prevents one NN result from being counted as multiple independent
leaf evaluations while also preventing repeated collision rollback/retry
churn.

If the owner is cancelled because the whole SearchRun stops:

- no new visit is promoted solely to preserve an initial count;
- the payload may still be stored for a future SearchRun;
- all surviving waiters are also being cancelled during run shutdown.

### 8.6 Backup

Backup starts with leaf q/d/m in leaf side-to-move perspective.

For each PathStep from leaf toward origin:

1. Update that node's node-level ValueStats if key and generation still match.
2. If the previous PathStep selected an edge into this node:
   - flip q for the parent perspective;
   - increment m by one;
   - under the parent shard lock, verify parent generation;
   - find the edge by Move;
   - assert in debug that in_flight is nonzero;
   - decrement in_flight;
   - increment edge visits and value sums.
3. Continue to the previous node even if the current node or parent generation
   was missing/stale.

Draw probability is not negated. Q is negated once per ply. Moves-left is
incremented once per ply.

Do not combine multiple paths into a heap as LC3 does. Independent workers can
batch local update calls later only if profiling shows graph locks dominate and
the batching preserves the same generation checks.

### 8.7 Cancellation

Cancellation traverses every PathStep that has selected_move:

- if parent key/generation still match, decrement edge.in_flight;
- do not change visits or value sums;
- if the node is missing or has a different generation, skip it;
- continue through the entire path;
- release the Visit slot exactly once.

In debug builds, underflow is a fatal invariant violation. In release builds,
record an invariant metric and avoid wrapping the counter.

### 8.8 Arbitrary-origin visits

The internal admission API accepts VisitOrigin rather than reading a root
global:

- root UCI searches pass current root key/history and an empty backup prefix;
- a non-root caller may pass any known key/history;
- backup normally stops at that origin;
- an optional backup_prefix allows a caller with a valid key/generation path to
  propagate results above the origin;
- eviction generation rules apply to the prefix exactly as to the new path.

Add direct tests for:

- non-root visit updates only the origin/subtree;
- non-root visit with prefix also updates surviving ancestors;
- origin eviction causes lazy rematerialization and safe completion.

Do not expose this through UCI in v1.

## 9. NodeStore and future tiered storage

### 9.1 Interface

NodeStore persists ExpansionPayload only. Mutable visits, reservations, and
ValueStats remain in GameGraph.

Define a blocking batched primitive:

~~~cpp
class NodeStore {
 public:
  virtual ~NodeStore() = default;

  virtual std::vector<std::optional<ExpansionPayload>> LoadBatch(
      std::span<const NodeKey> keys) = 0;

  virtual void StoreBatch(
      std::span<const StoredExpansion> entries) = 0;

  virtual void Clear() = 0;
};
~~~

The blocking contract is deliberate: a future disk/network implementation may
block internally. Search correctness must therefore never call it while
holding a graph/registry lock or on a visit worker.

### 9.2 Dispatcher

SearchRun owns NodeStore dispatcher workers:

- request queue carries ticket ID and key;
- dispatcher collects up to 256 requests or flushes when selection is
  dependency-starved;
- one LoadBatch returns hits/misses in input order;
- hits become materialization completions;
- misses are checked for terminal state and then become ready evaluations;
- StoreBatch is best-effort asynchronous and does not delay waking Visits;
- default dispatcher count equals automatic/configured evaluator count;
- dispatcher queues are bounded indirectly by the VisitPool/ticket count.

Do not let multiple dispatchers load the same key: ticket deduplication happens
before queue submission.

### 9.3 Memory implementation

MemoryNodeStore uses a sharded flat hash map and copies ExpansionPayload on
load. This copy is acceptable for v1 and gives the tiering boundary honest
value semantics.

Profile payload-copy cost. A future immutable shared blob/reference optimization
may be added behind NodeStore, but must not leak stable payload pointers into
GameGraph paths.

### 9.4 Eviction semantics

GameGraph exposes Erase(NodeKey) for tests and future eviction policy.

On eviction:

- mutable NodeState and all its reservations/statistics disappear;
- ExpansionPayload remains in NodeStore unless separately removed;
- a future access creates a new generation and materializes from NodeStore;
- old path updates skip the stale generation;
- reservations in surviving ancestor nodes are still completed/cancelled;
- ticket owner/waiters remain valid because they live in the registry.

V1 does not promise identical search results with and without eviction. It
promises safety, bounded counters, and forward progress.

No automatic eviction policy is enabled in v1. Do not add LRU complexity before
the forced-eviction invariants and performance baselines exist.

## 10. Neural evaluation and batch policy

### 10.1 Evaluator count

Settings:

- EvalThreads=0 means automatic.
- Automatic evaluator count is clamp(backend.suggested_num_search_threads,
  1, 128).
- An explicit value creates that many concurrent Backend computations.
- The current CUDA backend recommends one; multi-device/demux backends may
  recommend more.

### 10.2 Batch target

MinibatchSize=0 uses:

~~~text
min(backend.recommended_batch_size, backend.maximum_batch_size)
~~~

An explicit value is clamped/validated against maximum_batch_size and must
produce a clear option error rather than backend memory corruption.

The target controls only how many NN inputs an evaluator consumes. It does not
limit ready evaluation queue length.

### 10.3 Filling a computation

For one evaluator iteration:

1. Wait until one of:
   - ready NN requests >= target;
   - all runnable Visits are dependency-blocked and at least one request is
     ready;
   - the oldest ready request has waited MaxBatchDelayMs;
   - drain/stop requires flushing.
2. Pop no more than target requests.
3. Create one BackendComputation.
4. Add inputs one by one.
5. If AddInput returns FETCHED_IMMEDIATELY:
   - complete that ticket immediately;
   - continue consuming ready requests until NN UsedBatchSize reaches target
     or no more requests are eligible.
6. Call ComputeBlocking only when UsedBatchSize is nonzero.
7. Publish every result to ticket completion.
8. Record requested count, actual NN batch size, cache hits, compute duration,
   and total latency.

Surplus requests stay queued for the next computation. No rollback, dummy
input, or discard is allowed.

### 10.4 Starvation detection

Maintain atomics for:

- ready visit queue count;
- visit workers currently advancing a Visit;
- Visits waiting on materialization;
- ready NN request count;
- evaluator computations in progress.

Selection is dependency-starved when:

~~~text
ready_visit_count == 0
and workers_advancing_visits == 0
and ready_nn_request_count > 0
~~~

This condition requests an immediate partial flush. It is a hint, not a
one-time latch; evaluators must recheck under their queue mutex/CV.

### 10.5 Batch delay

MaxBatchDelayMs:

- default: 2 ms;
- range: 0 through 100 ms;
- zero disables the time-based flush but not dependency-starvation or drain
  flushes.

This is not adaptive batch tuning. Benchmark users must still set batch size
explicitly when the backend recommendation is poor.

## 11. Parallelism and synchronization

### 11.1 Visit workers

Threads controls visit worker count:

- range: 0 through 512;
- explicit value is used exactly;
- zero automatic:
  - CPU backend: max(1, backend.suggested_num_search_threads);
  - non-CPU backend:
    min(hardware_concurrency,
        max(2, 4 * automatic_or_explicit_evaluator_count)).

If hardware_concurrency returns zero, treat it as one.

The 512 upper limit intentionally supports tournament machines with 256
physical cores and SMT.

### 11.2 Lock ordering

Enforce these rules:

1. Never hold two graph shard locks simultaneously.
2. Never hold graph and materialization-registry locks simultaneously.
3. Never hold any search lock while calling Backend or NodeStore.
4. Queue operations happen after releasing graph/registry locks.
5. Visit slot state transitions use a per-slot mutex or atomic state/epoch; do
   not infer validity from queue membership.

Add debug lock-order annotations where supported.

### 11.3 Hot-root contention

V1 intentionally keeps exact per-visit root locking. Do not prematurely
reintroduce bulk root distribution.

Metrics must measure:

- selection operations per worker;
- cumulative graph-shard lock wait time sampled at a configurable low rate;
- maximum operations on one shard;
- worker busy/idle time.

If the scaling benchmark shows root lock saturation, the next optimization
should be an exact reservation block allocated under one root lock and consumed
as individually represented Visits. It must not approximate multiple PUCT
choices with one score calculation. That optimization is not part of v1.

### 11.4 Queues

Use the existing moodycamel concurrent queue implementation unless profiling
or correctness tests show a problem.

The queues themselves need not enforce capacity because all messages referring
to search work are derived from fixed Visit slots or deduplicated tickets, and
there can be at most MaxActiveVisits tickets/Visits.

Queue counters must be updated atomically with enqueue/dequeue wrappers so
starvation and metrics do not rely on approximate third-party queue size APIs.

## 12. Search lifecycle and UCI behavior

### 12.1 Engine ownership

Lc5Engine owns:

- current backend pointer;
- OptionsDict pointer;
- current GameState and reconstructed root PositionHistory;
- GameGraph;
- MemoryNodeStore;
- optional active SearchRun;
- game-start Position used to detect incompatible SetPosition calls.

SearchRun owns all per-go threads, queues, VisitPool, ticket registry, counters,
limits, and output-once state.

### 12.2 SetPosition and NewGame

Before changing position:

- AbortSearch;
- WaitSearch.

SetPosition:

- if starting Position changed, clear GameGraph and NodeStore;
- reconstruct repetition-aware PositionHistory;
- compute and store root NodeKey;
- retain same-game graph branches.

NewGame:

- abort/wait current search;
- clear graph and payload store;
- clear stored start position and root history;
- reset game metrics.

### 12.3 StartClock

StartClock stores steady_clock::now in Lc5Engine.

StartSearch requires a clock value. If the controller did not call StartClock,
use StartSearch entry time and record a warning metric rather than dereferencing
an empty optional.

### 12.4 Supported GoParams

Supported:

- nodes;
- movetime;
- infinite;
- ponder, treated as infinite until controller stop/restart;
- no-limit go, treated as infinite.

If nodes and movetime are both supplied, stop on whichever limit is reached
first.

Not supported in v1:

- wtime/btime/winc/binc/movestogo;
- depth;
- mate;
- searchmoves.

When unsupported fields are present:

- output one ThinkingInfo comment naming every ignored field;
- do not crash;
- if there is no supported limit, continue as infinite until stop.

### 12.5 Stop and abort

SearchBase requires StopSearch and AbortSearch not to block.

StopSearch:

- atomically upgrade stop mode from running to respond-bestmove;
- wake controller, Visit workers, store dispatchers, and evaluators;
- return immediately.

AbortSearch:

- atomically set abort unless bestmove was already committed;
- wake all workers;
- return immediately;
- never emit bestmove solely because of abort.

The control thread:

1. stops new Visit admission;
2. allows workers to cancel ready/waiting Visits;
3. lets an already running Backend computation finish because the API is
   blocking;
4. ignores graph-count application for results whose Visit was cancelled, but
   may store their immutable payload;
5. emits bestmove exactly once for graceful/automatic stop;
6. marks the run finished and wakes WaitSearch.

WaitSearch joins every run thread and may block.

Repeated stop/abort calls are idempotent.

### 12.6 Best move and PV

Bestmove uses committed edge visits only; in-flight reservations never decide
the move.

At each node:

1. choose greatest edge.visits;
2. tie-break by greatest edge Q;
3. then greatest prior;
4. then lowest Move raw representation.

Build PV by snapshotting one node at a time and applying the selected raw move
to a local PositionHistory. Stop on:

- missing/materializing/terminal node;
- no visited edge;
- 256 plies;
- repeated NodeKey in the PV builder.

Convert each stored side-to-move Move to white-perspective UCI form by flipping
the output copy whenever the local current Position is black to move. Apply the
unflipped stored Move to PositionHistory.

If stopped before root expansion:

- generate legal root moves synchronously;
- choose the deterministic lowest raw Move;
- return a null move only if the position is terminal.

Ponder move is the second PV move when present.

### 12.7 Thinking output

Emit at most once per second and once at final bestmove.

Populate:

- time from StartClock;
- completed logical visits in the current run;
- NPS from completed visits;
- actual backend evaluations and EPS;
- maximum completed depth;
- maximum selected depth;
- root WDL from node ValueStats when available;
- PV;
- hashfull as hot-node capacity fraction only after a capacity exists;
- comment containing compact Lc5 pipeline metrics.

Do not report spawned/admitted Visits as nodes.

## 13. Settings and option contract

Create an Lc5-specific Settings class. Do not call classic::SearchParams::Populate
because that would expose unsupported options.

Required options:

| Long/UCI name | Range | Default | Meaning |
|---|---:|---:|---|
| threads / Threads | 0..512 | 0 | Visit workers; 0 uses Section 11.1 auto rule |
| eval-threads / EvalThreads | 0..128 | 0 | Concurrent backend evaluators |
| minibatch-size / MinibatchSize | 0..backend max | 0 | NN target; 0 uses backend recommendation |
| max-active-visits / MaxActiveVisits | 0..1000000 | 0 | Fixed VisitPool size |
| history-key-length / HistoryKeyLength | 0..7 | 7 | Historical positions included beyond current |
| max-batch-delay-ms / MaxBatchDelayMs | 0..100 | 2 | Oldest-ready partial flush limit |
| cpuct / CPuct | 0..100 | 1.745 | PUCT initial constant |
| cpuct-base / CPuctBase | 1..1e9 | 38739 | PUCT growth base |
| cpuct-factor / CPuctFactor | 0..1000 | 3.894 | PUCT growth multiplier |
| fpu-strategy / FpuStrategy | absolute,reduction | reduction | Unvisited-edge Q rule |
| fpu-value / FpuValue | -100..100 | 0.33 | Absolute value or reduction |

MaxActiveVisits=0 resolves after evaluator count and batch target:

~~~text
max(1024, 4 * evaluator_count * batch_target)
~~~

Reject a resolved value smaller than evaluator_count because it cannot keep one
request per evaluator.

Settings must be immutable for one SearchRun. Option changes take effect on the
next run.

## 14. Metrics

Use atomics for inexpensive counters and merge per-worker local counters at
worker exit or output ticks.

Required counters/histograms:

### Visit lifecycle

- visits_admitted;
- visits_completed;
- visits_cancelled;
- visits_ready_high_water;
- active_visits_high_water;
- visits_suspended;
- visits_resumed;
- terminal_visits;
- maximum_depth;
- selection_node_steps;
- backup_node_steps;
- stale_generation_node_updates;
- stale_generation_edge_updates;
- invariant_underflow_prevented.

### Materialization/storage

- tickets_created;
- ticket_waiters;
- maximum_waiters_per_ticket;
- node_store_load_batches;
- node_store_load_keys;
- node_store_hits;
- node_store_misses;
- node_store_store_batches;
- graph_nodes_created;
- graph_nodes_rehydrated;
- graph_nodes_evicted;
- graph_size_high_water.

### Neural evaluation

- eval_requests;
- cache_hits;
- NN evaluations;
- evaluation_batches;
- partial_starvation_flushes;
- partial_timeout_flushes;
- partial_drain_flushes;
- ready_eval_high_water;
- batch-size histogram;
- compute-duration histogram;
- request-to-completion latency histogram.

### Concurrency

- per-worker visits/node steps;
- worker busy and idle durations;
- sampled shard-lock wait duration;
- maximum observed work on one shard.

Metrics must make it possible to determine whether low EPS comes from:

- selection starvation;
- store latency;
- small batches;
- backend compute;
- lock contention;
- excessive suspension on a narrow frontier.

## 15. Testing and verification

### 15.1 Unit tests: keying and graph

Add tests for:

- default HistoryKeyLength distinguishes two move orders reaching the same
  current board with different recent histories;
- HistoryKeyLength=0 merges those histories;
- rule-50 and repetition state affect keys as PositionHistory::HashLast
  specifies;
- one transposed key produces one NodeState;
- parents retain distinct EdgeState visit counts;
- graph operations never return persistent references;
- erase/recreate changes generation;
- stale generation operations report skipped and do not mutate;
- graph clear resets size and generation-visible entries.

Use a simple transposition such as exchanging the order of Nf3 and g3 with
matching black replies, and verify the resulting board before asserting key
behavior.

### 15.2 Unit tests: policy

Use synthetic NodeSnapshots to verify:

- CPuct growth formula;
- absolute and reduction FPU;
- in-flight reservations affect denominator before completion;
- completed edge Q is used instead of FPU;
- deterministic tie-breaking;
- visited-policy calculation;
- one worker's reservation changes the next worker's choice;
- terminal nodes never invoke selection.

Create a tiny sequential reference selector in test code and compare a fixed
series of selections/reservations/completions.

### 15.3 Unit tests: backup and cancellation

Verify:

- q flips every ply;
- d is unchanged;
- m increments every ply;
- node and edge visits increment exactly once;
- cancellation releases reservations without committing;
- missing leaf, middle node, parent, and root are skipped safely;
- stale generation at one step does not prevent updates above it;
- no counter wraps in release-style checked logic;
- arbitrary-origin backup stops at origin;
- arbitrary-origin prefix updates matching ancestors only.

### 15.4 Materialization tests

Use a deterministic fake NodeStore and fake Backend:

- one owner/store miss/NN result;
- one owner/store hit without NN;
- multiple waiters on nonterminal result: owner backs up, waiters resume;
- multiple waiters on terminal result: all back up terminal;
- node erased while store load is pending;
- node erased while NN evaluation is pending;
- ticket remains authoritative after node erase;
- duplicate requests for one key cause one store load and one NN eval;
- payload stored even when run stops during blocking evaluation;
- cancelled VisitId epoch prevents a late result from touching a reused slot.

### 15.5 Batch tests

With target two and five ready unique requests:

- computations consume 2, 2, and 1;
- the fifth request is flushed only by starvation/timeout/drain;
- no request is dropped or evaluated twice.

Also verify:

- startup root causes an immediate starvation batch of one;
- surplus accumulated during a blocking computation feeds the next batch;
- immediate cache hits are replaced in the same collection loop;
- evaluator count greater than one does not assign a ticket twice;
- MaxBatchDelayMs=0 disables timeout flush;
- stop drains or discards according to ownership without deadlock.

### 15.6 Search lifecycle tests

Add an Lc5-specific fake responder/backend integration fixture:

- StartSearch returns promptly;
- go nodes completes exactly the requested new logical visits;
- movetime stops within a reasonable scheduling/backend tolerance;
- infinite runs until stop;
- stop is nonblocking and emits exactly one bestmove;
- abort is nonblocking and emits no bestmove;
- stop/stop, abort/abort, and stop/abort are idempotent;
- go/stop/wait/go works repeatedly;
- SetPosition aborts and joins a prior run;
- same-game SetPosition retains graph state;
- NewGame and incompatible start clear it;
- immediate stop before root evaluation still emits a legal fallback;
- unsupported GoParams produce one warning and do not crash.

### 15.7 Concurrency and eviction stress

Run randomized tests with 1, 2, 8, and 32 visit workers:

- deterministic fake evaluations with random completion order;
- repeated transposition access;
- random Erase calls including current root and pending nodes;
- random stop timing;
- small VisitPool to force slot reuse;
- invariant audit after shutdown:
  - all Visit slots free;
  - no ticket remains;
  - no edge in_flight remains for matching surviving generations;
  - every completed Visit was counted once;
  - no queue counter is nonzero;
  - bestmove count is zero or one according to stop mode.

Provide a test-only GameGraph invariant walker that locks one shard at a time.

### 15.8 Sanitizers

Run CPU/trivial tests under:

- AddressSanitizer;
- UndefinedBehaviorSanitizer;
- ThreadSanitizer in a build without CUDA/backend components that conflict with
  TSan.

Do not waive a TSan report as harmless without documenting the exact
happens-before relationship and adding the corresponding annotation/fix.

### 15.9 Build verification

Required:

~~~text
meson setup or configure with -Dlc5=true -Dgtest=true
ninja -C build/<config>
meson test -C build/<config> --print-errorlogs
~~~

Also build with:

- lc5 disabled;
- lc3 disabled and lc5 enabled;
- release assertions disabled;
- debug assertions enabled.

## 16. Performance benchmark plan

### 16.1 Reproducible harness

Add a search comparison script or executable that:

- launches classic and lc5 through UCI;
- uses the same weights, backend, NN cache setting, FEN, duration/node limit,
  batch target, and total documented thread budget;
- performs a warm-up search before measurement;
- captures final and periodic ThinkingInfo;
- emits JSON or CSV containing all metrics needed by Section 2.3;
- records engine version, command line, CPU/GPU model, driver, and network hash.

Do not make hardware-dependent thresholds part of normal unit-test CI.

### 16.2 Reference GPU gate

Reference:

- RTX 5090;
- BT4-1024x15x32h-swa-6147500.pb.gz;
- cuda-auto;
- NNCacheSize=0;
- start position plus at least two representative middlegame FENs;
- MinibatchSize=128 for tuned comparison;
- at least 30 seconds after warm-up;
- three repetitions, median result.

Gate:

- Lc5 median EPS and completed-visit NPS must each be at least 95% of tuned
  classic where the metrics are comparable;
- no unbounded queue growth;
- batch histogram must explain any remaining gap;
- report CPU utilization and exact worker counts.

Also report MinibatchSize=0 to expose backend recommendation quality, but do not
confuse that with the tuned-search gate.

### 16.3 CPU/search-overhead gate

Use trivial or deterministic near-zero-latency backend:

- match total search-related thread budget between classic and Lc5;
- disable NN cache;
- run fixed duration after warm-up;
- compare completed visits, not spawned work.

Gate:

- Lc5 must match or exceed classic median NPS;
- Lc5 with eight visit workers must materially outperform Lc5 with one worker.
  Use 1.25x as the minimum initial scaling gate on the 16-core reference CPU;
- report lock-wait and per-worker balance.

If the 1.25x gate fails, profile before changing the architecture. Acceptable
first investigations are root-shard contention, queue contention, history
copying, move generation, and NodeStore dispatch. Do not mask the result by
counting reservations as visits.

### 16.4 Batch-overgather demonstration

Create a deterministic slow backend benchmark where:

- target batch is small enough to observe multiple queued batches;
- selection can prepare more than one target while evaluation blocks;
- ready_eval_high_water exceeds target;
- every request retains an owner and later completes;
- the backend stays busy after startup;
- memory remains bounded by MaxActiveVisits.

This benchmark is the direct acceptance test for the original
gather-exactly-batch limitation.

### 16.5 Strength smoke testing

Run fixed-node fastchess games against classic:

- same network/backend;
- alternate colors/openings;
- equal completed-visit limit;
- enough games to expose illegal moves, deterministic pathologies, or gross
  strength failure;
- store PGNs and command lines.

V1 does not require a statistically conclusive Elo result. Any large or
repeatable loss must be investigated before calling Lc5 successful, especially
for history-key settings, PUCT formulas, value perspective, or graph reuse.

## 17. Build and source layout

Add the Meson option:

~~~text
option('lc5', type: 'boolean', value: true,
       description: 'Build Lc5 search algorithm')
~~~

Recommended source organization:

~~~text
src/search/lc5/
  engine.h / engine.cc
      SearchBase adapter, factory registration, game/run ownership
  settings.h / settings.cc
      Lc5-only options and resolved backend-dependent settings
  key.h
      NodeKey and hashing
  value.h
      ValueStats and perspective helpers
  graph.h / graph.cc
      sharded GameGraph and generation-checked operations
  node_store.h
      ExpansionPayload and NodeStore interface
  memory_node_store.h / memory_node_store.cc
      initial payload store
  visit.h
      VisitId, VisitOrigin, PathStep, Visit state
  materialization.h / materialization.cc
      ticket registry and store dispatch
  evaluator.h / evaluator.cc
      ready queue, batching, BackendComputation flow
  policy.h / policy.cc
      PUCT/FPU and deterministic edge selection
  search.h / search.cc
      SearchRun, admission, workers, lifecycle, backup/cancel
  metrics.h / metrics.cc
      counters, histograms, snapshots, UCI comment formatting
  *_test.cc
      focused tests grouped by subsystem
~~~

This layout is guidance, but do not collapse graph, tickets, evaluator, and
engine lifecycle into one monolithic search.cc. Conversely, do not introduce an
abstract interface for every class; the only required pluggable boundary in v1
is NodeStore.

Update meson.build:

- add Lc5 production sources only when get_option('lc5');
- add Lc5 tests only when lc5 and gtest are enabled;
- keep classic stopper sources unchanged;
- do not alter default_search.

Add a concise Lc5 design/benchmark document after implementation, derived from
this plan and updated with actual final measurements. This implementation plan
remains the task specification.

## 18. Ordered implementation sequence

Implement in this order. Each stage must compile and have its listed tests
before proceeding.

### Stage 1: foundational types and settings

Implement:

- NodeKey;
- ValueStats/value perspective helpers;
- ExpansionPayload;
- Settings parsing and backend-dependent resolution.

Tests:

- option defaults/overrides/ranges;
- key history behavior;
- value flips and aggregate arithmetic.

Exit criterion:

- Lc5 can be enabled in Meson without factory registration;
- focused tests pass.

### Stage 2: sharded graph and eviction primitives

Implement:

- 1,024-shard GameGraph;
- generations;
- materializing/expanded nodes;
- reserve/select/update/cancel/snapshot/erase operations;
- invariant walker.

Tests:

- graph/DAG/generation/policy/backup/cancel tests with no threads.

Exit criterion:

- a deterministic single-thread synthetic Visit can traverse and back up a
  graph entirely through key-based operations;
- forced erase never uses a retained pointer.

### Stage 3: VisitPool and single-worker SearchRun core

Implement:

- VisitId epochs;
- fixed VisitPool;
- ready queue wrapper/counters;
- one Visit worker;
- root admission;
- terminal backup and cancellation;
- internal arbitrary origin.

Use a fake synchronous materialization callback temporarily, not Backend.

Tests:

- admission bounds;
- arbitrary origin;
- stop/cancel;
- slot reuse and late-message rejection.

Exit criterion:

- completed/cancelled accounting balances exactly;
- one-worker deterministic result matches reference policy.

### Stage 4: materialization registry and MemoryNodeStore

Implement:

- ticket deduplication;
- owner/waiter lifecycle;
- dispatcher workers and batch loads/stores;
- terminal detection;
- graph installation and generation refresh.

Tests:

- store hit/miss;
- nonterminal suspend/resume;
- terminal all-waiter completion;
- eviction during load;
- ticket cleanup.

Exit criterion:

- all search work remains bounded while store completion is delayed;
- no duplicate key load occurs.

### Stage 5: Backend evaluator and overgather

Implement:

- ready evaluation queue;
- evaluator workers;
- exact target collection;
- immediate cache-hit replacement;
- starvation/timeout/drain flush;
- result publication and async payload store.

Tests:

- deterministic batch sequences;
- surplus requests across batches;
- startup partial batch;
- multiple evaluators;
- stop during ComputeBlocking.

Exit criterion:

- ready queue may exceed target and all requests complete exactly once;
- no fixed gather wave exists anywhere in Lc5.

### Stage 6: parallel visit workers

Enable configured workers over the same graph/VisitPool:

- ensure PathStep/generation rules are unchanged;
- add worker-local metrics;
- audit lock ordering;
- add concurrency stress and randomized completion.

Exit criterion:

- 1/2/8/32-worker stress tests pass;
- TSan-clean CPU configuration;
- eight workers pass the scaling gate or a profile identifies a concrete
  follow-up optimization.

Do not postpone this stage from v1 merely because one worker feeds the current
5090.

### Stage 7: UCI engine and whole-game reuse

Implement:

- Lc5Engine and factory;
- StartClock/SetPosition/NewGame;
- SearchRun start/stop/abort/wait;
- supported GoParams;
- thinking/bestmove/PV;
- same-game re-rooting.

Tests:

- engine lifecycle and protocol scenarios;
- repeated searches and graph reuse;
- unsupported option warnings.

Exit criterion:

- build/release/lc0 lc5 is usable for go nodes, go movetime, infinite+stop,
  and ponder+stop;
- calls obey SearchBase blocking contracts.

### Stage 8: metrics, benchmarking, and tuning

Implement:

- full metrics set;
- benchmark harness;
- JSON/CSV output;
- reference workload scripts/configuration.

Run:

- unit/integration/sanitizer suites;
- GPU and trivial performance gates;
- forced-eviction long stress;
- fixed-node fastchess smoke games.

Only after these results:

- tune documented worker and batch configurations;
- optimize measured hot spots without weakening ownership/generation
  invariants;
- update the Lc5 design note with final evidence.

## 19. Expected risks and prescribed responses

### Risk: root shard lock limits scaling

Response:

- verify with sampled lock-wait metrics and profiles;
- first reduce work inside the lock and avoid copying snapshots;
- if still limiting, design exact preallocated reservation blocks where every
  reservation remains an individual Visit;
- do not return to LC3-style approximate visit distribution.

### Risk: PositionHistory copying dominates

Response:

- profile allocation and copy time;
- add small-vector or immutable parent-linked history owned by VisitPool;
- preserve the ability to provide contiguous Position spans to Backend;
- do not store history in NodeState because DAG arrivals may differ when the
  history-key setting is shortened.

### Risk: NodeStore dispatcher starves a fast backend

Response:

- inspect store batch sizes and queue latency;
- increase dispatcher count;
- specialize MemoryNodeStore with an immediate nonblocking fast path while
  retaining the same value-based result;
- keep remote/disk implementations off Visit workers.

### Risk: suspended Visits consume all capacity on a narrow frontier

Response:

- dependency-starvation flush must evaluate the frontier immediately;
- owner completion wakes waiters to continue below it;
- metrics expose maximum waiters/ticket and active state distribution;
- do not launch work beyond MaxActiveVisits.

### Risk: eviction repeatedly destroys hot progress

Response:

- v1 has no automatic eviction, so this occurs only in tests/future policies;
- future eviction policy should prefer cold generations but may still erase
  active nodes because correctness cannot depend on pinning;
- payload retention avoids repeated NN cost;
- behavioral quality under aggressive eviction is a later policy concern.

### Risk: parallel scheduling changes deterministic results

Response:

- one-worker mode is the deterministic reference;
- multiworker ordering is allowed to change visit order;
- invariants, legal output, accounting, and statistical strength matter in
  parallel mode;
- do not require bit-identical multiworker results.

### Risk: GPU gate is met by using excessive CPU

Response:

- benchmark reports total visit/eval/store/control threads and CPU utilization;
- compare equal thread budgets separately;
- preserve both throughput and efficiency tables.

### Risk: Lc5 grows into another classic-sized feature matrix

Response:

- keep v1 non-goals explicit;
- do not import classic SearchParams wholesale;
- add future features only after the ownership/storage core passes all gates;
- prefer extending small policy/output components over adding cross-stage
  lifecycle state.

## 20. Final implementation checklist

An implementer must not declare the task complete until every item is checked:

- [ ] Lc5 is new code, not renamed LC3.
- [ ] lc5 factory/mode and Meson option exist.
- [ ] Default search remains unchanged.
- [ ] Root PositionHistory has correct repetition counts.
- [ ] Default key includes eight NN positions; shorter keys are configurable.
- [ ] Transpositions share one node; parent edges retain independent stats.
- [ ] No path or queue owns a persistent NodeState pointer/reference.
- [ ] Every node recreation receives a new generation.
- [ ] Every Visit ends exactly once as completed or cancelled.
- [ ] In-flight reservations balance for surviving generations.
- [ ] Pending-node visits suspend/resume; there are no collision rollback
      events.
- [ ] Ready evaluations may exceed one batch and surplus is retained.
- [ ] Active work is hard-bounded by VisitPool.
- [ ] Visit selection/backup scales over configurable workers.
- [ ] Backend and NodeStore calls occur without graph/registry locks.
- [ ] Forced arbitrary eviction passes all recovery tests.
- [ ] Arbitrary-origin internal tests pass.
- [ ] Whole-game re-rooting retains same-game graph state.
- [ ] NewGame/incompatible start/backend replacement clears state.
- [ ] StartSearch/StopSearch/AbortSearch obey nonblocking contracts.
- [ ] WaitSearch joins all threads.
- [ ] go nodes/movetime/infinite/ponder-as-infinite work.
- [ ] Unsupported GoParams warn explicitly.
- [ ] Bestmove/PV are legal and white-perspective encoded.
- [ ] Nodes, NPS, EPS, batch, queue, suspension, storage, and eviction metrics
      are honest.
- [ ] Debug/release builds and Lc5-disabled build pass.
- [ ] ASan/UBSan/TSan CPU suites pass.
- [ ] GPU performance is within 5% of tuned classic.
- [ ] Equal-budget trivial performance matches or beats classic.
- [ ] Eight-worker scaling gate passes or is resolved with evidence.
- [ ] Long-run queues remain bounded.
- [ ] Fixed-node fastchess smoke test has no protocol/correctness failure.
- [ ] Final architecture/benchmark documentation contains actual commands and
      results.
