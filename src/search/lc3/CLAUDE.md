# Lc3 search algorithm overview

Algorithm is in a very early stage of development, working just barely.

* No time management, search params, stopping, stats, UCI info etc code yet.

## Algorithm structure

* Instead of traditional batch-based NN evaluation, the algorithm generates a stream of positions.
* The session has multiple `GatherWorker`, `EvalWorker` and `BackpropWorker` threads, and one `WatchdogWorker`.
* `gather_worker.{h,cc}`, `lczero::lc3::GatherWorker` traverses the tree from the root and generates positions to evaluate.
* `eval_worker.{h,cc}`, `lczero::lc3::EvalWorker` evaluates the positions using NN (but also looking up in the cache and detecting terminal positions) and sends the results further.
* `backprop_worker.{h,cc}`, `lczero::lc3::BackpropWorker` receives the results and backpropagates them towards the root.
* `node_repository.{h,cc}`, `lczero::lc3::NodeRepository` stores the nodes and provides methods for efficient access/update. Currently, as a draft, just a hashmap to quite bulky structure is used.
* `session.{h,cc}`, `lczero::lc3::Session` orchestrates the workers, sets up (lock-free) channels, etc.
* `watchdog_worker` (planned, nothing is written yet) will watch for the clock, monitor the batch sizes and otherwise manage the search.
* It will be one watchdog thread, and arbitrary number of gather, eval and backprop workers.

### GatherWorker

`GatherWorker` threads, in a loop, do the forward/gather step, and

* Send nodes either to the eval workers (if it's a newly discovered node),
* Or to backprop step (if it's a collision, e.g. visits from this or other gatherers led to the same node and it's already being evaluated; or if it's a known terminal node). 

* Gatherer tries to pick a certain batch size N.
* However, not all of the collected batch goes to eval (some are collisions and terminals which go directly to `BackpropWorker` queue)
* Batched gather greatly improves gather speed, but slightly reduces node quality.
* What also decreases quality, is too many nodes in the queue to eval. Instead of growing the queue, it's better to keep it as short as possible, but to keep eval fully utilized. (that's because after the evaluated nodes are backpropagated, subsequent gathers collects higher quality nodes)
  
### EvalWorker

Fetch nodes from the queue, determine whether they are terminal or cached

* If yes, this case send them to backprop immediately (fraction of such nodes changes smoothly),
* if not, collect the batch of size B, evaluate them through NN, and send to the backprop then. On one iteration, evaluator cannot compute more than B, and underutilization wastes compute too
