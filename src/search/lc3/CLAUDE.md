# Lc3 search algorithm overview

Algorithm is in a very early stage of development, not working even barely.

* No time management, search params, stopping, stats, UCI info etc code yet.

## Algorithm structure

* Instead of traditional batch-based NN evaluation, the algorithm generates a stream of positions.
* `gather_worker.{h,cc}`, `lczero::lc3::MctsGatherWorker` traverses the tree from the root and generates positions to evaluate.
* `eval_worker.{h,cc}`, `lczero::lc3::EvalWorker` evaluates the positions using NN (but also looking up in the cache and detecting terminal positions) and sends the results further.
* `backprop_worker.{h,cc}`, `lczero::lc3::BackpropWorker` receives the results and backpropagates them towards the root.
* `node_repository.{h,cc}`, `lczero::lc3::NodeRepository` stores the nodes and provides methods for efficient access/update. Currently, as a draft, just a hashmap to quite bulky structure is used.
* `session.{h,cc}`, `lczero::lc3::Session` orchestrates the workers, sets up (lock-free) channels, etc.
* `watchdog_worker` (planned, nothing is written yet) will watch for the clock, monitor the batch sizes and otherwise manage the search.
* It will be one watchdog thread, and arbitrary number of gather, eval and backprop workers.