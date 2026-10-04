# Lc5: bestmove fallback without completed root-edge visits

## Confirmed weakness

Lc5's principal-variation construction ignores root edges with no completed
visits. If no root edge has a completed visit, bestmove falls back to the legal
move with the smallest internal raw move encoding. It does not use the neural
policy, even when the root has already been evaluated and its priors are
available.

Relevant code: `SearchRun::BuildPv()` and `SearchRun::FallbackMove()` in
`src/search/lc5/search.cc`.

## Reproduction (2026-10-04)

Using the release engine, BT4-1024x15x32h-swa-6147500 weights, `cuda-auto` on an
RTX 5090, and `Threads=2`:

1. Warm the backend.
2. Send `ucinewgame`, `position startpos`, and `isready`; wait for `readyok`.
3. Send `go nodes 1`.

Across five fresh-root runs, lc5 completed one visit and one neural evaluation,
reported no PV, and returned `bestmove b1a3`. That visit evaluates the root;
it does not complete a root child-edge visit. An earlier trivial-backend probe
also reproduced the behavior; classic selected a policy-ranked move at the
same fresh-root node limit.

The CUDA measurements and harness are retained outside version control under
`/tmp/lc5-investigation/timing-diagnostics/` and
`build/lc5_timing_diagnostics.py`, respectively. These paths are temporary
investigation artifacts, not dependencies of this document.

## Potential impact and limits of the evidence

This can select a poor move when a search stops before completing any root edge,
including very short budgets or an exhausted clock. With the default 200 ms
move-overhead reserve, lc5's clock allocator assigns zero search time when the
remaining clock is at or below 200 ms.

The two investigated `8.0+0.08` games both drew, and every final PV was populated.
The fallback did not explain those games; its contribution to reported losses
has not been established.

Separating bestmove delivery from drain deliberately preserves this existing
fallback policy. Prompt response must not wait for extra evaluations merely to
avoid fallback. A separate change should decide how to use available root
policy when completed child-edge evidence is absent, while retaining a legal
fallback when the root has not been evaluated at all.

## Follow-up scope

- Prefer available root policy over raw move encoding when there are no
  completed root-edge visits.
- Specify consistent bestmove/PV behavior for an expanded but unvisited root.
- Retain safe behavior before root expansion, for Black, and with no legal moves.
- Add deterministic tests for fresh `nodes 1`, zero-time searches, and root
  policies whose highest-prior move differs from the raw-encoding fallback.
- Measure occurrence in short-clock games; do not infer strength from this
  single-position reproduction.
