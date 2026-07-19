#!/usr/bin/env python3
"""Small reproducible classic/Lc5 UCI comparison harness.

This deliberately records raw final info lines as well as parsed counters so
new Lc5 metrics remain available without changing the harness schema.
"""

import argparse
import json
import platform
import re
import subprocess
import time


def run_search(args, algorithm, warmup=False):
    command = [args.engine, algorithm, f"--backend={args.backend}",
               f"--threads={args.threads}",
               f"--minibatch-size={args.batch}"]
    if args.weights:
        command.append(f"--weights={args.weights}")
    if algorithm == "lc5":
        command += [f"--eval-threads={args.eval_threads}",
                    f"--max-active-visits={args.max_active_visits}"]
    process = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               text=True, bufsize=1)
    assert process.stdin and process.stdout
    process.stdin.write("uci\nisready\n")
    process.stdin.flush()
    for line in process.stdout:
        if line.strip() == "readyok":
            break
    position = "position startpos" if args.fen == "startpos" else f"position fen {args.fen}"
    process.stdin.write(position + "\n")
    process.stdin.write(f"go movetime {args.warmup_ms if warmup else args.movetime_ms}\n")
    process.stdin.flush()
    final_info = ""
    bestmove = ""
    for line in process.stdout:
        line = line.strip()
        if line.startswith("info "):
            final_info = line
        elif line.startswith("bestmove "):
            bestmove = line
            break
    process.stdin.write("quit\n")
    process.stdin.flush()
    process.wait(timeout=10)

    def field(name):
        match = re.search(rf"(?:^| ){name} (\d+)(?: |$)", final_info)
        return int(match.group(1)) if match else None

    return {"algorithm": algorithm, "command": command,
            "nodes": field("nodes"), "nps": field("nps"),
            "eps": field("eps"), "time_ms": field("time"),
            "bestmove": bestmove, "final_info": final_info}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="build/release/lc0")
    parser.add_argument("--weights")
    parser.add_argument("--backend", default="trivial")
    parser.add_argument("--fen", default="startpos")
    parser.add_argument("--movetime-ms", type=int, default=30000)
    parser.add_argument("--warmup-ms", type=int, default=2000)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--eval-threads", type=int, default=1)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--max-active-visits", type=int, default=4096)
    args = parser.parse_args()

    results = []
    for algorithm in ("classic", "lc5"):
        run_search(args, algorithm, warmup=True)
        for _ in range(args.repetitions):
            results.append(run_search(args, algorithm))
    output = {"timestamp": time.time(), "platform": platform.platform(),
              "processor": platform.processor(), "arguments": vars(args),
              "results": results}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
