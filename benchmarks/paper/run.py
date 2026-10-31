"""Run the two paper models sequentially and preserve every timing sample."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def summarize(records, directory):
    groups = {}
    for row in records:
        if row.get("event") == "sample":
            groups.setdefault((row["model"], row["size"], row["frontend"]), []).append(row)
    summary = []
    for (model, size, frontend), rows in sorted(groups.items()):
        values = [row["seconds"] for row in rows]
        summary.append(dict(model=model, size=size, frontend=frontend,
                            backend=rows[0]["backend"], variables=rows[0]["variables"],
                            samples=len(rows), median_seconds=statistics.median(values),
                            min_seconds=min(values), max_seconds=max(values),
                            peak_rss_bytes=max(row["peak_rss_bytes"] for row in rows)))
    if summary:
        with (directory / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("storage", "gurobi", "highs"), default="storage")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=HERE)
    parser.add_argument("--facility-sizes", type=int, nargs="*", default=[25, 50, 75, 100])
    parser.add_argument("--lqcp-sizes", type=int, nargs="*", default=[500, 1000, 1500, 2000])
    parser.add_argument("--frontends", nargs="+", default=["jump-direct", "jump-cached", "jumpy"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')), platform.processor()) if sys.platform == 'linux' else platform.processor()
    metadata = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), cpu=cpu,
                    os=platform.platform(), backend=args.backend, repeats=args.repeats,
                    threads=1, source_revision="b4a3479", solver_names_disabled=True,
                    native_image="JuliaC --trim=no; warmed before timing",
                    timing_scope="construction and cache materialization" if args.backend == "storage"
                    else "construction and optimize with zero time limit, presolve disabled",
                    command=sys.argv, library=str(args.library.resolve()))
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + '\n')
    env = dict(os.environ, JULIA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    records = []
    samples = (args.output / "samples.jsonl").open("w")
    for model, sizes in (("facility", args.facility_sizes), ("lqcp", args.lqcp_sizes)):
        if not sizes:
            continue
        for frontend in args.frontends:
            if frontend == "jumpy":
                command = [sys.executable, str(HERE / "run_jumpy.py"), "--backend", args.backend,
                           "--library", str(args.library.resolve()), "--model", model,
                           "--sizes", *map(str, sizes), "--repeats", str(args.repeats)]
            else:
                mode = frontend.removeprefix("jump-")
                command = ["julia", "--startup-file=no", f"--project={args.project.resolve()}",
                           str(HERE / "run_jump.jl"), "--backend", args.backend, "--mode", mode,
                           "--model", model, "--sizes", ','.join(map(str, sizes)),
                           "--repeats", str(args.repeats)]
            print(f"Running {frontend} {model}: {sizes}", flush=True)
            start = time.perf_counter()
            log_path = args.output / f"{frontend}-{model}.log"
            with log_path.open("w") as log:
                result = subprocess.run(command, env=env, text=True, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=args.timeout)
            output = log_path.read_text()
            if result.returncode:
                print(output[-6000:], flush=True)
                raise SystemExit(f"{frontend} {model} exited {result.returncode}")
            for line in output.splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("event") in ("sample", "warmup"):
                    records.append(row)
                    samples.write(json.dumps(row) + '\n')
                    samples.flush()
            summarize(records, args.output)
            print(f"Finished in {time.perf_counter() - start:.2f}s", flush=True)
    samples.close()
    for row in summarize(records, args.output):
        print(f"{row['model']:10} {row['size']:5} {row['frontend']:12} {row['median_seconds']:.6f}s", flush=True)


if __name__ == "__main__":
    main()
