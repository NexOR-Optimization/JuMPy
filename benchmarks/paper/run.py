"""Run the two paper models sequentially and preserve every timing sample."""

import argparse
import csv
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from report import FRONTENDS, write_report

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
    parser.add_argument("--backend", choices=("storage", "gurobi", "highs"), default="highs")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=HERE)
    parser.add_argument("--facility-sizes", type=int, nargs="*", default=None)
    parser.add_argument("--lqcp-sizes", type=int, nargs="*", default=[100, 250, 500, 1000])
    parser.add_argument("--frontends", nargs="+", choices=tuple(FRONTENDS), default=list(FRONTENDS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--plot", action="store_true", help="Generate SVG and PNG (requires matplotlib)")
    args = parser.parse_args()
    if args.facility_sizes is None:
        args.facility_sizes = [] if args.backend == "highs" else [25, 50, 75, 100]
    if args.backend == "highs" and args.facility_sizes:
        parser.error("HiGHS does not support facility's quadratic constraints; use --backend storage or gurobi")
    if args.repeats < 1 or any(n < 1 for n in args.facility_sizes) or any(n < 3 for n in args.lqcp_sizes):
        parser.error("Use positive repeats, facility sizes >= 1 and LQCP sizes >= 3")
    if not args.facility_sizes and not args.lqcp_sizes:
        parser.error("Select at least one model size")
    if len(set(args.frontends)) != len(args.frontends):
        parser.error("Frontends must be distinct")
    if args.plot:
        import importlib.util
        if importlib.util.find_spec("matplotlib") is None:
            parser.error("--plot requires matplotlib: python -m pip install matplotlib")
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
    metadata["jumpy_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    digest = hashlib.sha256()
    with args.library.open("rb") as library:
        for chunk in iter(lambda: library.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata["library_sha256"] = digest.hexdigest()
    native_build = args.library.resolve().parents[1] / "build.json"
    if native_build.is_file():
        metadata["native_build"] = json.loads(native_build.read_text())
    manifest = args.project / "Manifest.toml"
    if manifest.is_file():
        # Preserve the exact environment even on Python 3.10 (no tomllib).
        (args.output / "Manifest.toml").write_text(manifest.read_text())
        try:
            import tomllib
        except ImportError:
            pass
        else:
            deps = tomllib.loads(manifest.read_text())["deps"]
            metadata["julia_packages"] = {name: deps[name][0] for name in ("GenOpt", "HiGHS", "JuMP", "MathOptInterface")}
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
            for size in sizes:
                count = sum(row.get("event") == "sample" and row["model"] == model
                            and row["frontend"] == frontend and row["size"] == size
                            for row in records)
                if count != args.repeats:
                    raise RuntimeError(f"Expected {args.repeats} samples for {frontend} {model} n={size}, got {count}; see {log_path}")
            summarize(records, args.output)
            print(f"Finished in {time.perf_counter() - start:.2f}s", flush=True)
    samples.close()
    for row in summarize(records, args.output):
        print(f"{row['model']:10} {row['size']:5} {row['frontend']:12} {row['median_seconds']:.6f}s", flush=True)
    print(write_report(args.output / "summary.csv", args.output / "table.md", args.output / "charts.md"))
    if args.plot:
        subprocess.run([sys.executable, str(HERE / "plot.py"), str(args.output)], check=True)


if __name__ == "__main__":
    main()
