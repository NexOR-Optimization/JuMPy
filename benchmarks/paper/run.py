"""Compare JuMP and the normal JuMPy HiGHS/NexOR backends."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile

from report import FRONTENDS, write_report

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def summarize(records, directory, metric="seconds", filename="summary.csv"):
    groups = {}
    for row in records:
        if row.get("event") == "sample" and row.get(metric) is not None:
            groups.setdefault((row["model"], row["size"], row["frontend"]), []).append(row)
    summary = []
    for (model, size, frontend), rows in sorted(groups.items()):
        values = [row[metric] for row in rows]
        summary.append(dict(model=model, size=size, frontend=frontend,
                            backend=rows[0]["backend"], variables=rows[0]["variables"],
                            samples=len(rows), median_seconds=statistics.median(values),
                            min_seconds=min(values), max_seconds=max(values),
                            peak_rss_bytes=max(row["peak_rss_bytes"] for row in rows)))
    if summary:
        with (directory / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("highs", "nexor"), default="highs")
    parser.add_argument("--library", type=Path, help="Optional normal backend library override")
    parser.add_argument("--project", type=Path, default=HERE)
    parser.add_argument("--facility-sizes", type=int, nargs="*", default=None)
    parser.add_argument("--lqcp-sizes", type=int, nargs="*", default=[100, 250, 500, 1000])
    parser.add_argument("--frontends", nargs="+", choices=tuple(FRONTENDS), default=list(FRONTENDS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--export-mof", action="store_true", help="Also time local MOF export separately")
    parser.add_argument("--plot", action="store_true", help="Generate SVG and PNG (requires matplotlib)")
    args = parser.parse_args()
    if args.facility_sizes is None:
        args.facility_sizes = [25, 50, 75, 100] if args.backend == "nexor" else []
    if args.backend == "highs" and args.facility_sizes:
        parser.error("HiGHS does not support facility's quadratic constraints; use --backend nexor")
    if args.repeats < 1 or any(n < 1 for n in args.facility_sizes) or any(n < 3 for n in args.lqcp_sizes):
        parser.error("Use positive repeats, facility sizes >= 1 and LQCP sizes >= 3")
    if not args.facility_sizes and not args.lqcp_sizes:
        parser.error("Select at least one model size")
    if len(set(args.frontends)) != len(args.frontends):
        parser.error("Frontends must be distinct")
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, JULIA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    if args.library is not None:
        env["JUMPY_LIB"] = str(args.library.resolve())
    metadata = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    os=platform.platform(), cpu=platform.processor(), backend=args.backend,
                    repeats=args.repeats, threads=1, source_revision="b4a3479",
                    solver_names_disabled=True, command=sys.argv,
                    timing_scope="construction and HiGHS ingestion (zero solve time limit, presolve off)"
                    if args.backend == "highs" else "local construction and cache materialization; no optimize or network",
                    export_mof=args.export_mof,
                    jumpy_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    manifest = args.project / "Manifest.toml"
    if manifest.is_file():
        (args.output / "Manifest.toml").write_text(manifest.read_text())
    if "jumpy" in args.frontends:
        from jumpy import backend
        library = str(args.library.resolve()) if args.library is not None else backend.find_lib(args.backend)
        if library is None:
            parser.error(f"Install or build the normal jumpy.{args.backend} backend first")
        digest = hashlib.sha256()
        with open(library, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        metadata.update(library=library, library_sha256=digest.hexdigest())
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    records = []
    with (args.output / "samples.jsonl").open("w") as samples, tempfile.TemporaryDirectory(dir=args.output) as scratch:
        for model, sizes in (("facility", args.facility_sizes), ("lqcp", args.lqcp_sizes)):
            if not sizes:
                continue
            for frontend in args.frontends:
                if frontend == "jumpy":
                    command = [sys.executable, str(HERE / "run_jumpy.py"), "--backend", args.backend,
                               "--model", model, "--sizes", *map(str, sizes), "--repeats", str(args.repeats)]
                else:
                    command = ["julia", "--startup-file=no", f"--project={args.project.resolve()}",
                               str(HERE / "run_jump.jl"), "--backend", args.backend,
                               "--mode", frontend.removeprefix("jump-"), "--model", model,
                               "--sizes", ','.join(map(str, sizes)), "--repeats", str(args.repeats)]
                if args.export_mof:
                    command.extend(["--export", str(Path(scratch) / "model.mof.json")])
                print(f"Running {frontend} {model}: {sizes}", flush=True)
                log_path = args.output / f"{frontend}-{model}.log"
                with log_path.open("w") as log:
                    subprocess.run(command, env=env, text=True, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=args.timeout, check=True)
                for line in log_path.read_text().splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict) and row.get("event") in ("sample", "warmup"):
                        records.append(row)
                        samples.write(json.dumps(row) + "\n")
                samples.flush()
                for size in sizes:
                    count = sum(row.get("event") == "sample" and row["model"] == model
                                and row["frontend"] == frontend and row["size"] == size for row in records)
                    if count != args.repeats:
                        raise RuntimeError(f"Expected {args.repeats} samples for {frontend} {model} n={size}, got {count}; see {log_path}")
                summarize(records, args.output)
    print(write_report(args.output / "summary.csv", args.output / "table.md", args.output / "charts.md"))
    if args.export_mof:
        summarize(records, args.output, "export_seconds", "mof-summary.csv")
        print("Local MOF export:")
        print(write_report(args.output / "mof-summary.csv", args.output / "mof-table.md", args.output / "mof-charts.md"))
    if args.plot:
        subprocess.run([sys.executable, str(HERE / "plot.py"), str(args.output)], check=True)


if __name__ == "__main__":
    main()
