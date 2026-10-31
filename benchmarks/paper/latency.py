"""Time fresh processes for the paper's two-variable HiGHS LP (optimum 205)."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=HERE)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    env = dict(os.environ, JULIA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for frontend in ("jump-direct", "jump-cached", "jumpy"):
        for sample in range(args.repeats):
            if frontend == "jumpy":
                command = [sys.executable, str(HERE / "run_jumpy.py"), "--backend", "highs",
                           "--library", str(args.library.resolve()), "--model", "latency",
                           "--sizes", "1", "--warmup", "0", "--repeats", "1"]
            else:
                command = ["julia", "--startup-file=no", f"--project={args.project.resolve()}",
                           str(HERE / "run_jump.jl"), "--backend", "highs",
                           "--mode", frontend.removeprefix("jump-"), "--model", "latency",
                           "--sizes", "1", "--warmup", "0", "--repeats", "1"]
            started = time.perf_counter()
            result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, check=True, timeout=120)
            elapsed = time.perf_counter() - started
            rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
            row = next(r for r in rows if r.get('event') == 'sample')
            if abs(row['objective'] - 205.0) > 1e-7:
                raise AssertionError(row)
            records.append(dict(frontend=frontend, sample=sample, process_seconds=elapsed,
                                first_model_seconds=row['seconds'], objective=row['objective']))
            args.output.write_text(json.dumps(records, indent=2) + '\n')
        values = [r['process_seconds'] for r in records if r['frontend'] == frontend]
        print(f"{frontend}: median fresh-process time {statistics.median(values):.4f}s", flush=True)


if __name__ == "__main__":
    main()
