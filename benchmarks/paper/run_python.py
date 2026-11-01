"""Benchmark Python modeling libraries through their in-memory HiGHS interface."""

import argparse
import importlib
from importlib.metadata import version
import json
import platform
import resource
import sys
import time

from python_models import BUILDERS, construct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend", choices=tuple(BUILDERS), required=True)
    parser.add_argument("--model", choices=("lqcp",), required=True)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 3 for n in args.sizes):
        parser.error("Use positive repeats and LQCP sizes >= 3")

    startup = time.perf_counter()
    # Imports and their first-use initialization are excluded, as for JuMPy.
    for name in ("highspy", "numpy", "scipy", args.frontend):
        importlib.import_module(name)
    import highspy
    versions = {name: version(name) for name in ("highspy", "numpy", "scipy", args.frontend)}
    started = time.perf_counter()
    if args.warmup:
        warm = construct(args.frontend, args.warmup)
        warm.close()
    print(json.dumps(dict(event="warmup", frontend=args.frontend, backend="highs",
                          model=args.model, python=platform.python_version(), versions=versions,
                          highs_version=highspy.Highs().version(),
                          initialization_seconds=started-startup,
                          warmup_seconds=time.perf_counter()-started)), flush=True)
    for size in args.sizes:
        for sample in range(args.repeats):
            started = time.perf_counter()
            model = construct(args.frontend, size)
            seconds = time.perf_counter() - started
            try:
                peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                print(json.dumps(dict(event="sample", frontend=args.frontend, backend="highs",
                                      model=args.model, size=size, sample=sample, seconds=seconds,
                                      variables=model.highs.getNumCol(),
                                      peak_rss_bytes=peak if sys.platform == "darwin" else peak*1024)), flush=True)
            finally:
                model.close()


if __name__ == "__main__":
    main()
