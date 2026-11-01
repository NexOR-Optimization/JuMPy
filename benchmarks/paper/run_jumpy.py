"""Benchmark the installed jumpy.highs or jumpy.nexor interface."""

import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

from models import BUILDERS


def configure(model, backend, problem):
    if backend == "highs":
        model.set_attribute("threads", 1)
        if problem != "latency":
            model.set_attribute("time_limit", 0.0)
            model.set_attribute("presolve", "off")


def construct(interface, backend, problem, size):
    model = interface.Model()
    try:
        configure(model, backend, problem)
        BUILDERS[problem](model, size)
        if backend == "highs":
            model.optimize()
        return model
    except BaseException:
        model.close()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("highs", "nexor"), default="highs")
    parser.add_argument("--library", type=Path, help="Optional normal backend library override")
    parser.add_argument("--model", choices=tuple(BUILDERS), required=True)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--export", type=Path, help="Also time local MOF export")
    args = parser.parse_args()
    if args.backend == "highs" and args.model == "facility":
        parser.error("Facility requires --backend nexor (local construction only)")
    if args.library is not None:
        os.environ["JUMPY_LIB"] = str(args.library.resolve())
    if args.export is not None:
        args.export.parent.mkdir(parents=True, exist_ok=True)
    startup = time.perf_counter()
    interface = importlib.import_module(f"jumpy.{args.backend}")
    from jumpy import backend
    # Resolve the ordinary package library, including an installed wheel.
    library = backend.find_lib(args.backend)
    started = time.perf_counter()
    if args.warmup:
        warm = construct(interface, args.backend, args.model, args.warmup)
        try:
            if args.export is not None:
                warm.write_mof(args.export)
        finally:
            warm.close()
    print(json.dumps(dict(event="warmup", frontend="jumpy", backend=args.backend,
                          model=args.model, library=library,
                          initialization_seconds=started-startup,
                          warmup_seconds=time.perf_counter()-started)), flush=True)
    for size in args.sizes:
        for sample in range(args.repeats):
            started = time.perf_counter()
            model = construct(interface, args.backend, args.model, size)
            seconds = time.perf_counter() - started
            try:
                export_seconds = None
                if args.export is not None:
                    started = time.perf_counter()
                    model.write_mof(args.export)
                    export_seconds = time.perf_counter() - started
                objective = model.objective_value() if args.model == "latency" and args.backend == "highs" else None
                if objective is not None and abs(objective - 205.0) > 1e-7:
                    raise AssertionError(f"Unexpected latency LP objective: {objective}")
                peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                print(json.dumps(dict(event="sample", frontend="jumpy", backend=args.backend,
                                      model=args.model, size=size, sample=sample, seconds=seconds,
                                      export_seconds=export_seconds, variables=model.num_variables(),
                                      peak_rss_bytes=peak if sys.platform == "darwin" else peak*1024,
                                      python=platform.python_version(), objective=objective)), flush=True)
            finally:
                model.close()


if __name__ == "__main__":
    main()
