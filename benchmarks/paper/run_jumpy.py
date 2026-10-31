"""Measure iterator-based JuMPy models in a dedicated native Julia runtime."""

import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from jumpy import backend
from jumpy._moi import ScalarSets
from jumpy.model import Model
from models import BUILDERS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("storage", "gurobi", "highs"), default="storage")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(BUILDERS), required=True)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    os.environ["JUMPY_LIB"] = str(args.library.resolve())
    startup = time.perf_counter()
    lib = backend._load_lib(args.backend)
    lib.jumpy_paper_gc.argtypes = []
    lib.jumpy_paper_gc.restype = None
    lib.jumpy_paper_num_variables.argtypes = [ctypes.c_void_p]
    lib.jumpy_paper_num_variables.restype = ctypes.c_longlong
    lib.jumpy_paper_write_model.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.jumpy_paper_write_model.restype = ctypes.c_int
    initialization = time.perf_counter() - startup
    sets = ScalarSets(args.backend)

    def construct(size):
        model = Model(backend.JuliacOps(lib, sets))
        if args.backend != "storage":
            model.set_attribute("Threads" if args.backend == "gurobi" else "threads", 1)
        if args.backend != "storage" and args.model != "latency":
            model.set_attribute("TimeLimit" if args.backend == "gurobi" else "time_limit", 0.0)
            model.set_attribute("Presolve" if args.backend == "gurobi" else "presolve", 0 if args.backend == "gurobi" else "off")
        BUILDERS[args.model](model, size)
        if args.backend != "storage":
            model.optimize()
        return model

    start = time.perf_counter()
    if args.warmup:
        warm = construct(args.warmup)
        warm.close()
    warmup_seconds = time.perf_counter() - start
    print(json.dumps({"event": "warmup", "frontend": "jumpy", "backend": args.backend,
                      "model": args.model, "initialization_seconds": initialization,
                      "warmup_seconds": warmup_seconds}), flush=True)
    for size in args.sizes:
        for sample in range(args.repeats):
            gc.collect()
            lib.jumpy_paper_gc()
            start = time.perf_counter()
            model = construct(size)
            elapsed = time.perf_counter() - start
            variables = lib.jumpy_paper_num_variables(model._ops._m)
            if variables < 0:
                raise RuntimeError("Could not query variable count")
            if args.export is not None:
                args.export.parent.mkdir(parents=True, exist_ok=True)
                if lib.jumpy_paper_write_model(model._ops._m, os.fsencode(args.export.resolve())):
                    raise RuntimeError("Could not export validation model")
            objective = model.objective_value() if args.model == "latency" and args.backend != "storage" else None
            if objective is not None and abs(objective - 205.0) > 1e-7:
                raise AssertionError(f"Unexpected latency LP objective: {objective}")
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_bytes = peak if sys.platform == "darwin" else peak * 1024
            print(json.dumps({"event": "sample", "frontend": "jumpy", "backend": args.backend,
                              "model": args.model, "size": size, "sample": sample,
                              "seconds": elapsed, "variables": variables, "peak_rss_bytes": peak_bytes,
                              "python": platform.python_version(), "objective": objective}), flush=True)
            model.close()


if __name__ == "__main__":
    main()
