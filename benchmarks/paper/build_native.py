"""Build a C-ABI benchmark backend using the repository's JuMPy entry points."""

import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("storage", "gurobi", "highs"), default="storage")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--juliac-project", default="@juliac")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    output = (args.output or here / "build" / args.backend).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, JUMPY_PAPER_BACKEND=args.backend)
    # Keep the compiler available for this new solver-specific benchmark image.
    # Warm-up is reported separately; steady-state timings exclude compilation.
    command = [
        "julia", "--startup-file=no", f"--project={args.juliac_project}",
        "-m", "JuliaC", "--project", str(args.project.resolve()),
        "--compile-ccallable", "--trim=no", "--jl-option", "handle-signals=no",
        "--output-lib", str(output / f"libjumpy_{args.backend}"),
        "--bundle", str(output), str(here / "library_entry.jl"),
    ]
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
