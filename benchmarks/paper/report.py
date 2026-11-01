"""Render benchmark summary.csv as a README-ready Markdown table and chart."""

import argparse
import csv
import math
from pathlib import Path

FRONTENDS = {
    "jump-cached": "JuMP cached",
    "jump-direct": "JuMP direct",
    "jump-genopt": "JuMP direct + GenOpt",
    "jumpy": "JuMPy.HiGHS",
}


def load_results(path):
    groups = {}
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        key = (row["model"], int(row["size"]))
        frontend = row["frontend"]
        if frontend not in FRONTENDS:
            raise ValueError(f"Unknown frontend: {frontend}")
        seconds = float(row["median_seconds"])
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"Invalid median for {key}, {frontend}: {seconds}")
        group = groups.setdefault(key, {})
        if frontend in group:
            raise ValueError(f"Duplicate result for {key}, {frontend}")
        if group and any((other["backend"], other["variables"]) !=
                         (row["backend"], row["variables"]) for other in group.values()):
            raise ValueError(f"Mismatched backend or variable count for {key}")
        group[frontend] = row
    if not groups:
        raise ValueError("The CSV has no timing samples")
    return groups


def labels(groups):
    result = dict(FRONTENDS)
    backends = {row["backend"] for group in groups.values() for row in group.values()}
    if backends != {"highs"}:
        result["jumpy"] = "JuMPy + GenOpt"
    return result


def markdown(groups):
    headings = ["Model", "n", "Variables", *labels(groups).values(), "Python / grouped Julia"]
    lines = ["| " + " | ".join(headings) + " |", "| --- | " + " | ".join(["---:"] * 7) + " |"]
    for (model, size), group in sorted(groups.items()):
        values = [float(group[f]["median_seconds"]) if f in group else None for f in FRONTENDS]
        fastest = min(v for v in values if v is not None)
        times = []
        for value in values:
            formatted = "—" if value is None else f"{value * 1000:,.1f} ms"
            times.append(f"**{formatted}**" if value == fastest else formatted)
        ratio = "—"
        if "jumpy" in group and "jump-genopt" in group:
            ratio = f'{float(group["jumpy"]["median_seconds"]) / float(group["jump-genopt"]["median_seconds"]):.2f}×'
        variables = int(next(iter(group.values()))["variables"])
        lines.append("| " + " | ".join([model.upper(), str(size), f"{variables:,}", *times, ratio]) + " |")
    return "\n".join(lines) + "\n"


def mermaid(groups):
    # One bar series with four categories works without recent Mermaid
    # multi-series legend extensions. Show the largest case for each model.
    sections = []
    for model in sorted({model for model, _ in groups}):
        size = max(n for name, n in groups if name == model)
        group = groups[model, size]
        names = [labels(groups)[f] for f in FRONTENDS if f in group]
        times = [float(group[f]["median_seconds"]) * 1000 for f in FRONTENDS if f in group]
        sections.append('\n'.join([
            "```mermaid", "xychart-beta",
            f'    title "{model.upper()}, n = {size} (lower is better)"',
            "    x-axis [" + ", ".join(f'\"{name}\"' for name in names) + "]",
            f'    y-axis "Milliseconds" 0 --> {max(times) * 1.15:.3f}',
            "    bar [" + ", ".join(f"{value:.3f}" for value in times) + "]",
            "```",
        ]))
    return "\n\n".join(sections) + "\n"


def write_report(csv_path, output=None, chart=None):
    groups = load_results(csv_path)
    result = markdown(groups)
    if output is not None:
        Path(output).write_text(result)
    if chart is not None:
        Path(chart).write_text(mermaid(groups))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mermaid", type=Path, help="Also write Markdown with Mermaid bar charts")
    args = parser.parse_args()
    print(write_report(args.csv, args.output, args.mermaid), end="")


if __name__ == "__main__":
    main()
