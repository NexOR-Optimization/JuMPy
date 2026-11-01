"""Plot preserved timing samples; requires matplotlib."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    rows = list(csv.DictReader((args.results / "summary.csv").open()))
    from report import FRONTENDS, labels, load_results
    names = labels(load_results(args.results / "summary.csv"))
    styles = dict(zip(FRONTENDS, (("#b65e17", "s"), ("#3265a8", "o"),
                                ("#9b46a0", "D"), ("#18876b", "^"))))
    models = sorted({r["model"] for r in rows})
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.7),
                             layout="constrained", squeeze=False)
    for ax, model in zip(axes[0], models):
        title = {"facility": "Facility location", "lqcp": "Optimal control (LQCP)"}.get(model, model)
        for frontend, (color, marker) in styles.items():
            points = sorted((r for r in rows if r["model"] == model and r["frontend"] == frontend), key=lambda r: int(r["variables"]))
            if not points:
                continue
            x = [int(r["variables"]) for r in points]
            y = [float(r["median_seconds"]) for r in points]
            ax.plot(x, y, label=names[frontend], color=color, marker=marker)
            ax.fill_between(x, [float(r["min_seconds"]) for r in points],
                            [float(r["max_seconds"]) for r in points], color=color, alpha=.12)
        ax.set(xscale="log", yscale="log", title=title, xlabel="Variables", ylabel="Build time (seconds)")
        positions = sorted({int(r["variables"]) for r in rows if r["model"] == model})
        ax.set_xticks(positions)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e6:.3g} M" if x >= 1e6 else f"{x / 1e3:.3g} k"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="both", alpha=.2)
        ax.legend(fontsize=8)
    backend = rows[0]["backend"]
    repeats = sorted({r["samples"] for r in rows})
    fig.suptitle(f"JuMP paper models — warmed construction, {backend}\n"
                 f"Median of {'/'.join(repeats)} runs; shaded range; one thread; lower is better", fontsize=11)
    fig.savefig(args.results / "comparison.png", dpi=180)
    svg = args.results / "comparison.svg"
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == "__main__":
    main()
