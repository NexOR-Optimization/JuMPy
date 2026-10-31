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
    styles = {"jump-direct": ("JuMP direct", "#3265a8", "o"),
              "jump-cached": ("JuMP cached + materialize", "#b65e17", "s"),
              "jumpy": ("JuMPy + GenOpt (C ABI)", "#18876b", "^")}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    for ax, model, title in zip(axes, ("facility", "lqcp"), ("Facility location", "Optimal control (LQCP)")):
        for frontend, (label, color, marker) in styles.items():
            points = sorted((r for r in rows if r["model"] == model and r["frontend"] == frontend), key=lambda r: int(r["variables"]))
            if not points:
                continue
            x = [int(r["variables"]) for r in points]
            y = [float(r["median_seconds"]) for r in points]
            ax.plot(x, y, label=label, color=color, marker=marker)
            ax.fill_between(x, [float(r["min_seconds"]) for r in points],
                            [float(r["max_seconds"]) for r in points], color=color, alpha=.12)
        ax.set(xscale="log", yscale="log", title=title, xlabel="Variables", ylabel="Build time (seconds)")
        positions = sorted({int(r["variables"]) for r in rows if r["model"] == model})
        ax.set_xticks(positions)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e6:.3g} M" if x >= 1e6 else f"{x / 1e3:.3g} k"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="both", alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("JuMP paper models — warmed construction, identical MOI storage\nMedian of 3 runs; shaded range; one thread; lower is better", fontsize=11)
    fig.savefig(args.results / "comparison.png", dpi=180)
    svg = args.results / "comparison.svg"
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == "__main__":
    main()
