"""Plot the scalable-GP sweep and fit a measured scaling order.

For every method the wall time and the peak resident memory are plotted
against the swept quantity on log-log axes, and a least-squares slope is fitted
over the largest points. That slope is the measured order, printed beside the
order the review claims, so the comparison is explicit rather than asserted.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SWEEP_LABEL = {
    "samples": "training points n",
    "inducing": "inducing size m",
    "experts": "expert count M",
    "dimension": "input dimension d",
}

# Colorblind-safe blue, orange, magenta, and teal. Markers and line styles form
# a second encoding so larger local-expert comparisons remain legible without
# color.
PALETTE = ("#0072B2", "#E69F00", "#CC79A7", "#009E73")
MARKERS = ("o", "s", "^", "D", "v", "P", "X")
LINESTYLES = ("-", "--", "-.", ":")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fitted_slope(xs: list[float], ys: list[float], tail: int = 3) -> float | None:
    """Least-squares log-log slope over the largest `tail` points."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pairs) < 2:
        return None
    pairs = pairs[-max(tail, 2):]
    log_x = [math.log(x) for x, _ in pairs]
    log_y = [math.log(y) for _, y in pairs]
    mean_x = sum(log_x) / len(log_x)
    mean_y = sum(log_y) / len(log_y)
    denominator = sum((value - mean_x) ** 2 for value in log_x)
    if denominator <= 0:
        return None
    numerator = sum(
        (a - mean_x) * (b - mean_y) for a, b in zip(log_x, log_y)
    )
    return numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument(
        "--metric",
        choices=("train_seconds", "predict_seconds", "peak_kib", "smse"),
        default="train_seconds",
    )
    arguments = parser.parse_args()

    rows = read_rows(arguments.input)
    passed = [
        row for row in rows if row.get("status", "").strip() in {"", "pass"}
    ]
    if not passed:
        raise SystemExit("no pass or legacy unstamped rows to plot")
    sweep = passed[0]["sweep"]
    if any(row["sweep"] != sweep for row in passed):
        raise SystemExit("pass rows contain more than one sweep")

    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    skipped_legacy = 0
    for row in passed:
        try:
            value = float(row["swept_value"])
            metric = float(row[arguments.metric])
        except (KeyError, ValueError) as error:
            raise SystemExit(
                f"invalid {arguments.metric} row for {row.get('method', '?')}"
            ) from error
        if not math.isfinite(value):
            raise SystemExit(
                f"nonfinite {arguments.metric} row for {row.get('method', '?')}"
            )
        if not math.isfinite(metric):
            # Historical CSVs predate explicit status/refusal columns and use
            # bare NaNs for allocation failures or undefined predictive
            # densities. Keep those rows in the audit trail, but omit them
            # from a metric plot. A modern `pass` row with a nonfinite metric
            # remains an error, so new measurements cannot hide a failure.
            if row.get("status", "").strip() == "":
                skipped_legacy += 1
                continue
            raise SystemExit(
                f"nonfinite {arguments.metric} row for {row.get('method', '?')}"
            )
        if value <= 0:
            raise SystemExit("logarithmic swept values must be positive")
        if arguments.metric != "smse" and metric <= 0:
            raise SystemExit(
                f"logarithmic {arguments.metric} values must be positive"
            )
        series[row["method"]].append((value, metric))

    if not series:
        raise SystemExit("no finite rows remain after filtering")

    figure, axes = plt.subplots(figsize=(9.0, 6.0))
    slopes: dict[str, float | None] = {}
    for index, method in enumerate(sorted(series)):
        points = sorted(series[method])
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        slopes[method] = fitted_slope(xs, ys)
        label = method
        if slopes[method] is not None:
            label = f"{method} (slope {slopes[method]:.2f})"
        axes.plot(
            xs,
            ys,
            color=PALETTE[index % len(PALETTE)],
            marker=MARKERS[index % len(MARKERS)],
            linestyle=LINESTYLES[index % len(LINESTYLES)],
            linewidth=1.6,
            markersize=5.5,
            markeredgecolor="#202020",
            markeredgewidth=0.5,
            label=label,
        )

    axes.set_xscale("log")
    if arguments.metric != "smse":
        axes.set_yscale("log")
    axes.set_xlabel(SWEEP_LABEL.get(sweep, sweep))
    axes.set_ylabel(
        {
            "train_seconds": "training wall time [s]",
            "predict_seconds": "prediction wall time [s]",
            "peak_kib": "peak resident memory [KiB]",
            "smse": "standardized mean squared error",
        }[arguments.metric]
    )
    axes.set_title(
        f"Scalable GPs: {arguments.metric.replace('_', ' ')} against "
        f"{SWEEP_LABEL.get(sweep, sweep)}"
    )
    axes.grid(True, which="both", alpha=0.3)
    axes.set_axisbelow(True)
    axes.legend(
        fontsize=7.5,
        ncol=min(4, len(series)),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
    )
    if skipped_legacy:
        print(
            f"skipped {skipped_legacy} legacy unstamped NaN row(s) "
            f"from {arguments.metric}",
            file=sys.stderr,
        )
    figure.subplots_adjust(bottom=0.30, left=0.11, right=0.98, top=0.90)

    output = arguments.prefix.with_name(
        f"{arguments.prefix.name}_{arguments.metric}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    print(f"wrote {output}")
    for method, slope in sorted(slopes.items()):
        if slope is None:
            continue
        print(f"  {method:6s} measured slope {slope:+.2f}")


if __name__ == "__main__":
    main()
