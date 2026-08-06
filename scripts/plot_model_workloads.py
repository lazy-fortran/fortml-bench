#!/usr/bin/env python3
"""Plot matched exact-GP and MLP phase timings from the raw CSV record."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


SERIES = (
    ("fortml_gfortran", "cpu", "FortML, gfortran CPU", "#0072B2", ""),
    ("fortml_nvfortran", "cpu", "FortML, nvfortran CPU", "#56B4E9", "//"),
    ("gpytorch_exact", "cpu", "GPyTorch, CPU", "#E69F00", "xx"),
    ("gpytorch_exact", "cuda", "GPyTorch, CUDA", "#CC79A7", ".."),
    ("pytorch", "cpu", "PyTorch, CPU", "#E69F00", "xx"),
    ("pytorch", "cuda", "PyTorch, CUDA", "#CC79A7", ".."),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def series_for(workload: str) -> tuple[tuple[str, str, str, str, str], ...]:
    reference = "gpytorch_exact" if workload == "exact_gp" else "pytorch"
    return tuple(
        entry for entry in SERIES if entry[0].startswith("fortml_") or entry[0] == reference
    )


def plot_workload(
    rows: list[dict[str, str]],
    workload: str,
    phases: tuple[str, str],
    title: str,
    subtitle: str,
    output: Path,
) -> None:
    selected = [
        row
        for row in rows
        if row["workload"] == workload and row["status"] == "pass"
    ]
    styles = series_for(workload)
    positions = np.arange(len(phases), dtype=float)
    width = 0.18

    figure, axes = plt.subplots(figsize=(9.2, 6.0))
    for index, (backend, device, label, color, hatch) in enumerate(styles):
        values: list[float] = []
        for phase in phases:
            matches = [
                row
                for row in selected
                if row["phase"] == phase
                and row["backend"] == backend
                and row["device"] == device
            ]
            values.append(
                float(matches[0]["seconds_per_operation"]) * 1000.0
                if matches
                else np.nan
            )
        offset = (index - (len(styles) - 1) / 2.0) * width
        bars = axes.bar(
            positions + offset,
            values,
            width=width,
            label=label,
            color=color,
            edgecolor="#202020",
            linewidth=0.7,
            hatch=hatch,
        )
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            axes.annotate(
                f"{value:.3g}",
                (bar.get_x() + bar.get_width() / 2.0, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    axes.set_xticks(positions, [phase.capitalize() for phase in phases])
    axes.set_ylabel("milliseconds per operation (log scale)")
    axes.set_yscale("log")
    axes.grid(axis="y", which="both", alpha=0.25)
    axes.set_axisbelow(True)
    axes.set_title(f"{title}\n{subtitle}", pad=12)
    axes.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    figure.subplots_adjust(bottom=0.25, left=0.10, right=0.98, top=0.82)
    figure.text(
        0.5,
        0.01,
        "FortML CUDA: unsupported by these exact-GP and MLP application paths.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("results/model_workloads.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    arguments = parser.parse_args()
    rows = read_rows(arguments.input)
    if not rows:
        raise SystemExit("no benchmark rows found")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
        }
    )
    plot_workload(
        rows,
        "exact_gp",
        ("fit", "predict"),
        "Exact small GP: matched fit and posterior prediction",
        "float64; train 128×4→2, predict 32 points; one CPU thread",
        arguments.output_dir / "exact_gp_workloads.png",
    )
    plot_workload(
        rows,
        "mlp",
        ("forward", "vjp"),
        "MLP products: matched forward pass and reverse product",
        "float64; batch 512, 16–32–4 tanh network; one CPU thread",
        arguments.output_dir / "mlp_workloads.png",
    )


if __name__ == "__main__":
    main()
