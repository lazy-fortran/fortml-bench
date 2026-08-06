from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def plot_device(
    rows: list[dict[str, str]], device: str, output: Path, include_setup: bool
) -> None:
    groups: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if row["device"] != device or row["status"] != "pass":
            continue
        if not row.get("seconds_per_solve"):
            continue
        seconds = float(row["seconds_per_solve"])
        if include_setup and row.get("setup_seconds"):
            seconds += float(row["setup_seconds"])
        groups[(row["backend"], row["residency"])].append(
            (int(row["n_samples"]), 1.0e3 * seconds)
        )
    if not groups:
        return
    figure, axis = plt.subplots(figsize=(9, 6))
    colors = plt.get_cmap("cividis")([0.1, 0.35, 0.6, 0.85])
    markers = ("o", "s", "^", "D")
    linestyles = ("-", "--", "-.", ":")
    for index, (key, values) in enumerate(sorted(groups.items())):
        values.sort()
        x_values, y_values = zip(*values)
        axis.plot(
            x_values,
            y_values,
            color=colors[index % len(colors)],
            linestyle=linestyles[index % len(linestyles)],
            marker=markers[index % len(markers)],
            label=" / ".join(key),
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("number of samples")
    ylabel = (
        "milliseconds per first multi-RHS CG solve"
        if include_setup
        else "milliseconds per multi-RHS CG solve"
    )
    axis.set_ylabel(ylabel)
    title = "RBF multi-RHS matrix-free CG scaling on " + device
    if include_setup:
        title += " (setup plus solve)"
    axis.set_title(title + " (lower is better)")
    axis.grid(which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--include-setup", action="store_true")
    args = parser.parse_args()
    with args.csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for device in ("cpu", "cuda"):
        plot_device(
            rows,
            device,
            args.output_prefix.with_name(f"{args.output_prefix.name}_{device}.png"),
            args.include_setup,
        )


if __name__ == "__main__":
    main()
