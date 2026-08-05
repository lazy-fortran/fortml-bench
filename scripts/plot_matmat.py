from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def plot_device(rows: list[dict[str, str]], device: str, output: Path) -> None:
    groups: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if (
            row["device"] != device
            or row["residency"] != "resident"
            or row["status"] != "pass"
        ):
            continue
        groups[(row["backend"], row["device"])].append(
            (int(row["rhs"]), 1.0e3 * float(row["seconds_per_operation"]))
        )
    if not groups:
        return
    figure, axis = plt.subplots(figsize=(8, 5))
    for backend, device in sorted(groups):
        values = sorted(groups[(backend, device)])
        axis.plot(
            [x for x, _ in values],
            [y for _, y in values],
            marker="o",
            label=f"{backend} / {device}",
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("right-hand sides")
    axis.set_ylabel("milliseconds per resident matmat")
    axis.set_title(f"RBF matrix-matrix operator on {device} (lower is better)")
    axis.grid(which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_prefix", type=Path)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.csv_path.open(newline="")))
    for device in ("cpu", "cuda"):
        plot_device(
            rows,
            device,
            args.output_prefix.with_name(f"{args.output_prefix.name}_{device}.png"),
        )


if __name__ == "__main__":
    main()
