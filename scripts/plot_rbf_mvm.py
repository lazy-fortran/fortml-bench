from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("png_path", type=Path)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.csv_path.open()))
    groups = {}
    for row in rows:
        if row["status"] == "pass":
            key = (row["backend"], row["device"], row["residency"])
            groups[key] = float(row["seconds_per_mvm"]) * 1.0e3
    labels = []
    values = []
    for key, timing in sorted(groups.items()):
        labels.append(" / ".join(key))
        values.append(timing)
    figure, axis = plt.subplots(figsize=(11, 7))
    bars = axis.barh(labels, values)
    axis.bar_label(bars, fmt="%.3g", padding=4)
    axis.set_xlabel("milliseconds per MVM (log scale)")
    axis.set_xscale("log")
    axis.set_title("RBF matrix-vector product (lower is better)")
    axis.grid(axis="x", alpha=0.25)
    axis.invert_yaxis()
    figure.tight_layout()
    args.png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.png_path, dpi=180)
    print(args.png_path)


if __name__ == "__main__":
    main()
