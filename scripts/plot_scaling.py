from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def plot_device(rows: list[dict[str, str]], device: str, output: Path) -> None:
    groups: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if row["device"] != device or row["status"] != "pass":
            continue
        groups[(row["backend"], row["residency"])].append(
            (int(row["n_samples"]), 1.0e3 * float(row["seconds_per_mvm"]))
        )
    if not groups:
        return
    figure, axis = plt.subplots(figsize=(9, 6))
    for key, values in sorted(groups.items()):
        values.sort()
        x_values, y_values = zip(*values)
        axis.plot(x_values, y_values, marker="o", label=" / ".join(key))
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("number of samples")
    axis.set_ylabel("milliseconds per MVM")
    axis.set_title(f"RBF MVM scaling on {device} (lower is better)")
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
    args = parser.parse_args()
    with args.csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for device in ("cpu", "cuda"):
        plot_device(rows, device, args.output_prefix.with_name(
            f"{args.output_prefix.name}_{device}.png"
        ))


if __name__ == "__main__":
    main()
