from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

COLORS = {
    "fortml_composite": "#0072B2",
    "keops": "#D55E00",
    "gpytorch_keops": "#009E73",
    "pytorch_dense": "#CC79A7",
}
STYLES = {
    "fortml_composite": ("-", "o"),
    "keops": ("--", "s"),
    "gpytorch_keops": ("-.", "^"),
    "pytorch_dense": (":", "D"),
}
LABELS = {
    "fortml_composite": "FortML static composite",
    "keops": "KeOps",
    "gpytorch_keops": "GPyTorch + KeOps",
    "pytorch_dense": "PyTorch dense",
}


def plot_device(rows: list[dict[str, str]], device: str, output: Path) -> None:
    groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if row["device"] != device or row["status"] != "pass":
            continue
        groups[row["backend"]].append(
            (int(row["n_samples"]), 1.0e3 * float(row["seconds_per_mvm"]))
        )
    if not groups:
        return
    figure, axis = plt.subplots(figsize=(9, 6))
    for backend in ("fortml_composite", "keops", "gpytorch_keops", "pytorch_dense"):
        if backend not in groups:
            continue
        values = sorted(groups[backend])
        x_values, y_values = zip(*values)
        linestyle, marker = STYLES[backend]
        axis.plot(
            x_values,
            y_values,
            color=COLORS[backend],
            linestyle=linestyle,
            marker=marker,
            linewidth=2.0,
            markersize=6,
            label=LABELS[backend],
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("number of samples")
    axis.set_ylabel("milliseconds per MVM")
    axis.set_title(f"RBF + constant MVM scaling on {device} (lower is better)")
    axis.grid(which="both", alpha=0.25)
    # Keep the key inside the raster: this is reliable for both the sparse
    # high-range CPU plot and the denser CUDA plot when shared externally.
    axis.legend(loc="upper left", framealpha=0.9)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_prefix", type=Path)
    args = parser.parse_args()
    with args.csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for device in ("cpu", "cuda"):
        plot_device(
            rows,
            device,
            args.output_prefix.with_name(f"{args.output_prefix.name}_{device}.png"),
        )


if __name__ == "__main__":
    main()
