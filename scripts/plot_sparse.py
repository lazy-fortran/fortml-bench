from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def plot_backend(ax, rows, backend, label, color, marker, linestyle):
    selected = [
        row for row in rows
        if row["backend"] == backend and row["residency"] == "resident"
        and row["status"] == "pass"
    ]
    selected.sort(key=lambda row: int(row["n_samples"]))
    if not selected:
        return
    ax.plot(
        [int(row["n_samples"]) for row in selected],
        [float(row["seconds_per_operation"]) for row in selected],
        label=label,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=2.0,
        markersize=6,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.csv_path)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = {"fortml_sparse": "#0072B2", "keops": "#D55E00", "pytorch_dense": "#009E73"}
    markers = {"fortml_sparse": "o", "keops": "s", "pytorch_dense": "^"}
    linestyles = {"fortml_sparse": "-", "keops": "--", "pytorch_dense": ":"}
    labels = {
        "fortml_sparse": "FortML sparse CSR",
        "keops": "KeOps dense-pair reduction",
        "pytorch_dense": "PyTorch dense",
    }
    for ax, device, title in zip(
        axes, ("cpu", "cuda"), ("CPU, one thread", "RTX 5060 Ti CUDA")
    ):
        device_rows = [row for row in rows if row["device"] == device]
        for backend in ("fortml_sparse", "keops", "pytorch_dense"):
            plot_backend(
                ax,
                device_rows,
                backend,
                labels[backend],
                colors[backend],
                markers[backend],
                linestyles[backend],
            )
        oom = sorted(
            int(row["n_samples"])
            for row in device_rows
            if row["backend"] == "pytorch_dense"
            and row["status"] == "oom"
        )
        if oom:
            ax.annotate(
                "PyTorch OOM at N≥" + str(oom[0]),
                xy=(oom[0], 1.0e-2),
                xytext=(0.98, 0.08),
                textcoords="axes fraction",
                ha="right",
                color=colors["pytorch_dense"],
                fontsize=9,
                arrowprops={"arrowstyle": "->", "color": colors["pytorch_dense"]},
            )
        ax.set_title(title)
        ax.set_xlabel("samples N")
        ax.set_ylabel("seconds per 4-RHS product")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Compact-support Wendland C2 product: float64, radius 8, four RHS\n"
        "Resident curves; setup excluded, correctness checked against row-wise oracle"
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_path, dpi=180)
    print(args.output_path)


if __name__ == "__main__":
    main()
