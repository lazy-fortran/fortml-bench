#!/usr/bin/env python3
"""Correctness-gated binary and one-vs-rest ROC-AUC benchmark.

The oracle is a small independent NumPy pairwise implementation with the same
documented half-credit tie rule; it does not call scikit-learn.  CUDA is a
typed unavailable row until a resident ranking/reduction kernel is linked.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path

import numpy as np

FIELDS = (
    "workload", "metric", "phase", "backend", "device", "status",
    "n_samples", "n_classes", "seconds_per_operation", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            pass
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        if line[3:].strip() not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def pair_auc(scores: np.ndarray, labels: np.ndarray, positive: int) -> float:
    positive_scores = scores[labels == positive]
    negative_scores = scores[labels != positive]
    if not positive_scores.size or not negative_scores.size:
        raise ValueError("degenerate support")
    comparison = positive_scores[:, None] - negative_scores[None, :]
    return float(np.mean((comparison > 0.0) + 0.5 * (comparison == 0.0)))


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    binary_scores = np.array([0.9, 0.8, 0.8, 0.2], dtype=np.float64)
    binary_labels = np.array([42, 42, -7, -7], dtype=np.int64)
    labels = np.array([-2, -2, 4, 4, 9, 9], dtype=np.int64)
    classes = np.array([-2, 4, 9], dtype=np.int64)
    scores = np.column_stack((
        [0.9, 0.7, 0.8, 0.2, 0.1, 0.3],
        [0.05, 0.1, 0.8, 0.7, 0.1, 0.2],
        [0.05, 0.1, 0.1, 0.1, 0.8, 0.7],
    ))
    return binary_scores, binary_labels, scores, labels


def parse(stdout: str) -> dict[str, float | str]:
    values: dict[str, float | str] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or not fields[0].startswith("roc_auc_"):
            continue
        key, raw = fields
        values[key] = raw if key.endswith("_cuda") else float(raw)
    required = {
        "roc_auc_binary", "roc_auc_binary_error", "roc_auc_binary_seconds",
        "roc_auc_ovr", "roc_auc_ovr_error", "roc_auc_ovr_seconds", "roc_auc_cuda",
    }
    missing = required.difference(values)
    if missing:
        raise RuntimeError(f"FortML omitted ROC-AUC fields: {sorted(missing)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/roc_auc.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    ignored = tuple((root / "results" / name).resolve()
                    for name in ("multilabel_metrics.csv", "roc_auc.csv",
                                 "device_contracts.csv"))
    binary_scores, binary_labels, scores, labels = fixture()
    binary_expected = pair_auc(binary_scores, binary_labels, 42)
    per_class = np.array([pair_auc(scores[:, i], labels, c) for i, c in enumerate((-2, 4, 9))])
    macro_expected = float(np.mean(per_class))
    metadata = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml), "benchmark_revision": revision(root, (ignored,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(metadata); row.update(values); rows.append(row)

    add(workload="roc_auc", metric="binary", phase="oracle", backend="numpy_oracle",
        device="cpu", status="pass", n_samples=4, n_classes=2,
        value=binary_expected, max_abs_error=0.0,
        oracle="independent NumPy pairwise ordering with half-credit ties",
        notes="arbitrary integer labels [42,-7]")
    add(workload="roc_auc", metric="ovr_macro", phase="oracle", backend="numpy_oracle",
        device="cpu", status="pass", n_samples=6, n_classes=3, value=macro_expected,
        max_abs_error=0.0, oracle="independent NumPy one-vs-rest pairwise oracle",
        notes="macro of per-class values %s" % per_class.tolist())

    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, check=True)
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_roc_auc"], cwd=fortml,
        check=True, capture_output=True, text=True,
    )
    values = parse(completed.stdout)
    binary_error = float(values["roc_auc_binary_error"])
    macro_error = float(values["roc_auc_ovr_error"])
    if binary_error > 1e-13 or macro_error > 1e-13:
        raise RuntimeError(f"ROC-AUC oracle mismatch: binary={binary_error:.3e}, macro={macro_error:.3e}")
    if values["roc_auc_cuda"] != "unavailable":
        raise RuntimeError("ROC-AUC CUDA refusal row changed unexpectedly")
    add(workload="roc_auc", metric="binary", phase="predict", backend="fortml_cpu",
        device="cpu", status="pass", n_samples=4, n_classes=2,
        seconds_per_operation=values["roc_auc_binary_seconds"], value=float(values["roc_auc_binary"]),
        max_abs_error=binary_error, oracle="independent NumPy pairwise ordering with half-credit ties")
    add(workload="roc_auc", metric="ovr_macro", phase="predict", backend="fortml_cpu",
        device="cpu", status="pass", n_samples=6, n_classes=3,
        seconds_per_operation=values["roc_auc_ovr_seconds"], value=float(values["roc_auc_ovr"]),
        max_abs_error=macro_error, oracle="independent NumPy one-vs-rest pairwise oracle")
    add(workload="roc_auc", metric="binary+ovr", phase="predict", backend="fortml_cuda",
        device="cuda", status="unavailable", n_samples=6, n_classes=3,
        oracle="typed_device_contract", notes="no resident CUDA ranking/reduction kernel")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
