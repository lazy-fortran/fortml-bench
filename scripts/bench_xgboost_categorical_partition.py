#!/usr/bin/env python3
"""Correctness-gated exhaustive small-cardinality XGBoost benchmark.

The NumPy oracle enumerates every nontrivial subset of four integer category
codes and computes the one-tree squared-loss Newton gain independently. CUDA
is recorded as unavailable because the categorical tree path has no resident
CUDA kernel.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


N_SAMPLES = 256
N_FEATURES = 2
N_ESTIMATORS = 1
TOLERANCE = 2.0e-12
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_estimators", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def exhaustive_oracle(categories: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Enumerate canonical subsets and return the best Newton stump values."""
    codes = np.sort(np.unique(categories).astype(np.int64))
    base = float(np.mean(target))
    gradient = base - target
    hessian = np.ones_like(target)
    best_gain = 0.0
    best_left: np.ndarray | None = None
    if codes.size > 1:
        # The first code is always left; subsets that put every code left are
        # omitted, so complements are represented exactly once.
        for subset in range(2 ** (codes.size - 1) - 1):
            left = np.array(
                [codes[0]]
                + [codes[index] for index in range(1, codes.size)
                   if subset & (1 << (index - 1))],
                dtype=np.int64,
            )
            mask = np.isin(categories.astype(np.int64), left)
            right = ~mask
            left_hessian = float(np.sum(hessian[mask]))
            right_hessian = float(np.sum(hessian[right]))
            left_gradient = float(np.sum(gradient[mask]))
            right_gradient = float(np.sum(gradient[right]))
            gain = 0.5 * (
                left_gradient * left_gradient / left_hessian
                + right_gradient * right_gradient / right_hessian
                - float(np.sum(gradient)) ** 2 / float(np.sum(hessian))
            )
            if gain > best_gain:
                best_gain = gain
                best_left = left
    expected = np.full(target.shape, base, dtype=np.float64)
    if best_left is not None:
        mask = np.isin(categories.astype(np.int64), best_left)
        expected[mask] = base - float(np.sum(gradient[mask])) / float(np.sum(hessian[mask]))
        expected[~mask] = base - float(np.sum(gradient[~mask])) / float(np.sum(hessian[~mask]))
    return expected


def parse(stdout: str) -> tuple[np.ndarray, float, list[str]]:
    values: np.ndarray | None = None
    fit: list[str] | None = None
    predict_seconds: float | None = None
    for line in stdout.splitlines():
        if line.startswith("xgb_categorical_partition_values "):
            values = np.asarray([float(item) for item in line.split()[1:]], dtype=np.float64)
        elif line.startswith("xgb_categorical_partition_fit,"):
            fit = line.split(",")
        elif line.startswith("xgb_categorical_partition_predict_seconds "):
            predict_seconds = float(line.split()[1])
    if values is None or values.size != N_SAMPLES or fit is None or predict_seconds is None:
        raise RuntimeError("FortML app omitted exhaustive categorical records")
    return values, predict_seconds, fit


def run_fortran(fortml: Path, details: dict[str, str]) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_xgboost_categorical_partition"],
        cwd=fortml, env=environment, check=True, capture_output=True, text=True,
    )
    values, predict_seconds, fit = parse(completed.stdout)
    categories = np.repeat(np.arange(4, dtype=np.float64), N_SAMPLES // 4)
    target = np.where(np.isin(categories, [0.0, 2.0]), 0.0, 4.0)
    expected = exhaustive_oracle(categories, target)
    error = float(np.max(np.abs(values - expected)))
    if error > TOLERANCE:
        raise RuntimeError(f"exhaustive categorical error {error:.3e}")
    if int(fit[5]) != 3:
        raise RuntimeError("exhaustive categorical fit did not produce one split")
    if float(fit[6]) > TOLERANCE:
        raise RuntimeError("FortML exhaustive categorical diagnostic disagrees with oracle")
    return [
        row(details, workload="xgboost_categorical_partition", phase="fit",
            backend="fortml", device="cpu", status="pass", n_samples=N_SAMPLES,
            n_features=N_FEATURES, n_estimators=N_ESTIMATORS,
            seconds_per_operation=float(fit[4]), metric="max_abs_error", value=error,
            max_abs_error=error, oracle="independent NumPy exhaustive subset Newton oracle",
            notes="four-code integer feature; canonical subsets and complement suppression"),
        row(details, workload="xgboost_categorical_partition", phase="predict",
            backend="fortml", device="cpu", status="pass", n_samples=N_SAMPLES,
            n_features=N_FEATURES, n_estimators=N_ESTIMATORS,
            seconds_per_operation=predict_seconds, metric="max_abs_error", value=error,
            max_abs_error=error, oracle="independent NumPy exhaustive subset Newton oracle",
            notes="repeated fixed-tree prediction timing"),
    ]


def unavailable(details: dict[str, str]) -> list[dict[str, Any]]:
    return [row(
        details, workload="xgboost_categorical_partition", phase="capability_check",
        backend="fortml", device="cuda", status="unavailable", n_samples=N_SAMPLES,
        n_features=N_FEATURES, n_estimators=N_ESTIMATORS,
        metric="resident_categorical_tree", value="FORTNUM_NOT_IMPLEMENTED",
        max_abs_error=0.0,
        oracle="declared device contract",
        notes="no resident CUDA categorical-tree kernel is linked; typed refusal",
    )]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/xgboost_categorical_partition.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    details = metadata(root, fortml, args.output.resolve())
    rows = run_fortran(fortml, details) + unavailable(details)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
