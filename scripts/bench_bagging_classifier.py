#!/usr/bin/env python3
"""Correctness-gated seeded CART bagging benchmark."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


N_QUERY = 6
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_query", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def expected_query_labels() -> np.ndarray:
    query = np.array([-1.5, -0.7, -0.1, 0.1, 0.7, 1.5], dtype=np.float64)
    return np.where(query < -0.65, -3, np.where(query > 0.65, 11, 4))


def parse(stdout: str) -> dict[str, float | int | str]:
    values: dict[str, float | int | str] = {}
    for line in stdout.splitlines():
        if line.startswith("bagging_fit_seconds,"):
            values["fit_seconds"] = float(line.split(",", 1)[1])
        elif line.startswith("bagging_predict_seconds,"):
            values["predict_seconds"] = float(line.split(",", 1)[1])
        elif line.startswith("bagging_probability_sum_error,"):
            values["simplex_error"] = float(line.split(",", 1)[1])
        elif line.startswith("bagging_query_correct,"):
            values["query_correct"] = int(line.split(",", 1)[1])
        elif line.startswith("bagging_cuda,"):
            values["cuda"] = line.split(",", 1)[1].strip()
    required = {"fit_seconds", "predict_seconds", "simplex_error", "query_correct", "cuda"}
    if set(values) != required:
        raise RuntimeError(f"release app omitted bagging metrics: {sorted(values)}")
    return values


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/bagging_classifier.csv"))
    parser.add_argument("--target", default="fortml_bench_bagging_classifier")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (args.output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                   env=environment, check=True, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE, text=True)
    completed = subprocess.run(
        ["fo", "exec", "--no-build", args.target], cwd=fortml,
        env=environment, check=True, capture_output=True, text=True,
    )
    observed = parse(completed.stdout)
    expected = expected_query_labels()
    expected_correct = int(expected.size)
    simplex_error = float(observed["simplex_error"])
    query_correct = int(observed["query_correct"])
    if simplex_error > 5.0e-13 or query_correct != expected_correct:
        raise RuntimeError(
            f"bagging oracle mismatch: simplex={simplex_error:.3e}, "
            f"query_correct={query_correct}, expected={expected_correct}"
        )
    rows = [
        row(details, workload="bagging_classifier", phase="fit", backend="fortml",
            device="cpu", status="pass", n_samples=240, n_features=3,
            n_query=N_QUERY, seconds_per_operation=observed["fit_seconds"],
            metric="query_label_count", value=query_correct,
            max_abs_error=0.0,
            oracle="independent NumPy threshold labels and probability simplex",
            notes="32-tree seeded bootstrap CART fixture"),
        row(details, workload="bagging_classifier", phase="predict", backend="fortml",
            device="cpu", status="pass", n_samples=240, n_features=3,
            n_query=N_QUERY, seconds_per_operation=observed["predict_seconds"],
            metric="probability_simplex_max_abs_error", value=simplex_error,
            max_abs_error=simplex_error,
            oracle="independent NumPy simplex and cluster-label oracle",
            notes="six query labels all agree"),
        row(details, workload="bagging_classifier", phase="device_contract",
            backend="fortml", device="cuda", status="unavailable", n_samples=240,
            n_features=3, n_query=N_QUERY, metric="api_surface", value=observed["cuda"],
            max_abs_error=0.0, oracle="typed device capability contract",
            notes="no resident bagging CUDA ensemble kernel; typed refusal"),
    ]
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
