#!/usr/bin/env python3
"""Correctness-gated benchmark for dense scalar radius-neighbor regression."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np


N_SAMPLES, N_FEATURES, N_QUERY = 96, 2, 6
RADIUS = 0.38
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_query", "seconds_per_operation", "accuracy",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    ignored_names = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            pass
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    )
    dirty = any(line[3:].strip() not in ignored_names for line in status.splitlines())
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    index = np.arange(1, N_SAMPLES + 1, dtype=np.float64)
    a = -1.2 + 2.4 * (index - 1.0) / (N_SAMPLES - 1.0)
    b = np.sin(0.19 * index)
    x = np.column_stack((a, b))
    targets = 1.4 * a - 0.7 * b + 0.2 * a * a
    weights = 0.8 + 0.4 * (np.mod(index, 5.0) / 4.0)
    query = np.array(
        ((-1.05, 0.0), (-0.15, 0.35), (0.25, -0.45),
         (0.85, 0.15), (1.18, -0.2), (0.0, 1.4)), dtype=np.float64
    )
    return x, targets, weights, query


def oracle(x: np.ndarray, targets: np.ndarray, weights: np.ndarray,
           query: np.ndarray) -> np.ndarray:
    predictions = np.zeros(len(query), dtype=np.float64)
    radius_squared = RADIUS * RADIUS
    for row, point in enumerate(query):
        distances = np.sum((x - point) ** 2, axis=1)
        selected = distances <= radius_squared
        if not np.any(selected):
            continue
        selected_distances = distances[selected]
        exact = np.any(selected_distances == 0.0)
        local_weights = weights[selected].copy()
        if exact:
            local_weights[selected_distances != 0.0] = 0.0
        else:
            local_weights /= np.sqrt(selected_distances)
        predictions[row] = np.sum(local_weights * targets[selected]) / np.sum(local_weights)
    return predictions


def read_app(path: Path) -> np.ndarray:
    predictions = np.full(N_QUERY, np.nan, dtype=np.float64)
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            if record["quantity"] != "prediction":
                raise RuntimeError(f"unknown quantity {record['quantity']!r}")
            row = int(record["row"]) - 1
            predictions[row] = float(record["value"])
    if np.isnan(predictions).any():
        raise RuntimeError("release app omitted a radius-neighbor regression output")
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/radius_neighbors_regression.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    x, targets, weights, query = fixture()
    expected = oracle(x, targets, weights, query)
    metadata = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (args.output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }

    def row(**values: object) -> dict[str, str]:
        result = {field: "" for field in FIELDS}
        result.update({
            "workload": "radius_neighbors_regression", "backend": "fortml",
            "device": "cpu", "status": "pass", "n_samples": str(N_SAMPLES),
            "n_features": str(N_FEATURES), "n_query": str(N_QUERY),
            "oracle": "independent NumPy inverse-distance radius oracle", **metadata,
        })
        result.update({key: str(value) for key, value in values.items()})
        return result

    rows: list[dict[str, str]] = []
    started = time.perf_counter()
    for _ in range(128):
        oracle(x, targets, weights, query)
    oracle_seconds = (time.perf_counter() - started) / 128.0
    rows.append(row(phase="fit_predict", backend="numpy_oracle",
                    seconds_per_operation=oracle_seconds, max_abs_error="0.0"))

    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, check=True)
    environment = os.environ.copy()
    with tempfile.TemporaryDirectory(
        dir="/mnt/storage/code/lazy-fortran/fortml/build"
    ) as directory:
        output = Path(directory) / "radius_regression.csv"
        environment["FORTML_BENCH_RADIUS_REGRESSION_OUTPUT"] = str(output)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", "fortml_bench_radius_neighbors_regression"],
            cwd=fortml, env=environment, capture_output=True, text=True, check=True,
        )
        actual = read_app(output)
    error = float(np.max(np.abs(actual - expected)))
    if error > 3.0e-12:
        raise RuntimeError(f"radius-neighbor regression oracle mismatch: {error:.3e}")
    timing = {}
    for line in completed.stdout.splitlines():
        fields = line.strip().split(",")
        if fields and fields[0] in {
            "radius_neighbors_regression_fit", "radius_neighbors_regression_predict",
        }:
            timing[fields[0]] = float(fields[-1])
    rows.append(row(phase="fit", seconds_per_operation=timing[
        "radius_neighbors_regression_fit"], accuracy=1.0,
        max_abs_error=error, notes="complete scalar prediction oracle passed"))
    rows.append(row(phase="predict", seconds_per_operation=timing[
        "radius_neighbors_regression_predict"], accuracy=1.0,
        max_abs_error=error, notes="inverse-distance CPU lane"))
    rows.append(row(phase="predict", device="cuda", status="unavailable",
                    seconds_per_operation="", accuracy="", max_abs_error="",
                    oracle="typed_device_contract",
                    notes="no resident radius-search kernel; FORTNUM_NOT_IMPLEMENTED"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
