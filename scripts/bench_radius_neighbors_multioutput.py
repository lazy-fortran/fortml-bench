#!/usr/bin/env python3
"""Correctness-gated benchmark for multi-output radius-neighbor regression."""

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


N_SAMPLES, N_FEATURES, N_OUTPUTS, N_QUERY = 320, 3, 2, 7
RADIUS = 0.4
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_outputs", "n_query", "seconds_per_operation",
    "accuracy", "max_abs_error", "oracle", "python_version", "numpy_version",
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


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = np.arange(1, N_SAMPLES + 1, dtype=np.float64)
    x = np.column_stack((
        -2.0 + 4.0 * np.mod(index - 1.0, 80.0) / 79.0,
        np.sin(0.13 * index),
        np.cos(0.07 * index),
    ))
    targets = np.column_stack((x[:, 0] + 0.1 * x[:, 1], x[:, 0] ** 2 + x[:, 2]))
    query = np.column_stack((
        [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        np.zeros(7), np.ones(7),
    ))
    return x, targets, query


def oracle(x: np.ndarray, targets: np.ndarray, query: np.ndarray) -> np.ndarray:
    predictions = np.zeros((len(query), targets.shape[1]), dtype=np.float64)
    radius_squared = RADIUS * RADIUS
    for row, point in enumerate(query):
        distances = np.sum((x - point) ** 2, axis=1)
        selected = distances <= radius_squared
        if np.any(selected):
            predictions[row] = np.mean(targets[selected], axis=0)
    return predictions


def read_app(stdout: str) -> tuple[np.ndarray, dict[str, float]]:
    predictions = np.full((N_QUERY, N_OUTPUTS), np.nan, dtype=np.float64)
    timing: dict[str, float] = {}
    cuda = False
    for line in stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if not fields or not fields[0]:
            continue
        if fields[0] in {
            "radius_multioutput_fit_seconds",
            "radius_multioutput_predict_seconds",
            "radius_multioutput_max_abs_prediction",
        }:
            timing[fields[0]] = float(fields[1])
        elif fields[0] == "radius_multioutput_prediction":
            row, output = int(fields[1]) - 1, int(fields[2]) - 1
            predictions[row, output] = float(fields[3])
        elif fields[0] == "radius_multioutput_cuda" and fields[1] == "unavailable":
            cuda = True
    if np.isnan(predictions).any():
        raise RuntimeError("release app omitted a multi-output radius prediction")
    if not cuda:
        raise RuntimeError("release app omitted its typed CUDA capability row")
    return predictions, timing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/radius_neighbors_multioutput.csv"))
    parser.add_argument("--no-build", action="store_true",
                        help="reuse a previously built release app")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    x, targets, query = fixture()
    expected = oracle(x, targets, query)
    metadata = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (args.output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }

    def row(**values: object) -> dict[str, str]:
        result = {field: "" for field in FIELDS}
        result.update({
            "workload": "radius_neighbors_multioutput", "backend": "fortml",
            "device": "cpu", "status": "pass", "n_samples": str(N_SAMPLES),
            "n_features": str(N_FEATURES), "n_outputs": str(N_OUTPUTS),
            "n_query": str(N_QUERY),
            "oracle": "independent NumPy uniform multi-output radius oracle", **metadata,
        })
        result.update({key: str(value) for key, value in values.items()})
        return result

    rows: list[dict[str, str]] = []
    started = time.perf_counter()
    for _ in range(128):
        oracle(x, targets, query)
    oracle_seconds = (time.perf_counter() - started) / 128.0
    rows.append(row(phase="fit_predict", backend="numpy_oracle",
                    seconds_per_operation=oracle_seconds, max_abs_error="0.0"))

    if not args.no_build:
        subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, check=True)
    with tempfile.TemporaryDirectory(dir="/mnt/storage/code/lazy-fortran/fortml/build"):
        completed = subprocess.run(
            ["fo", "exec", "--no-build", "fortml_bench_radius_neighbors_multioutput"],
            cwd=fortml, capture_output=True, text=True, check=True,
        )
    actual, timing = read_app(completed.stdout)
    error = float(np.max(np.abs(actual - expected)))
    if error > 3.0e-12:
        raise RuntimeError(f"multi-output radius oracle mismatch: {error:.3e}")
    reported_max = timing["radius_multioutput_max_abs_prediction"]
    if abs(reported_max - float(np.max(np.abs(expected)))) > 3.0e-12:
        raise RuntimeError("release app max-absolute prediction summary is inconsistent")
    rows.append(row(phase="fit", seconds_per_operation=timing[
        "radius_multioutput_fit_seconds"], accuracy=1.0,
        max_abs_error=error, notes="complete multi-output prediction oracle passed"))
    rows.append(row(phase="predict", seconds_per_operation=timing[
        "radius_multioutput_predict_seconds"], accuracy=1.0,
        max_abs_error=error, notes="uniform CPU reduction across two target columns"))
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
