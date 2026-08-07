#!/usr/bin/env python3
"""Correctness-gated dense k-nearest-neighbor benchmark.

The NumPy lane is the behavioral oracle for sorted integer classes, stable
distance ties, uniform votes, and inverse-distance votes.  FortML timings are
accepted only after both probability sums and predictions agree with that
oracle; derivative refusals remain part of the documented estimator contract.
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


N_TRAIN = 96
N_QUERY = 48
N_FEATURES = 3
N_CLASSES = 3
N_NEIGHBORS = 7
REPETITIONS = 32
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_train", "n_query",
    "n_features", "n_neighbors", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version", "fortml_revision",
    "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = np.empty((N_TRAIN, N_FEATURES), dtype=np.float64)
    labels = np.empty(N_TRAIN, dtype=np.int64)
    query = np.empty((N_QUERY, N_FEATURES), dtype=np.float64)
    query_labels = np.empty(N_QUERY, dtype=np.int64)
    for i in range(1, N_TRAIN + 1):
        class_index = 1 + (i - 1) % N_CLASSES
        labels[i - 1] = 10 * class_index - 7
        for j in range(1, N_FEATURES + 1):
            train[i - 1, j - 1] = (1.7 * (class_index - 2)
                + 0.13 * np.sin(0.17 * i * j)
                + 0.04 * np.cos(0.11 * (i + 2 * j)))
    for i in range(1, N_QUERY + 1):
        class_index = 1 + (i + 1) % N_CLASSES
        query_labels[i - 1] = 10 * class_index - 7
        for j in range(1, N_FEATURES + 1):
            query[i - 1, j - 1] = (1.7 * (class_index - 2)
                + 0.09 * np.cos(0.19 * i * j)
                + 0.02 * np.sin(0.07 * (i + j)))
    return train, labels, query, query_labels


def oracle(weights: str) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    train, labels, query, query_labels = fixture()
    classes = np.unique(labels)
    probabilities = np.zeros((N_QUERY, N_CLASSES), dtype=np.float64)
    for row, point in enumerate(query):
        distances = np.sum((train - point) ** 2, axis=1)
        order = np.lexsort((np.arange(N_TRAIN), distances))[:N_NEIGHBORS]
        selected = distances[order]
        if weights == "uniform":
            vote = np.ones(N_NEIGHBORS)
        elif np.any(selected == 0.0):
            vote = (selected == 0.0).astype(np.float64)
        else:
            vote = 1.0 / np.sqrt(selected)
        for index, weight in zip(order, vote):
            vote_class = np.searchsorted(classes, labels[index])
            probabilities[row, vote_class] += weight
        probabilities[row] /= np.sum(probabilities[row])
    prediction = classes[np.argmax(probabilities, axis=1)]
    accuracy = float(np.mean(prediction == query_labels))
    return probabilities, prediction, accuracy, float(np.sum(probabilities**2)), float(np.sum(prediction))


def base(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    row.update(details)
    row.update({"device": "cpu", "n_train": N_TRAIN, "n_query": N_QUERY,
                "n_features": N_FEATURES, "n_neighbors": N_NEIGHBORS})
    row.update(values)
    return row


def run_fortml(fortml: Path, target: str, details: dict[str, str],
               expected: dict[str, tuple[np.ndarray, np.ndarray, float, float, float]]) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    source = fortml / "app" / f"{target}.f90"
    if not source.is_file():
        return [base(details, workload="knn", phase="predict", backend="fortml", status="unavailable",
                     oracle="FortML release-app protocol", notes=f"release target source is absent: {source.name}")]
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
                           capture_output=True, text=True)
    if build.returncode != 0:
        return [base(details, workload="knn", phase="predict", backend="fortml", status="unavailable",
                     oracle="FortML release-app protocol", notes="fo build failed")]
    run = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml, env=environment,
                         capture_output=True, text=True)
    if run.returncode != 0:
        return [base(details, workload="knn", phase="predict", backend="fortml", status="unavailable",
                     oracle="FortML release-app protocol", notes="release target execution failed")]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in run.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if not fields or not fields[0].startswith("knn_"):
            continue
        variant = fields[0].removeprefix("knn_")
        if variant not in expected or len(fields) != 9:
            continue
        seen.add(variant)
        probabilities, prediction, accuracy, probability_norm, prediction_sum = expected[variant]
        actual_accuracy = float(fields[5])
        actual_sum = float(fields[6])
        actual_norm = float(fields[7])
        actual_prediction_sum = float(fields[8])
        error = max(abs(actual_accuracy - accuracy), abs(actual_sum - float(N_QUERY)),
                    abs(actual_norm - probability_norm), abs(actual_prediction_sum - prediction_sum))
        if error > 2.0e-13:
            raise RuntimeError(f"FortML kNN {variant} oracle mismatch: {error:.3e}")
        rows.append(base(details, workload=f"knn_{variant}", phase="predict", backend="fortml",
                         status="pass", seconds_per_operation=float(fields[4]), metric="accuracy",
                         value=actual_accuracy, max_abs_error=error,
                         oracle="independent NumPy stable-neighbor vote oracle",
                         notes="probability_sum checked against N_QUERY"))
    for variant in sorted(set(expected) - seen):
        rows.append(base(details, workload=f"knn_{variant}", phase="predict", backend="fortml",
                         status="unavailable", oracle="FortML release-app protocol",
                         notes="release app emitted no parseable row"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/knn.csv"))
    parser.add_argument("--target", default="fortml_bench_knn")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    details = metadata(root, fortml, output)
    rows: list[dict[str, Any]] = []
    expected: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for variant in ("uniform", "distance"):
        started = __import__("time").perf_counter()
        probabilities, prediction, accuracy, probability_norm, prediction_sum = oracle(variant)
        elapsed = (__import__("time").perf_counter() - started) / REPETITIONS
        expected[variant] = (probabilities, prediction, accuracy, probability_norm, prediction_sum)
        rows.append(base(details, workload=f"knn_{variant}", phase="predict", backend="numpy_oracle",
                         status="pass", seconds_per_operation=elapsed, metric="accuracy", value=accuracy,
                         max_abs_error=0.0, oracle="independent stable lexsort and vote recurrence",
                         notes="sorted integer classes; tie order is original training row"))
    if args.skip_fortml:
        rows.extend(base(details, workload=f"knn_{variant}", phase="predict", backend="fortml", status="skipped",
                         oracle="FortML release-app protocol", notes="--skip-fortml") for variant in expected)
    else:
        rows.extend(run_fortml(fortml, args.target, details, expected))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
