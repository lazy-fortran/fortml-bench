#!/usr/bin/env python3
"""Correctness-gated deterministic random-forest permutation benchmark."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


N_REPEATS = 24
PERMUTATION_SEED = 991
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_trees", "n_repeats", "feature", "seconds_per_operation",
    "metric", "value", "max_abs_error", "oracle", "python_version",
    "numpy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
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


def parse(stdout: str) -> dict[str, float | int | str]:
    records: dict[str, float | int | str] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields[:2] == ["rf_permutation", "metrics"]:
            if len(fields) != 16:
                raise RuntimeError(f"malformed permutation record: {line!r}")
            names = (
                "n_samples", "n_features", "n_trees", "n_repeats", "fit_seconds",
                "permutation_seconds", "baseline", "importance_1", "importance_2",
                "importance_3", "std_1", "std_2", "std_3", "oracle_correct",
            )
            integer_names = {"n_samples", "n_features", "n_trees", "n_repeats", "oracle_correct"}
            for name, value in zip(names, fields[2:], strict=True):
                records[name] = int(value) if name in integer_names else float(value)
        elif fields[:2] == ["rf_permutation", "cuda"]:
            records["cuda"] = fields[2]
    required = {
        "n_samples", "n_features", "n_trees", "n_repeats", "fit_seconds",
        "permutation_seconds", "baseline", "importance_1", "importance_2",
        "importance_3", "std_1", "std_2", "std_3", "oracle_correct", "cuda",
    }
    missing = required.difference(records)
    if missing:
        raise RuntimeError(f"release app omitted permutation metrics: {sorted(missing)}")
    return records


def oracle(n_samples: int, n_features: int, n_repeats: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Replay the threshold classifier and Park--Miller permutation stream in NumPy."""
    if n_features != 3:
        raise RuntimeError(f"unexpected fixture feature count: {n_features}")
    rows = np.arange(n_samples)
    x = np.empty((n_samples, n_features), dtype=np.float64)
    x[:, 0] = -2.0 + 4.0*np.mod(rows, 80)/79.0
    x[:, 1] = np.sin(0.17*(rows + 1))
    x[:, 2] = np.cos(0.11*(rows + 1))
    labels = np.where(x[:, 0] < -0.65, -3, np.where(x[:, 0] > 0.65, 11, 4))
    baseline = float(np.mean(labels == labels))
    values = np.empty((n_features, n_repeats), dtype=np.float64)
    state = int(seed)
    for feature in range(n_features):
        for repeat in range(n_repeats):
            permutation = np.arange(n_samples)
            for size in range(n_samples, 1, -1):
                state = (48271*state) % 2147483647
                index = state % size
                permutation[size - 1], permutation[index] = (
                    permutation[index], permutation[size - 1]
                )
            permuted = x.copy()
            permuted[:, feature] = x[permutation, feature]
            predicted = np.where(
                permuted[:, 0] < -0.65, -3,
                np.where(permuted[:, 0] > 0.65, 11, 4),
            )
            values[feature, repeat] = baseline - float(np.mean(predicted == labels))
    return np.mean(values, axis=1), np.std(values, axis=1)


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/random_forest_permutation.csv"))
    parser.add_argument("--target", default="fortml_bench_random_forest_permutation")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(["fo", "build", "--flag", "-O2"], cwd=fortml,
                   env=environment, check=True, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE, text=True)
    completed = subprocess.run(
        ["fo", "exec", "--no-build", args.target], cwd=fortml,
        env=environment, check=True, capture_output=True, text=True,
    )
    observed = parse(completed.stdout)
    n_samples = int(observed["n_samples"])
    n_features = int(observed["n_features"])
    n_trees = int(observed["n_trees"])
    n_repeats = int(observed["n_repeats"])
    expected_importance, expected_std = oracle(
        n_samples, n_features, n_repeats, PERMUTATION_SEED,
    )
    observed_importance = np.array(
        [float(observed[f"importance_{index}"]) for index in range(1, n_features + 1)],
    )
    observed_std = np.array(
        [float(observed[f"std_{index}"]) for index in range(1, n_features + 1)],
    )
    importance_error = float(np.max(np.abs(observed_importance - expected_importance)))
    std_error = float(np.max(np.abs(observed_std - expected_std)))
    if n_repeats != N_REPEATS or importance_error > 5.0e-13 or std_error > 5.0e-13:
        raise RuntimeError(
            f"permutation oracle mismatch: importance={importance_error:.3e}, "
            f"std={std_error:.3e}, repeats={n_repeats}"
        )
    if int(observed["oracle_correct"]) != n_samples or abs(float(observed["baseline"]) - 1.0) > 5.0e-13:
        raise RuntimeError("independent threshold baseline oracle mismatch")
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O2",
        "oracle": "independent NumPy threshold classifier and Park--Miller Fisher--Yates replay",
    }
    rows = [
        row(details, workload="random_forest_permutation", phase="fit", backend="fortml",
            device="cpu", status="pass", n_samples=n_samples, n_features=n_features,
            n_trees=n_trees, n_repeats=n_repeats,
            seconds_per_operation=observed["fit_seconds"], metric="baseline_accuracy",
            value=observed["baseline"], max_abs_error=abs(float(observed["baseline"]) - 1.0),
            notes="64-tree seeded bootstrap CART fixture"),
        row(details, workload="random_forest_permutation", phase="importance", backend="fortml",
            device="cpu", status="pass", n_samples=n_samples, n_features=n_features,
            n_trees=n_trees, n_repeats=n_repeats,
            seconds_per_operation=observed["permutation_seconds"], feature="all",
            metric="mean_absolute_importance_oracle_error", value=importance_error,
            max_abs_error=importance_error,
            notes="accuracy decrease; fitted routing fixed across repeats"),
        row(details, workload="random_forest_permutation", phase="dispersion", backend="fortml",
            device="cpu", status="pass", n_samples=n_samples, n_features=n_features,
            n_trees=n_trees, n_repeats=n_repeats, feature="all",
            metric="std_max_abs_oracle_error", value=std_error, max_abs_error=std_error,
            notes="population standard deviation across deterministic repeats"),
        row(details, workload="random_forest_permutation", phase="device_contract",
            backend="fortml", device="cuda", status="unavailable", n_samples=n_samples,
            n_features=n_features, n_trees=n_trees, n_repeats=n_repeats,
            feature="all", metric="api_surface", value=observed["cuda"],
            max_abs_error=0.0, oracle="typed CUDA refusal preserving all output buffers",
            notes="no resident CUDA permutation kernel"),
    ]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
