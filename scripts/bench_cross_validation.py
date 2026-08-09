#!/usr/bin/env python3
"""Correctness-gated weighted cross-validation scoring benchmark.

The NumPy path independently computes K-fold test means, weighted reduction,
and the oriented FortOpt objective.  FortML output is accepted only after all
fold diagnostics and products agree.  CUDA is a typed control-plane refusal.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


N_SAMPLES = 17
N_SPLITS = 4
PARAMETER = 0.25
REPETITIONS = 512
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_splits", "aggregation", "repetitions", "seconds_per_operation",
    "value", "gradient", "objective_value", "max_abs_error", "oracle",
    "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names: set[str] = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            pass
    dirty = any(
        line[3:].split(" -> ")[-1].strip() not in ignored_names
        for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True,
        ).splitlines()
    )
    return head + ("+dirty" if dirty else "")


def folds() -> list[np.ndarray]:
    sizes = [N_SAMPLES // N_SPLITS + int(index < N_SAMPLES % N_SPLITS)
             for index in range(N_SPLITS)]
    result: list[np.ndarray] = []
    start = 1
    for size in sizes:
        result.append(np.arange(start, start + size, dtype=np.int64))
        start += size
    return result


def independent_products() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    test_indices = folds()
    values = np.asarray([PARAMETER + np.mean(test) for test in test_indices], dtype=float)
    weights = np.asarray([test.size for test in test_indices], dtype=float)
    gradients = np.ones(N_SPLITS, dtype=float)
    value = float(np.sum(values * weights) / np.sum(weights))
    objective = -value
    return values, gradients, weights, value, objective


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "workload": "cross_validation", "phase": "weighted_mean", "backend": "",
        "device": "cpu", "status": "", "n_samples": N_SAMPLES,
        "n_splits": N_SPLITS, "aggregation": "weighted_mean",
        "repetitions": REPETITIONS, "seconds_per_operation": "", "value": "",
        "gradient": "", "objective_value": "", "max_abs_error": "",
        "oracle": "", "notes": "",
    }
    result.update(details)
    result.update(values)
    return result


def run_numpy(details: dict[str, str]) -> list[dict[str, Any]]:
    expected_values, expected_gradients, expected_weights, expected_value, expected_objective = \
        independent_products()
    started = time.perf_counter()
    for _ in range(REPETITIONS):
        values, gradients, weights, value, objective = independent_products()
        if (not np.array_equal(values, expected_values) or
                not np.array_equal(gradients, expected_gradients) or
                not np.array_equal(weights, expected_weights) or
                value != expected_value or objective != expected_objective):
            raise RuntimeError("NumPy cross-validation oracle is not self-consistent")
    seconds = (time.perf_counter() - started) / REPETITIONS
    return [row(
        details, backend="numpy_oracle", status="pass", seconds_per_operation=seconds,
        value=expected_value, gradient=1.0, objective_value=expected_objective,
        max_abs_error=0.0,
        oracle="independent K-fold fold means and weighted reduction",
        notes="hand-computed test-fold means, weights, and maximize/minimize orientation",
    )]


def read_fold_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values: list[float] = []
    gradients: list[float] = []
    weights: list[float] = []
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            values.append(float(record["score"]))
            gradients.append(float(record["gradient"]))
            weights.append(float(record["weight"]))
    return (np.asarray(values), np.asarray(gradients), np.asarray(weights))


def run_fortml(fortml: Path, details: dict[str, str]) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"),
                        "OMP_NUM_THREADS": "1"})
    build = subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
        capture_output=True, text=True,
    )
    if build.returncode:
        raise RuntimeError(build.stderr.strip() or build.stdout.strip() or "fo build failed")
    expected_values, expected_gradients, expected_weights, expected_value, expected_objective = \
        independent_products()
    with tempfile.TemporaryDirectory(dir="/mnt/storage", prefix="fortml-cross-validation-") as directory:
        oracle_path = Path(directory) / "cross_validation.csv"
        check_environment = dict(environment)
        check_environment["FORTML_BENCH_CROSS_VALIDATION_ORACLE"] = str(oracle_path)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", "fortml_bench_cross_validation"],
            cwd=fortml, env=check_environment, capture_output=True, text=True,
        )
        if completed.returncode or not oracle_path.is_file():
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or
                               "release target did not emit its complete oracle")
        values, gradients, weights = read_fold_csv(oracle_path)
        if not (np.array_equal(weights, expected_weights) and
                np.max(np.abs(values - expected_values)) <= 2.0e-14 and
                np.max(np.abs(gradients - expected_gradients)) == 0.0):
            raise RuntimeError("FortML fold diagnostics disagree with NumPy")
        output: dict[str, float] = {}
        for line in completed.stdout.splitlines():
            match = re.match(r"^(cross_validation(?:_value|_gradient|_objective)?),([^,]+)$",
                             line.strip())
            if match:
                output[match.group(1)] = float(match.group(2))
        timing = output.get("cross_validation")
        if timing is None:
            raise RuntimeError("FortML release app emitted no cross-validation timing")
        value_error = abs(output.get("cross_validation_value", np.nan) - expected_value)
        objective_error = abs(output.get("cross_validation_objective", np.nan) - expected_objective)
        gradient_error = abs(output.get("cross_validation_gradient", np.nan) - 1.0)
        error = max(float(np.max(np.abs(values - expected_values))), value_error,
                    objective_error, gradient_error)
        if not np.isfinite(error) or error > 2.0e-14:
            raise RuntimeError(f"FortML products disagree with NumPy: {error:.3e}")
    return [row(
        details, backend="fortml", status="pass", seconds_per_operation=timing,
        value=output["cross_validation_value"], gradient=output["cross_validation_gradient"],
        objective_value=output["cross_validation_objective"], max_abs_error=error,
        oracle="independent K-fold fold means and weighted reduction",
        notes="all folds, weights, parameter gradient, and FortOpt orientation checked",
    )]


def device_refusal(details: dict[str, str]) -> list[dict[str, Any]]:
    return [row(
        {**details, "device": "cuda"}, backend="fortml", status="unavailable",
        oracle="FortML CPU index/scoring control-plane boundary",
        notes="CUDA scoring is a typed refusal until estimator callbacks are resident",
    )]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/cross_validation.csv"))
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = arguments.fortml.resolve()
    output = arguments.output.resolve()
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }
    records = run_numpy(details) + run_fortml(fortml, details) + device_refusal(details)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: item.get(field, "") for field in FIELDS} for item in records)
    print(f"wrote {len(records)} rows to {output}")


if __name__ == "__main__":
    main()
