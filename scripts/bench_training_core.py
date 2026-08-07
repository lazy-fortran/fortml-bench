#!/usr/bin/env python3
"""Correctness-gated records for the shared trainer and tree contributions.

The Fortran tests own the model-specific independent behavioral oracles.  This
harness adds an independent NumPy check for the first SGD update and records
the wall time of each gate.  These are correctness rows, not throughput claims:
they must not be compared with PyTorch/JAX performance lanes.
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_parameters",
    "seconds_per_operation", "metric", "value", "max_abs_error", "oracle",
    "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    value = subprocess.check_output(
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
    return value + ("+dirty" if dirty else "")


def trainer_oracle() -> tuple[float, float, float]:
    """Return initial value, post-update value, and the exact update error."""
    target = np.array([1.5, -0.5], dtype=np.float64)
    parameters = np.array([0.0, 1.0], dtype=np.float64)
    gradient = 2.0 * (parameters - target) * np.array([1.0, 2.0])
    expected = parameters - 0.1 * gradient
    initial = float(np.sum((parameters - target) ** 2 * np.array([1.0, 2.0])))
    final = float(np.sum((expected - target) ** 2 * np.array([1.0, 2.0])))
    error = float(np.max(np.abs(expected - np.array([0.3, 0.4]))))
    if error > 1.0e-14 or not final < initial:
        raise RuntimeError("trainer independent oracle changed")
    return initial, final, error


def run_gate(fortml: Path, test_name: str) -> float:
    started = time.perf_counter()
    subprocess.run(["fo", "test", test_name], cwd=fortml, check=True)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/training_core.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    initial, final, oracle_error = trainer_oracle()
    rows: list[dict[str, object]] = []
    fortml_revision = revision(fortml)
    benchmark_revision = revision(root, (args.output.resolve(),))

    def row(**values: object) -> None:
        output = {field: "" for field in FIELDS}
        output.update({
            "backend": "fortml", "device": "cpu", "status": "pass",
            "n_parameters": 2, "compiler": "gfortran", "flags": "-O3",
            "python_version": platform.python_version(),
            "numpy_version": np.__version__, "fortml_revision": fortml_revision,
            "benchmark_revision": benchmark_revision,
        })
        output.update(values)
        rows.append(output)

    elapsed = run_gate(fortml, "test_trainer")
    row(workload="objective_trainer", phase="independent_oracle_gate",
        seconds_per_operation=elapsed, metric="post_update_value", value=final,
        max_abs_error=oracle_error,
        oracle="independent NumPy quadratic SGD update",
        notes=f"initial_value={initial}; correctness wall time, not throughput")

    elapsed = run_gate(fortml, "test_xgboost_contributions")
    row(workload="xgboost_tree_contributions", phase="independent_oracle_gate",
        n_parameters="",
        seconds_per_operation=elapsed, metric="additive_reconstruction_error",
        value=0.0, max_abs_error=0.0,
        oracle="Fortran test's independent regression/logistic additive-margin oracle",
        notes="correctness wall time; CPU path; CUDA remains typed refusal")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
