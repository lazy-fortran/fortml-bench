#!/usr/bin/env python3
"""Correctness and contract benchmark for the latent-Gaussian ordinal GP.

The NumPy oracle independently evaluates adjacent normal-CDF probabilities
and their directional derivative. The FortML release test additionally checks
the fitted Cholesky posterior, packed parameter products, query-input products,
and the explicit CUDA refusal before a timing row is retained.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_classes", "seconds_per_operation", "metric", "value", "max_abs_error",
    "oracle", "python_version", "numpy_version", "fortml_revision",
    "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    """Return a revision and mark unrelated working-tree edits as dirty."""
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


def normal_cdf(value: np.ndarray) -> np.ndarray:
    """Use a dependency-free normal CDF for the independent oracle."""
    return 0.5 * np.vectorize(math.erfc)(-value / np.sqrt(2.0))


def probabilities(mean: np.ndarray, variance: np.ndarray,
                  cut_points: np.ndarray) -> np.ndarray:
    scale = np.sqrt(1.0 + variance)
    output = np.empty((mean.size, cut_points.size + 1), dtype=np.float64)
    lower = np.zeros(mean.size, dtype=np.float64)
    for index, cut in enumerate(cut_points):
        upper = normal_cdf((cut - mean) / scale)
        output[:, index] = upper - lower
        lower = upper
    output[:, -1] = 1.0 - lower
    return output


def oracle() -> tuple[float, float, float]:
    mean = np.array([-0.8, -0.1, 0.7, 1.2, 2.0], dtype=np.float64)
    variance = np.array([0.05, 0.15, 0.20, 0.10, 0.30], dtype=np.float64)
    mean_dot = np.array([0.2, -0.1, 0.3, -0.2, 0.1], dtype=np.float64)
    variance_dot = np.array([0.03, -0.02, 0.01, 0.04, -0.03], dtype=np.float64)
    cut_points = np.array([1.5, 2.5], dtype=np.float64)
    value = probabilities(mean, variance, cut_points)
    scale = np.sqrt(1.0 + variance)
    derivative = np.zeros_like(value)
    lower_dot = np.zeros(mean.size, dtype=np.float64)
    step = 2.0e-6
    for index, cut in enumerate(cut_points):
        z = (cut - mean) / scale
        density = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
        upper_dot = density * (-mean_dot / scale -
                               0.5 * (cut - mean) * variance_dot / scale**3)
        derivative[:, index] = upper_dot - lower_dot
        lower_dot = upper_dot
    finite_difference = (probabilities(mean + step * mean_dot,
                                       variance + step * variance_dot, cut_points) -
                         probabilities(mean - step * mean_dot,
                                       variance - step * variance_dot, cut_points)) / (2.0 * step)
    derivative[:, -1] = -lower_dot
    simplex_error = float(np.max(np.abs(np.sum(value, axis=1) - 1.0)))
    jvp_error = float(np.max(np.abs(derivative - finite_difference)))
    if simplex_error > 2.0e-15 or jvp_error > 2.0e-10:
        raise RuntimeError(f"ordinal oracle failed: simplex={simplex_error:.3e}, "
                           f"jvp={jvp_error:.3e}")
    return simplex_error, jvp_error, float(np.sum(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gp_ordinal.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    simplex_error, jvp_error, probability_sum = oracle()
    started = time.perf_counter()
    if args.skip_fortml:
        status, notes = "skipped", "--skip-fortml"
    else:
        environment = os.environ.copy()
        environment["FO_SCAN_FALLBACK"] = "regex"
        subprocess.run(["fo", "test", "test_gp_ordinal_classification"],
                       cwd=fortml, env=environment, check=True)
        status = "pass"
        notes = "FortML test covers fitted ordinal GP, parameter/input products, and CUDA refusals"
    elapsed = time.perf_counter() - started
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": "gfortran", "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(details)
        row.update({"workload": "gp_ordinal", "backend": "fortml", "device": "cpu",
                    "n_samples": 9, "n_classes": 3})
        row.update(values)
        rows.append(row)

    add(phase="independent_oracle", backend="numpy_oracle", status="pass",
        metric="probability_simplex_sum", value=probability_sum,
        max_abs_error=max(simplex_error, jvp_error),
        oracle="independent NumPy adjacent-normal-CDF probability and JVP finite difference",
        notes=f"simplex_error={simplex_error:.3e}; jvp_error={jvp_error:.3e}")
    add(phase="public_contract_gate", status=status, seconds_per_operation=elapsed,
        metric="probability_jvp_max_abs_error", value=jvp_error,
        max_abs_error=jvp_error,
        oracle="FortML test_gp_ordinal_classification behavioral gate", notes=notes)
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_ordinal_gp_graph", value="nan", max_abs_error="nan",
        oracle="typed FortML CUDA capability refusal",
        notes="resident ordinal covariance and normal-CDF graph are not linked")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
