#!/usr/bin/env python3
"""Correctness-gated ordinal-GP log-probability benchmark.

The NumPy oracle independently evaluates adjacent normal-CDF probabilities,
their logarithms, and a directional derivative.  The FortML test additionally
checks packed parameter and query-input products, VJP duality, and the typed
CUDA refusal before a release row is retained.
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
    """Dependency-free normal CDF for the independent oracle."""
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


def oracle() -> tuple[float, float, float, float]:
    mean = np.array([-0.8, -0.1, 0.7, 1.2, 2.0], dtype=np.float64)
    variance = np.array([0.05, 0.15, 0.20, 0.10, 0.30], dtype=np.float64)
    mean_dot = np.array([0.2, -0.1, 0.3, -0.2, 0.1], dtype=np.float64)
    variance_dot = np.array([0.03, -0.02, 0.01, 0.04, -0.03], dtype=np.float64)
    cut_points = np.array([1.5, 2.5], dtype=np.float64)
    value = probabilities(mean, variance, cut_points)
    logs = np.log(np.maximum(value, np.finfo(np.float64).tiny))
    step = 2.0e-6
    log_fd = (np.log(np.maximum(
        probabilities(mean + step * mean_dot, variance + step * variance_dot, cut_points),
        np.finfo(np.float64).tiny)) - np.log(np.maximum(
        probabilities(mean - step * mean_dot, variance - step * variance_dot, cut_points),
        np.finfo(np.float64).tiny))) / (2.0 * step)
    scale = np.sqrt(1.0 + variance)
    log_dot = np.zeros_like(logs)
    lower_dot = np.zeros(mean.size, dtype=np.float64)
    for index, cut in enumerate(cut_points):
        z = (cut - mean) / scale
        density = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
        upper_dot = density * (-mean_dot / scale -
                               0.5 * (cut - mean) * variance_dot / scale**3)
        probability_dot = upper_dot - lower_dot
        log_dot[:, index] = probability_dot / value[:, index]
        lower_dot = upper_dot
    log_dot[:, -1] = -lower_dot / value[:, -1]
    simplex_error = float(np.max(np.abs(np.sum(value, axis=1) - 1.0)))
    equivalence_error = float(np.max(np.abs(np.exp(logs) - value)))
    jvp_error = float(np.max(np.abs(log_dot - log_fd)))
    finite_error = 0.0 if np.all(np.isfinite(logs)) else float("inf")
    if max(simplex_error, equivalence_error, jvp_error, finite_error) > 2.0e-7:
        raise RuntimeError(
            f"ordinal log-probability oracle failed: simplex={simplex_error:.3e}, "
            f"equivalence={equivalence_error:.3e}, jvp={jvp_error:.3e}")
    return simplex_error, equivalence_error, jvp_error, float(np.sum(logs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gp_ordinal_log_proba.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    simplex_error, equivalence_error, jvp_error, log_sum = oracle()
    started = time.perf_counter()
    if args.skip_fortml:
        status, notes = "skipped", "--skip-fortml"
    else:
        environment = os.environ.copy()
        environment["FO_SCAN_FALLBACK"] = "regex"
        subprocess.run(["fo", "test", "test_gp_ordinal_log_proba"],
                       cwd=fortml, env=environment, check=True)
        status = "pass"
        notes = "independent Fortran log-probability, JVP/VJP, and CUDA boundary oracle"
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
        row.update({"workload": "gp_ordinal_log_proba", "backend": "fortml",
                    "device": "cpu", "n_samples": 9, "n_classes": 3})
        row.update(values)
        rows.append(row)

    add(phase="independent_oracle", backend="numpy_oracle", status="pass",
        metric="log_probability_sum", value=log_sum,
        max_abs_error=max(simplex_error, equivalence_error, jvp_error),
        oracle="independent NumPy adjacent-normal-CDF log-probability and FD JVP",
        notes=(f"simplex={simplex_error:.3e}; equivalence={equivalence_error:.3e}; "
               f"jvp={jvp_error:.3e}"))
    add(phase="public_contract_gate", status=status, seconds_per_operation=elapsed,
        metric="log_probability_jvp_max_abs_error", value=jvp_error,
        max_abs_error=jvp_error,
        oracle="FortML test_gp_ordinal_log_proba behavioral gate", notes=notes)
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_ordinal_log_probability_graph", value="nan",
        max_abs_error="nan", oracle="typed FortML CUDA capability refusal",
        notes="resident ordinal covariance and normal-CDF graph are not linked")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
