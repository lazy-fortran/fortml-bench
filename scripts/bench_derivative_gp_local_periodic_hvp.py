#!/usr/bin/env python3
"""Correctness-gated local-periodic derivative-GP HVP benchmark.

The NumPy reference independently assembles the mixed value/first-derivative
covariance and central-differences its dense likelihood gradient.  FortML is
accepted only when the release app checksum agrees; CUDA remains an explicit
resident-graph refusal.
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


X = np.array([[-0.80, 0.72], [0.10, -0.15], [0.35, 1.05],
              [-0.55, 0.60], [0.90, -0.95]], dtype=np.float64)
Y = np.array([[0.70], [-0.20], [0.95], [0.30], [-0.65]], dtype=np.float64)
COMPONENTS = np.array([0, 1, 2, 1, 0], dtype=np.int64)
THETA = np.log(np.array([1.30, 0.85, 0.62, 1.70, 0.045], dtype=np.float64))
DIRECTION = np.array([0.11, -0.08, 0.14, -0.06, 0.17], dtype=np.float64)
JITTER = 1.0e-10
REPETITIONS = 32
FIELDS = (
    "workload", "operation", "backend", "device", "status", "n_samples",
    "n_features", "repetitions", "seconds_per_operation", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"],
                                   text=True).strip()
    ignored_names = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            pass
    dirty = [line for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines() if line[3:].strip() not in ignored_names]
    return head + ("+dirty" if dirty else "")


def covariance(theta: np.ndarray, x1: np.ndarray, component1: int,
               x2: np.ndarray, component2: int) -> float:
    difference = x1 - x2
    squared_distance = float(np.dot(difference, difference))
    distance = np.sqrt(squared_distance)
    variance, envelope, periodic, period = np.exp(theta[:4])
    a = 0.5 / envelope**2
    b = 2.0 / periodic**2
    c = np.pi / period
    if distance <= 1.0e-8:
        t0 = c*c*squared_distance - c**4*squared_distance**2/3.0
        t1 = c*c - 2.0*c**4*squared_distance/3.0
        t2 = -2.0*c**4/3.0 + 4.0*c**6*squared_distance/15.0
    else:
        z = c*distance
        t0 = np.sin(z)**2
        t1 = c*np.sin(2.0*z)/(2.0*distance)
        t2 = c*(2.0*z*np.cos(2.0*z) - np.sin(2.0*z))/(4.0*distance**3)
    value = variance*np.exp(-a*squared_distance - b*t0)
    first = -value*(a + b*t1)
    second = value*((a + b*t1)**2 - b*t2)
    if component1 == 0 and component2 == 0:
        return float(value)
    if component1 > 0 and component2 == 0:
        return float(2.0*first*difference[component1 - 1])
    if component1 == 0 and component2 > 0:
        return float(-2.0*first*difference[component2 - 1])
    return float(-2.0*first*float(component1 == component2) - 4.0*second*
                 difference[component1 - 1]*difference[component2 - 1])


def likelihood(theta: np.ndarray) -> float:
    n = X.shape[0]
    matrix = np.empty((n, n), dtype=np.float64)
    for j in range(n):
        for i in range(n):
            matrix[i, j] = covariance(theta, X[i], int(COMPONENTS[i]),
                                      X[j], int(COMPONENTS[j]))
    matrix[np.diag_indices(n)] += np.exp(theta[-1]) + JITTER
    chol = np.linalg.cholesky(matrix)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, Y))
    return float(-0.5*np.sum(Y*alpha) - np.sum(np.log(np.diag(chol))) -
                 0.5*n*np.log(2.0*np.pi))


def gradient(theta: np.ndarray) -> np.ndarray:
    step = 3.0e-6
    result = np.empty(theta.size)
    for index in range(theta.size):
        perturbation = np.zeros_like(theta)
        perturbation[index] = step
        result[index] = (likelihood(theta + perturbation) -
                         likelihood(theta - perturbation))/(2.0*step)
    return result


def oracle_hvp() -> np.ndarray:
    step = 2.0e-4
    return (gradient(THETA + step*DIRECTION) -
            gradient(THETA - step*DIRECTION))/(2.0*step)


def row(metadata: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(metadata)
    result.update({"n_samples": X.shape[0], "n_features": X.shape[1],
                   "repetitions": REPETITIONS, **values})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/derivative_gp_local_periodic_hvp.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ignored = ((root / "results" / "derivative_gp_local_periodic_hvp.csv").resolve(),)
    metadata = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(args.fortml), "benchmark_revision": revision(root, ignored),
        "compiler": "gfortran", "flags": "-O3",
    }
    expected = oracle_hvp()
    rows: list[dict[str, Any]] = [row(
        metadata, workload="derivative_gp_local_periodic", operation="mixed_parameter_hvp",
        backend="numpy_oracle", device="cpu", status="pass", value=float(np.sum(expected)),
        max_abs_error=0.0,
        oracle="independent NumPy dense mixed-observation likelihood HVP oracle",
        notes="central differences only in the independent behavioral oracle")]

    env = os.environ.copy()
    env.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1", "FO_SCAN_FALLBACK": "regex"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=args.fortml, env=env,
                           text=True, capture_output=True)
    if build.returncode != 0:
        raise RuntimeError(f"FortML HVP app did not build: {build.stderr[-1000:]}")
    run = subprocess.run(["fo", "exec", "--no-build",
                          "fortml_bench_derivative_gp_local_periodic_hvp"],
                         cwd=args.fortml, env=env, text=True, capture_output=True)
    records: dict[str, tuple[float, float]] = {}
    if run.returncode == 0:
        for line in run.stdout.splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) == 5 and fields[0] == "derivative_gp_local_periodic":
                records[fields[2]] = (float(fields[3]), float(fields[4]))
    if "hvp" not in records:
        raise RuntimeError(f"FortML HVP app failed: {run.stderr[-1000:]}")
    seconds, actual = records["hvp"]
    error = abs(actual - float(np.sum(expected)))
    if error > 5.0e-4*max(1.0, abs(float(np.sum(expected)))):
        raise RuntimeError(f"FortML HVP checksum mismatch: {error:.3e}")
    rows.append(row(metadata, workload="derivative_gp_local_periodic",
                    operation="mixed_parameter_hvp", backend="fortml", device="cpu",
                    status="pass", value=actual, max_abs_error=error,
                    seconds_per_operation=seconds,
                    oracle="independent NumPy dense mixed-observation likelihood HVP oracle",
                    notes="analytic radial local-periodic F/Fs/Fss products"))
    rows.append(row(metadata, workload="derivative_gp_local_periodic",
                    operation="mixed_parameter_hvp", backend="fortml", device="cuda",
                    status="unavailable", oracle="typed_device_contract",
                    notes="FORTNUM_NOT_IMPLEMENTED: resident derivative-GP covariance/factorization graph is not linked"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
