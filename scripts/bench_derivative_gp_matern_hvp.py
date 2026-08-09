#!/usr/bin/env python3
"""Correctness-gated mixed-observation Matérn GP HVP lane.

The NumPy path is an independent dense covariance and central-difference
oracle.  FortML is timed only after its CPU HVP checksum agrees; CUDA is
recorded as the typed resident-graph refusal rather than silently falling back
to host execution.
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


X = np.array([[-0.20], [0.35], [0.90], [1.40]])
Y = np.array([[0.70], [-0.10], [0.55], [-0.35]])
COMPONENTS = np.array([0, 1, 0, 1])
THETA = np.array([np.log(1.35), np.log(0.78), np.log(0.045)])
DIRECTION = np.array([0.17, -0.11, 0.08])
JITTER = 1.0e-10
REPETITIONS = 32
KINDS = ("matern32", "matern52")
FIELDS = (
    "workload", "operation", "backend", "device", "status", "n_samples",
    "n_features", "repetitions", "seconds_per_operation", "value", "max_abs_error",
    "oracle", "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    """Return a commit plus a dirty marker, excluding generated result files."""
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    ignored_names = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            pass
    dirty = [
        line for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True
        ).splitlines()
        if line[3:].strip() not in ignored_names
    ]
    return head + ("+dirty" if dirty else "")


def covariance(theta: np.ndarray, x1: float, component1: int,
               x2: float, component2: int, kind: str) -> float:
    """Independent one-dimensional value/first-derivative Matérn covariance."""
    delta = x1 - x2
    distance = abs(delta)
    variance = np.exp(theta[0])
    lengthscale = np.exp(theta[1])
    z = distance / lengthscale
    if kind == "matern32":
        a = np.sqrt(3.0)
        exponential = np.exp(-a * z)
        value = variance * (1.0 + a * z) * exponential
        first = -3.0 * variance * z * exponential / lengthscale
        second = 3.0 * variance * (a * z - 1.0) * exponential / lengthscale**2
    else:
        a = np.sqrt(5.0)
        exponential = np.exp(-a * z)
        value = variance * (1.0 + a * z + 5.0 * z * z / 3.0) * exponential
        first = -(5.0 / 3.0) * variance * z * (1.0 + a * z) * exponential / lengthscale
        second = (5.0 / 3.0) * variance * (5.0 * z * z - a * z - 1.0) * exponential / lengthscale**2
    if component1 == 0 and component2 == 0:
        return float(value)
    if component1 > 0 and component2 == 0:
        return float(first * (0.0 if distance == 0.0 else delta / distance))
    if component1 == 0 and component2 > 0:
        return float(-first * (0.0 if distance == 0.0 else delta / distance))
    return float(-second)


def likelihood(theta: np.ndarray, kind: str) -> float:
    n = X.shape[0]
    matrix = np.empty((n, n))
    for j in range(n):
        for i in range(n):
            matrix[i, j] = covariance(theta, float(X[i, 0]), int(COMPONENTS[i]),
                                       float(X[j, 0]), int(COMPONENTS[j]), kind)
    matrix[np.diag_indices(n)] += np.exp(theta[2]) + JITTER
    chol = np.linalg.cholesky(matrix)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, Y))
    return float(-0.5 * np.sum(Y * alpha) - np.sum(np.log(np.diag(chol))) -
                 0.5 * n * np.log(2.0 * np.pi))


def gradient(theta: np.ndarray, kind: str) -> np.ndarray:
    step = 3.0e-6
    result = np.empty(theta.size)
    for index in range(theta.size):
        perturbation = np.zeros_like(theta)
        perturbation[index] = step
        result[index] = (likelihood(theta + perturbation, kind) -
                         likelihood(theta - perturbation, kind)) / (2.0 * step)
    return result


def oracle(kind: str) -> float:
    """Return the HVP checksum from the independent central-difference oracle."""
    step = 2.0e-4
    return float(np.sum((gradient(THETA + step * DIRECTION, kind) -
                         gradient(THETA - step * DIRECTION, kind)) / (2.0 * step)))


def row(metadata: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(metadata)
    result.update({"n_samples": X.shape[0], "n_features": X.shape[1],
                   "repetitions": REPETITIONS, **values})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fortml", type=Path,
        default=Path(__file__).resolve().parents[2] / "fortml-gp-matern-derivative",
    )
    parser.add_argument("--output", type=Path,
                        default=Path("results/derivative_gp_matern_hvp.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ignored = (args.output.resolve(),)
    metadata = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(args.fortml), "benchmark_revision": revision(root, ignored),
        "compiler": "gfortran", "flags": "-O3",
    }
    expected = {kind: oracle(kind) for kind in KINDS}
    oracle_description = "independent NumPy dense mixed-observation likelihood HVP oracle"
    rows: list[dict[str, Any]] = []
    for kind in KINDS:
        rows.append(row(
            metadata, workload=f"derivative_gp_{kind}", operation="mixed_parameter_hvp",
            backend="numpy_oracle", device="cpu", status="pass", value=expected[kind],
            max_abs_error=0.0, oracle=oracle_description,
            notes="central differences only in the independent behavioral oracle",
        ))

    env = os.environ.copy()
    env.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1", "FO_SCAN_FALLBACK": "regex"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=args.fortml, env=env,
                           text=True, capture_output=True)
    target = args.fortml / "app" / "fortml_bench_derivative_gp_matern_hvp.f90"
    if not target.is_file() or build.returncode != 0:
        raise RuntimeError(f"FortML HVP app did not build: {build.stderr[-1000:]}")
    run = subprocess.run(["fo", "exec", "--no-build", "fortml_bench_derivative_gp_matern_hvp"],
                         cwd=args.fortml, env=env, text=True, capture_output=True)
    records: dict[str, tuple[float, float]] = {}
    if run.returncode == 0:
        for line in run.stdout.splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) == 5 and fields[0] == "derivative_gp_matern":
                records[fields[1]] = (float(fields[3]), float(fields[4]))
    for kind in KINDS:
        if kind not in records:
            raise RuntimeError(f"FortML {kind} HVP app failed: {run.stderr[-1000:]}")
        seconds, actual = records[kind]
        error = abs(actual - expected[kind])
        if error > 3.0e-4 * max(1.0, abs(expected[kind])):
            raise RuntimeError(f"FortML {kind} HVP checksum mismatch: {error:.3e}")
        rows.append(row(
            metadata, workload=f"derivative_gp_{kind}", operation="mixed_parameter_hvp",
            backend="fortml", device="cpu", status="pass", value=actual,
            max_abs_error=error, seconds_per_operation=seconds, oracle=oracle_description,
            notes="CPU reference app; analytic Matérn radial parameter products",
        ))
        rows.append(row(
            metadata, workload=f"derivative_gp_{kind}", operation="mixed_parameter_hvp",
            backend="fortml", device="cuda", status="unavailable", oracle="typed_device_contract",
            notes="FORTNUM_NOT_IMPLEMENTED: resident derivative-GP covariance/factorization graph is not linked",
        ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
