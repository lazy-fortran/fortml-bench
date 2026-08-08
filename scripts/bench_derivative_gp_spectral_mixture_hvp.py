#!/usr/bin/env python3
"""Correctness-gated mixed-observation spectral-mixture GP HVP lane.

The NumPy path independently assembles the derivative-observation covariance
and central-differences its likelihood gradient.  FortML is retained only
after its CPU checksum agrees; CUDA is recorded as the typed resident-graph
refusal.
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


X = np.array([[-0.7, 0.65], [0.2, -0.1], [0.35, 1.1], [-0.4, 0.55], [0.9, -0.85]])
Y = np.array([[0.8], [-0.25], [0.45], [1.1], [-0.6]])
COMPONENTS = np.array([0, 1, 2, 0, 1])
THETA = np.array([
    np.log(1.15), np.log(0.31), np.log(0.22), 0.21, 0.48,
    np.log(0.63), np.log(0.57), np.log(0.44), -0.37, 0.16,
    np.log(0.055),
])
DIRECTION = np.array([0.13, -0.08, 0.11, -0.05, 0.07, -0.04, 0.09, -0.06, 0.12, -0.03, 0.17])
JITTER = 1.0e-10
NOISE = 0.055
REPETITIONS = 32
FIELDS = (
    "workload", "operation", "backend", "device", "status", "n_samples",
    "n_features", "num_mixtures", "repetitions", "seconds_per_operation", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version", "fortml_revision",
    "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
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


def covariance(theta: np.ndarray, x1: np.ndarray, component1: int, x2: np.ndarray, component2: int) -> float:
    d = x1.size
    block = 1 + 2 * d
    result = 0.0
    tau = x1 - x2
    for q in range((theta.size - 1) // block):
        base = q * block
        weight = np.exp(theta[base])
        scales = np.exp(theta[base + 1:base + 1 + d])
        means = theta[base + 1 + d:base + 1 + 2 * d]
        f = np.empty(d)
        f1 = np.empty(d)
        f2 = np.empty(d)
        for i in range(d):
            phase = 2.0 * np.pi * tau[i] * means[i]
            l1 = -4.0 * np.pi**2 * tau[i] * scales[i]**2
            l2 = -4.0 * np.pi**2 * scales[i]**2
            e = np.exp(-2.0 * np.pi**2 * tau[i]**2 * scales[i]**2)
            c = np.cos(phase)
            c1_factor = -2.0 * np.pi * means[i] * np.sin(phase)
            c2_factor = -4.0 * np.pi**2 * means[i]**2 * c
            f[i] = e * c
            f1[i] = e * (l1 * c + c1_factor)
            f2[i] = e * ((l2 + l1 * l1) * c + 2.0 * l1 * c1_factor + c2_factor)
        if component1 == 0 and component2 == 0:
            block_value = np.prod(f)
        elif component1 > 0 and component2 == 0:
            block_value = f1[component1 - 1] * np.prod(np.delete(f, component1 - 1))
        elif component1 == 0 and component2 > 0:
            block_value = -f1[component2 - 1] * np.prod(np.delete(f, component2 - 1))
        elif component1 == component2:
            block_value = -f2[component1 - 1] * np.prod(np.delete(f, component1 - 1))
        else:
            keep = [i for i in range(d) if i not in (component1 - 1, component2 - 1)]
            block_value = -f1[component1 - 1] * f1[component2 - 1] * np.prod(f[keep])
        result += weight * block_value
    return float(result)


def likelihood(theta: np.ndarray) -> float:
    n = X.shape[0]
    matrix = np.empty((n, n))
    for j in range(n):
        for i in range(n):
            matrix[i, j] = covariance(theta, X[i], int(COMPONENTS[i]), X[j], int(COMPONENTS[j]))
    matrix[np.diag_indices(n)] += np.exp(theta[-1]) + JITTER
    chol = np.linalg.cholesky(matrix)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, Y))
    return float(-0.5 * np.sum(Y * alpha) - np.sum(np.log(np.diag(chol))) -
                 0.5 * n * np.log(2.0 * np.pi))


def gradient(theta: np.ndarray) -> np.ndarray:
    step = 3.0e-6
    result = np.empty(theta.size)
    for index in range(theta.size):
        direction = np.zeros_like(theta)
        direction[index] = step
        result[index] = (likelihood(theta + direction) - likelihood(theta - direction)) / (2.0 * step)
    return result


def oracle() -> float:
    step = 1.0e-3
    return float(np.sum((gradient(THETA + step * DIRECTION) -
                         gradient(THETA - step * DIRECTION)) / (2.0 * step)))


def row(metadata: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(metadata)
    result.update({"n_samples": X.shape[0], "n_features": X.shape[1], "num_mixtures": 2,
                   "repetitions": REPETITIONS, **values})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path,
                        default=Path(__file__).resolve().parents[2] / "fortml-spectral-mixture-hvp")
    parser.add_argument("--output", type=Path, default=Path("results/derivative_gp_spectral_mixture_hvp.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ignored = ((root / "results" / "derivative_gp_spectral_mixture_hvp.csv").resolve(),)
    metadata = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(args.fortml), "benchmark_revision": revision(root, ignored),
        "compiler": "gfortran", "flags": "-O3",
    }
    expected = oracle()
    rows: list[dict[str, Any]] = [row(metadata, workload="derivative_gp_spectral_mixture",
        operation="mixed_parameter_hvp", backend="numpy_oracle", device="cpu", status="pass",
        value=expected, max_abs_error=0.0,
        oracle="independent NumPy dense mixed-observation likelihood HVP oracle",
        notes="central differences only in the independent behavioral oracle")]

    env = os.environ.copy()
    env.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1", "FO_SCAN_FALLBACK": "regex"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=args.fortml, env=env,
                           text=True, capture_output=True)
    target = args.fortml / "app" / "fortml_bench_derivative_gp_spectral_mixture_hvp.f90"
    if target.is_file() and build.returncode == 0:
        run = subprocess.run(["fo", "exec", "--no-build", "fortml_bench_derivative_gp_spectral_mixture_hvp"],
                             cwd=args.fortml, env=env, text=True, capture_output=True)
        records: dict[str, tuple[float, float]] = {}
        if run.returncode == 0:
            for line in run.stdout.splitlines():
                fields = [part.strip() for part in line.split(",")]
                if len(fields) == 5 and fields[0] == "derivative_gp_spectral_mixture":
                    records[fields[2]] = (float(fields[3]), float(fields[4]))
        if "hvp" not in records:
            raise RuntimeError(f"FortML HVP app failed: {run.stderr[-1000:]}")
        seconds, actual = records["hvp"]
        error = abs(actual - expected)
        if error > 3.0e-4 * max(1.0, abs(expected)):
            raise RuntimeError(f"FortML HVP checksum mismatch: {error:.3e}")
        rows.append(row(metadata, workload="derivative_gp_spectral_mixture",
            operation="mixed_parameter_hvp", backend="fortml", device="cpu", status="pass",
            value=actual, max_abs_error=error, seconds_per_operation=seconds,
            oracle="independent NumPy dense mixed-observation likelihood HVP oracle",
            notes="CPU reference app; analytic four-jet covariance blocks"))
    else:
        raise RuntimeError(f"FortML HVP app did not build: {build.stderr[-1000:]}")

    rows.append(row(metadata, workload="derivative_gp_spectral_mixture", operation="mixed_parameter_hvp",
        backend="fortml", device="cuda", status="unavailable", oracle="typed_device_contract",
        notes="FORTNUM_NOT_IMPLEMENTED: resident derivative-GP covariance/factorization graph is not linked"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
