#!/usr/bin/env python3
"""Correctness-gated cosine mixed-observation GP HVP lane.

The NumPy implementation is an independent dense covariance and central-
difference oracle. FortML is timed only after its checksum agrees; CUDA is
recorded as the typed resident-graph refusal.
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


X = np.array([[0.0], [0.42], [1.03]])
Y = np.array([[0.8], [-0.2], [0.6]])
COMPONENTS = np.array([0, 1, 0])
THETA = np.array([np.log(1.2), np.log(0.75), np.log(0.08)])
DIRECTION = np.array([0.17, -0.11, 0.08])
JITTER = 1.0e-10
REPETITIONS = 32
FIELDS = (
    "workload", "operation", "backend", "device", "status", "n_samples",
    "n_features", "repetitions", "seconds_per_operation", "value", "max_abs_error",
    "oracle", "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
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
    dirty = [
        line for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True
        ).splitlines()
        if line[3:].strip() not in ignored_names
    ]
    return head + ("+dirty" if dirty else "")


def covariance(theta: np.ndarray, x1: float, component1: int,
               x2: float, component2: int) -> float:
    delta = x1 - x2
    distance = abs(delta)
    variance, lengthscale = np.exp(theta[:2])
    if distance <= 1.0e-8:
        value = variance
        first = -variance / (2.0 * lengthscale**2)
        second = variance / (12.0 * lengthscale**4)
    else:
        z = distance / lengthscale
        value = variance * np.cos(z)
        first = -variance * np.sin(z) / (2.0 * lengthscale * distance)
        second = variance * (np.sin(z) - z * np.cos(z)) / (4.0 * lengthscale**4 * z**3)
    if component1 == 0 and component2 == 0:
        return float(value)
    if component1 > 0 and component2 == 0:
        return float(2.0 * first * delta)
    if component1 == 0 and component2 > 0:
        return float(-2.0 * first * delta)
    return float(-2.0 * first - 4.0 * second * delta * delta)


def likelihood(theta: np.ndarray) -> float:
    n = X.shape[0]
    matrix = np.empty((n, n))
    for j in range(n):
        for i in range(n):
            matrix[i, j] = covariance(theta, float(X[i, 0]), int(COMPONENTS[i]),
                                       float(X[j, 0]), int(COMPONENTS[j]))
    matrix[np.diag_indices(n)] += np.exp(theta[2]) + JITTER
    chol = np.linalg.cholesky(matrix)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, Y))
    return float(-0.5 * np.sum(Y * alpha) - np.sum(np.log(np.diag(chol))) -
                 0.5 * n * np.log(2.0 * np.pi))


def gradient(theta: np.ndarray) -> np.ndarray:
    step = 2.0e-5
    result = np.empty(theta.size)
    for index in range(theta.size):
        perturbation = np.zeros_like(theta)
        perturbation[index] = step
        result[index] = (likelihood(theta + perturbation) -
                         likelihood(theta - perturbation)) / (2.0 * step)
    return result


def oracle() -> float:
    step = 2.0e-4
    return float(np.sum((gradient(THETA + step * DIRECTION) -
                         gradient(THETA - step * DIRECTION)) / (2.0 * step)))


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
                        default=Path("results/derivative_gp_cosine_hvp.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/DERIVATIVE_GP_COSINE_HVP.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ignored = (args.output.resolve(), args.report.resolve())
    metadata = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(args.fortml), "benchmark_revision": revision(root, ignored),
        "compiler": "gfortran", "flags": "-O3",
    }
    expected = oracle()
    oracle_description = "independent NumPy dense cosine mixed-observation likelihood HVP oracle"
    rows: list[dict[str, Any]] = [row(
        metadata, workload="derivative_gp_cosine", operation="mixed_parameter_hvp",
        backend="numpy_oracle", device="cpu", status="pass", value=expected,
        max_abs_error=0.0, oracle=oracle_description,
        notes="central differences only in the independent behavioral oracle",
    )]

    env = os.environ.copy()
    env.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1", "FO_SCAN_FALLBACK": "regex"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=args.fortml, env=env,
                           text=True, capture_output=True)
    target = args.fortml / "app" / "fortml_bench_derivative_gp_cosine_hvp.f90"
    if not target.is_file() or build.returncode != 0:
        raise RuntimeError(f"FortML cosine HVP app did not build: {build.stderr[-1000:]}")
    run = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_derivative_gp_cosine_hvp"],
        cwd=args.fortml, env=env, text=True, capture_output=True,
    )
    cpu_record: tuple[float, float] | None = None
    cuda_code: str | None = None
    if run.returncode == 0:
        for line in run.stdout.splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) == 6 and fields[:4] == [
                    "derivative_gp_cosine", "cosine", "hvp", "cpu"]:
                cpu_record = (float(fields[4]), float(fields[5]))
            elif len(fields) == 4 and fields[:3] == [
                    "derivative_gp_cosine", "hvp", "cuda_refused"]:
                cuda_code = fields[3]
    if cpu_record is None:
        raise RuntimeError(f"FortML cosine HVP app failed: {run.stderr[-1000:]}")
    seconds, actual = cpu_record
    error = abs(actual - expected)
    if error > 5.0e-4 * max(1.0, abs(expected)):
        raise RuntimeError(f"FortML cosine HVP checksum mismatch: {error:.3e}")
    rows.append(row(
        metadata, workload="derivative_gp_cosine", operation="mixed_parameter_hvp",
        backend="fortml", device="cpu", status="pass", value=actual,
        max_abs_error=error, seconds_per_operation=seconds, oracle=oracle_description,
        notes="CPU reference app; analytic cosine radial products",
    ))
    rows.append(row(
        metadata, workload="derivative_gp_cosine", operation="mixed_parameter_hvp",
        backend="fortml", device="cuda", status="unavailable", oracle="typed_device_contract",
        notes=f"FORTNUM_NOT_IMPLEMENTED refusal code {cuda_code}: resident derivative-GP graph is not linked",
    ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# Cosine mixed-observation GP HVP\n\n"
        f"FortML revision: `{metadata['fortml_revision']}`  \n"
        f"Benchmark revision: `{metadata['benchmark_revision']}`  \n\n"
        "The independent NumPy dense covariance oracle checks the packed "
        "`[log variance, log lengthscale, log noise]` HVP by central "
        "differences of the likelihood gradient. The Fortran CPU checksum is "
        f"{actual:.12e}, with absolute error {error:.3e}; the measured mean "
        f"time is {seconds:.3e} s/HVP over {REPETITIONS} repetitions.\n\n"
        f"CUDA is recorded as the typed refusal code `{cuda_code}`; no host "
        "fallback is hidden behind the device row.\n",
    )


if __name__ == "__main__":
    main()
