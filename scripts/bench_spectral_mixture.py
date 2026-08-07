#!/usr/bin/env python3
"""Correctness-gated GPyTorch-compatible spectral-mixture benchmark.

NumPy independently evaluates the spectral-mixture covariance and central
parameter/input products.  FortML is retained only after checksum comparison;
CUDA is represented as an explicit typed refusal until a resident kernel is
linked.
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


N, D, Q, REPETITIONS = 256, 3, 2, 24
PARAMETERS = np.array([
    np.log(1.3), np.log(0.24), np.log(0.18), np.log(0.29), 0.18, 0.41, 0.33,
    np.log(0.8), np.log(0.31), np.log(0.37), np.log(0.22), -0.27, 0.12, -0.21,
])
DIRECTION = 0.08 - 0.013 * np.arange(1, PARAMETERS.size + 1, dtype=np.float64)
OPERATIONS = ("matrix", "matrix_jvp", "parameter_vjp", "parameter_hvp", "input_derivatives")
FIELDS = (
    "workload", "operation", "backend", "device", "status", "n_samples",
    "n_features", "num_mixtures", "repetitions", "seconds_per_operation",
    "value", "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    ignored_names = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            pass
    dirty = []
    for line in subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).splitlines():
        if line[3:].strip() not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(1, N + 1, dtype=np.float64)[:, None]
    j = np.arange(1, D + 1, dtype=np.float64)[None, :]
    points = np.sin(0.013 * i + 0.17 * j) + 0.2 * np.cos(0.007 * i * j)
    rows = np.arange(1, N + 1, dtype=np.float64)[:, None]
    cols = np.arange(1, N + 1, dtype=np.float64)[None, :]
    return points, np.sin(0.003 * (rows + 2.0 * cols))


def kernel(a: np.ndarray, b: np.ndarray, theta: np.ndarray = PARAMETERS) -> np.ndarray:
    delta = a[:, None, :] - b[None, :, :]
    value = np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    block = 1 + 2 * D
    two_pi = 2.0 * np.pi
    for q in range(Q):
        base = q * block
        weight = np.exp(theta[base])
        scales = np.exp(theta[base + 1:base + 1 + D])
        means = theta[base + 1 + D:base + 1 + 2 * D]
        component = np.ones_like(value) * weight
        for d in range(D):
            tau = delta[:, :, d]
            component *= np.exp(-0.5 * two_pi**2 * tau**2 * scales[d]**2) * np.cos(two_pi * tau * means[d])
        value += component
    return value


def weighted_vjp(points: np.ndarray, cotangent: np.ndarray, theta: np.ndarray) -> np.ndarray:
    h = 1.0e-5
    result = np.empty(theta.size)
    for index in range(theta.size):
        direction = np.zeros_like(theta)
        direction[index] = h
        result[index] = np.sum(cotangent * (kernel(points, points, theta + direction) -
                                             kernel(points, points, theta - direction))) / (2.0 * h)
    return result


def oracle() -> dict[str, float]:
    points, cotangent = fixture()
    h = 1.0e-5
    matrix = kernel(points, points)
    matrix_jvp = (kernel(points, points, PARAMETERS + h * DIRECTION) -
                  kernel(points, points, PARAMETERS - h * DIRECTION)) / (2.0 * h)
    parameter_vjp = weighted_vjp(points, cotangent, PARAMETERS)
    parameter_hvp = (weighted_vjp(points, cotangent, PARAMETERS + h * DIRECTION) -
                     weighted_vjp(points, cotangent, PARAMETERS - h * DIRECTION)) / (2.0 * h)
    a, b = points[0], points[1]
    input_h = 2.0e-5

    def scalar(x: np.ndarray, y: np.ndarray) -> float:
        return float(kernel(x[None, :], y[None, :])[0, 0])

    value = scalar(a, b)
    gradient_a = np.empty(D); gradient_b = np.empty(D); mixed = np.empty((D, D))
    for d in range(D):
        ap, am = a.copy(), a.copy(); ap[d] += input_h; am[d] -= input_h
        gradient_a[d] = (scalar(ap, b) - scalar(am, b)) / (2.0 * input_h)
        bp, bm = b.copy(), b.copy(); bp[d] += input_h; bm[d] -= input_h
        gradient_b[d] = (scalar(a, bp) - scalar(a, bm)) / (2.0 * input_h)
        for e in range(D):
            ape, ame = a.copy(), a.copy(); ape[e] += input_h; ame[e] -= input_h
            mixed[e, d] = ((scalar(ape, bp) - scalar(ape, bm)) -
                           (scalar(ame, bp) - scalar(ame, bm))) / (4.0 * input_h**2)
    return {
        "matrix": float(np.sum(matrix)),
        "matrix_jvp": float(np.sum(matrix_jvp)),
        "parameter_vjp": float(np.sum(parameter_vjp)),
        "parameter_hvp": float(np.sum(parameter_hvp)),
        "input_derivatives": float(value + np.sum(gradient_a) + np.sum(gradient_b) + np.sum(mixed)),
    }


def row(metadata: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(metadata)
    result.update({"n_samples": N, "n_features": D, "num_mixtures": Q,
                   "repetitions": REPETITIONS, **values})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path(__file__).resolve().parents[2] / "fortml")
    parser.add_argument("--output", type=Path, default=Path("results/spectral_mixture.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ignored = ((root / "results" / "spectral_mixture.csv").resolve(),)
    metadata = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(args.fortml), "benchmark_revision": revision(root, ignored),
        "compiler": "gfortran", "flags": "-O3",
    }
    expected = oracle()
    rows: list[dict[str, Any]] = []
    for operation, value in expected.items():
        rows.append(row(metadata, workload="spectral_mixture", operation=operation,
                        backend="numpy_oracle", device="cpu", status="pass", value=value,
                        max_abs_error=0.0, oracle="independent NumPy central-product oracle",
                        notes="reference checksum; not a timing"))
        rows.append(row(metadata, workload="spectral_mixture", operation=operation,
                        backend="fortml", device="cuda", status="unavailable", oracle="typed_device_contract",
                        notes="typed refusal: resident CUDA spectral-mixture kernel is not linked"))

    target = args.fortml / "app" / "fortml_bench_spectral_mixture.f90"
    environment = os.environ.copy(); environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=args.fortml, env=environment,
                           text=True, capture_output=True)
    if target.is_file() and build.returncode == 0:
        run = subprocess.run(["fo", "exec", "--no-build", "fortml_bench_spectral_mixture"], cwd=args.fortml,
                             env=environment, text=True, capture_output=True)
        records: dict[str, tuple[float, float]] = {}
        if run.returncode == 0:
            for line in run.stdout.splitlines():
                fields = [part.strip() for part in line.split(",")]
                if len(fields) == 5 and fields[0] == "spectral_mixture":
                    records[fields[2]] = (float(fields[3]), float(fields[4]))
        for operation, oracle_value in expected.items():
            if operation not in records:
                continue
            seconds, actual = records[operation]
            error = abs(actual - oracle_value)
            if error > 2.0e-6 * max(1.0, abs(oracle_value)):
                raise RuntimeError(f"FortML checksum mismatch {operation}: {error:.3e}")
            rows.append(row(metadata, workload="spectral_mixture", operation=operation,
                            backend="fortml", device="cpu", status="pass", value=actual,
                            max_abs_error=error, seconds_per_operation=seconds,
                            oracle="independent NumPy central-product oracle", notes="release app; host CPU timing"))
    else:
        for operation in expected:
            rows.append(row(metadata, workload="spectral_mixture", operation=operation,
                            backend="fortml", device="cpu", status="unavailable", oracle="",
                            notes="release app did not build; no timing retained"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
