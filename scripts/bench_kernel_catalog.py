#!/usr/bin/env python3
"""Correctness-gated periodic and rational-quadratic kernel benchmark.

NumPy supplies an independent covariance and central-product oracle.  The
FortML release app is retained only when all checksums agree; CUDA remains an
explicit refusal because the resident postfix ABI does not yet carry the
third leaf parameter.
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


N = 256
D = 3
REPETITIONS = 24
PARAMETERS = {
    "periodic": np.array([np.log(1.3), np.log(0.7), np.log(1.1)]),
    "rational_quadratic": np.array([np.log(1.2), np.log(0.9), np.log(1.4)]),
}
DIRECTION = np.array([0.11, -0.07, 0.03])
OPERATIONS = ("matrix", "matrix_jvp", "parameter_vjp", "parameter_hvp", "input_derivatives")
FIELDS = (
    "workload", "operation", "backend", "device", "status", "n_samples",
    "n_features", "repetitions", "seconds_per_operation", "value", "max_abs_error",
    "oracle", "python_version", "numpy_version", "fortml_revision",
    "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(1, N + 1, dtype=np.float64)[:, None]
    j = np.arange(1, D + 1, dtype=np.float64)[None, :]
    points = np.sin(0.013 * i + 0.17 * j) + 0.2 * np.cos(0.007 * i * j)
    rows = np.arange(1, N + 1, dtype=np.float64)[:, None]
    cols = np.arange(1, N + 1, dtype=np.float64)[None, :]
    cotangent = np.sin(0.003 * (rows + 2.0 * cols))
    return points, cotangent


def kernel(points_a: np.ndarray, points_b: np.ndarray, name: str, parameters: np.ndarray) -> np.ndarray:
    delta = points_a[:, None, :] - points_b[None, :, :]
    squared_distance = np.sum(delta * delta, axis=2)
    variance, lengthscale, third = np.exp(parameters)
    if name == "periodic":
        argument = np.pi * np.sqrt(squared_distance) / third
        return variance * np.exp(-2.0 * np.sin(argument) ** 2 / lengthscale ** 2)
    denominator = 1.0 + squared_distance / (2.0 * third * lengthscale ** 2)
    return variance * denominator ** (-third)


def scalar_value(a: np.ndarray, b: np.ndarray, name: str, parameters: np.ndarray) -> float:
    return float(kernel(a[None, :], b[None, :], name, parameters)[0, 0])


def parameter_derivatives(points_a: np.ndarray, points_b: np.ndarray, name: str,
                          parameters: np.ndarray) -> tuple[np.ndarray, ...]:
    delta = points_a[:, None, :] - points_b[None, :, :]
    squared_distance = np.sum(delta * delta, axis=2)
    value = kernel(points_a, points_b, name, parameters)
    _, lengthscale, third = np.exp(parameters)
    inverse_length_squared = 1.0 / lengthscale ** 2
    if name == "periodic":
        argument = np.pi * np.sqrt(squared_distance) / third
        sine = np.sin(argument)
        cosine = np.cos(argument)
        return (value, value * 4.0 * inverse_length_squared * sine ** 2,
                value * 4.0 * inverse_length_squared * argument * sine * cosine)
    tail = squared_distance / (2.0 * third * lengthscale ** 2)
    denominator = 1.0 + tail
    return (value, value * 2.0 * third * tail / denominator,
            value * third * (tail / denominator - np.log(denominator)))


def oracle(name: str) -> dict[str, float]:
    points, cotangent = fixture()
    parameters = PARAMETERS[name]
    matrix = kernel(points, points, name, parameters)
    h = 1.0e-5
    matrix_plus = kernel(points, points, name, parameters + h * DIRECTION)
    matrix_minus = kernel(points, points, name, parameters - h * DIRECTION)
    matrix_jvp = (matrix_plus - matrix_minus) / (2.0 * h)

    def weighted_gradient(theta: np.ndarray) -> np.ndarray:
        return np.asarray([np.sum(cotangent * derivative)
                           for derivative in parameter_derivatives(points, points, name, theta)])

    parameter_vjp = weighted_gradient(parameters)
    parameter_hvp = (weighted_gradient(parameters + h * DIRECTION) - weighted_gradient(parameters - h * DIRECTION)) / (2.0 * h)

    a, b = points[0], points[1]
    value = scalar_value(a, b, name, parameters)
    input_h = 2.0e-5
    gradient_a = np.zeros(D); gradient_b = np.zeros(D); mixed = np.zeros((D, D))
    for k in range(D):
        ap = a.copy(); am = a.copy(); ap[k] += input_h; am[k] -= input_h
        gradient_a[k] = (scalar_value(ap, b, name, parameters) - scalar_value(am, b, name, parameters)) / (2.0 * input_h)
        bp = b.copy(); bm = b.copy(); bp[k] += input_h; bm[k] -= input_h
        gradient_b[k] = (scalar_value(a, bp, name, parameters) - scalar_value(a, bm, name, parameters)) / (2.0 * input_h)
        for q in range(D):
            ap = a.copy(); am = a.copy(); ap[q] += input_h; am[q] -= input_h
            mixed[q, k] = ((scalar_value(ap, bp, name, parameters) - scalar_value(ap, bm, name, parameters)) -
                           (scalar_value(am, bp, name, parameters) - scalar_value(am, bm, name, parameters))) / (4.0 * input_h ** 2)
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
    result.update({"n_samples": N, "n_features": D, "repetitions": REPETITIONS, **values})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path(__file__).resolve().parents[2] / "fortml")
    parser.add_argument("--output", type=Path, default=Path("results/kernel_catalog.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    metadata = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(args.fortml), "benchmark_revision": revision(root, (args.output,)),
        "compiler": "gfortran", "flags": "-O3",
    }
    expected = {name: oracle(name) for name in PARAMETERS}
    rows: list[dict[str, Any]] = []
    for name, values in expected.items():
        for operation in OPERATIONS:
            rows.append(row(metadata, workload=f"kernel_catalog_{name}", operation=operation,
                            backend="numpy_oracle", device="cpu", status="pass", value=values[operation],
                            max_abs_error=0.0, seconds_per_operation="", oracle="independent NumPy central-product oracle",
                            notes="reference checksum; not a timing"))
            rows.append(row(metadata, workload=f"kernel_catalog_{name}", operation=operation,
                            backend="fortml", device="cuda", status="unavailable", value="", max_abs_error="",
                            seconds_per_operation="", oracle="", notes="typed refusal: resident CUDA ABI lacks third leaf parameter"))

    target = args.fortml / "app" / "fortml_bench_kernel_catalog.f90"
    environment = os.environ.copy(); environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=args.fortml, env=environment, text=True, capture_output=True)
    if target.is_file() and build.returncode == 0:
        run = subprocess.run(["fo", "exec", "--no-build", "fortml_bench_kernel_catalog"], cwd=args.fortml, env=environment, text=True, capture_output=True)
        records: dict[tuple[str, str], tuple[float, float]] = {}
        if run.returncode == 0:
            for line in run.stdout.splitlines():
                fields = [part.strip() for part in line.split(",")]
                if len(fields) == 5 and fields[0] == "kernel_catalog":
                    records[(fields[1], fields[2])] = (float(fields[3]), float(fields[4]))
        for name, values in expected.items():
            for operation in OPERATIONS:
                if (name, operation) not in records:
                    continue
                seconds, actual = records[(name, operation)]
                error = abs(actual - values[operation])
                if error > 2.0e-6 * max(1.0, abs(values[operation])):
                    raise RuntimeError(f"FortML checksum mismatch {name}/{operation}: {error:.3e}")
                rows.append(row(metadata, workload=f"kernel_catalog_{name}", operation=operation,
                                backend="fortml", device="cpu", status="pass", value=actual,
                                max_abs_error=error, seconds_per_operation=seconds,
                                oracle="independent NumPy central-product oracle", notes="release app; host CPU timing"))
    else:
        for name in expected:
            for operation in OPERATIONS:
                rows.append(row(metadata, workload=f"kernel_catalog_{name}", operation=operation,
                                backend="fortml", device="cpu", status="unavailable", value="", max_abs_error="",
                                seconds_per_operation="", oracle="", notes="release app did not build; no timing retained"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
