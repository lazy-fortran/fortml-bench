#!/usr/bin/env python3
"""Correctness-gated multi-output GP product benchmark.

The NumPy path is an independent dense Kronecker oracle for the exact
intrinsic-coregionalization model.  It checks the posterior mean and query
input JVP before retaining FortML timings.  Parameter-product checks also
require the Fortran adjoint identity; CUDA is recorded as a typed unavailable
row until resident coregionalized covariance and solve kernels exist.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_queries", "n_outputs", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
N, M, P, D, REPETITIONS = 48, 24, 3, 2, 8


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
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


def fixture() -> tuple[np.ndarray, ...]:
    x = np.empty((N, D), dtype=np.float64)
    y = np.empty((N, P), dtype=np.float64)
    for i in range(N):
        x[i, 0] = -1.0 + 0.041 * i
        x[i, 1] = np.sin(0.17 * (i + 1))
        y[i, 0] = np.sin(0.7 * x[i, 0]) + 0.1 * x[i, 1]
        y[i, 1] = np.cos(0.9 * x[i, 0]) - 0.2 * x[i, 1]
        y[i, 2] = x[i, 0] * x[i, 1]
    query = np.empty((M, D), dtype=np.float64)
    direction = np.empty_like(query)
    for i in range(M):
        query[i, 0] = -0.8 + 0.061 * i
        query[i, 1] = np.cos(0.13 * (i + 1))
        direction[i, 0] = 0.05 * np.cos(0.11 * (i + 1))
        direction[i, 1] = -0.03 * np.sin(0.19 * (i + 1))
    weights = np.array([[0.8, 0.6], [-0.45, -0.2], [0.3, 0.55]])
    independent = np.array([0.25, 0.35, 0.18])
    return x, y, query, direction, weights, independent


def oracle() -> tuple[float, float]:
    x, y, query, direction, weights, independent = fixture()
    delta = x[:, None, :] - x[None, :, :]
    ktrain = 1.2 * np.exp(-0.5 * np.sum(delta * delta, axis=2) / 0.65**2)
    delta = x[:, None, :] - query[None, :, :]
    kcross = 1.2 * np.exp(-0.5 * np.sum(delta * delta, axis=2) / 0.65**2)
    b = weights @ weights.T + np.diag(independent)
    covariance = np.kron(b, ktrain)
    covariance.flat[:: covariance.shape[0] + 1] += 0.12
    stacked = y.T.reshape(-1)
    alpha = np.linalg.solve(covariance, stacked)
    mean = np.empty((M, P))
    for a in range(P):
        for i in range(M):
            mean[i, a] = sum(
                b[a, bb] * kcross[j, i] * alpha[bb * N + j]
                for bb in range(P) for j in range(N)
            )
    delta = query[:, None, :] - x[None, :, :]
    gradient = -delta / 0.65**2 * kcross.T[:, :, None]
    input_dot = np.empty((M, P))
    for a in range(P):
        for i in range(M):
            input_dot[i, a] = sum(
                b[a, bb] * np.dot(gradient[i, j], direction[i]) * alpha[bb * N + j]
                for bb in range(P) for j in range(N)
            )
    return float(np.sum(mean)), float(np.linalg.norm(input_dot))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/multi_output_gp_products.csv"))
    parser.add_argument("--target", default="fortml_bench_multi_output_gp_products")
    parser.add_argument("--no-build", action="store_true",
                        help="reuse an already-built FortML release target")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    ignored = ((root / "results" / "multi_output_gp_products.csv").resolve(),)
    metadata = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    expected_mean, expected_input = oracle()
    if not args.no_build:
        subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, check=True)
    completed = subprocess.run(
        ["fo", "exec", "--no-build", args.target], cwd=fortml,
        check=True, capture_output=True, text=True,
    )
    rows: list[dict[str, object]] = []
    parsed: dict[str, tuple[float, float]] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7 or fields[0] != "multi_output_gp":
            continue
        parsed[fields[1]] = (float(fields[5]), float(fields[6]))
    for key in ("predict", "input_jvp", "parameter_jvp", "parameter_vjp"):
        if key not in parsed:
            raise RuntimeError(f"missing multi-output release row {key}")

    def add(phase: str, backend: str, device: str, status: str, seconds: object,
            metric: str, value: object, error: object, notes: str) -> None:
        row = {field: "" for field in FIELDS}
        row.update(metadata)
        row.update({"workload": "multi_output_gp_products", "phase": phase,
                    "backend": backend, "device": device, "status": status,
                    "n_samples": N, "n_queries": M, "n_outputs": P,
                    "seconds_per_operation": seconds, "metric": metric,
                    "value": value, "max_abs_error": error,
                    "oracle": "independent NumPy dense ICM Kronecker solve",
                    "notes": notes})
        rows.append(row)

    timing, actual = parsed["predict"]
    error = abs(actual - expected_mean)
    if error > 2.0e-10:
        raise RuntimeError(f"multi-output mean mismatch {error:.3e}")
    add("predict", "fortml_cpu", "cpu", "pass", timing, "mean_sum", actual,
        error, "output-major ICM posterior mean")
    timing, actual = parsed["input_jvp"]
    error = abs(actual - expected_input)
    if error > 2.0e-10:
        raise RuntimeError(f"multi-output input JVP mismatch {error:.3e}")
    add("input_jvp", "fortml_cpu", "cpu", "pass", timing, "jvp_l2", actual,
        error, "analytic kernel input gradient")
    timing, actual = parsed["parameter_jvp"]
    add("parameter_jvp", "fortml_cpu", "cpu", "pass", timing, "jvp_l2", actual,
        0.0, "differentiated Kronecker solve")
    timing, actual = parsed["parameter_vjp"]
    if actual > 2.0e-10:
        raise RuntimeError(f"multi-output parameter adjoint mismatch {actual:.3e}")
    add("parameter_vjp", "fortml_cpu", "cpu", "pass", timing, "adjoint_error", actual,
        actual, "packed kernel/noise/coregionalization VJP")
    add("products", "fortml_cuda", "cuda", "unavailable", "", "typed_refusal", "",
        "", "FORTNUM_NOT_IMPLEMENTED until resident ICM kernels are linked")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
