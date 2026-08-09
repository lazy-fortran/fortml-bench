#!/usr/bin/env python3
"""Correctness-gated weighted nonnegative linear-regression benchmark."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


N_SAMPLES = 64
N_FEATURES = 3
N_OUTPUTS = 2
MAX_ITERATIONS = 20_000
TOLERANCE = 1.0e-11
REPETITIONS = 12
ERROR_TOLERANCE = 3.0e-8
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_outputs", "fit_intercept", "repetitions",
    "seconds_per_operation", "metric", "value", "max_abs_error", "oracle",
    "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
)


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
    x = np.empty((N_SAMPLES, N_FEATURES), dtype=np.float64)
    x_dot = np.empty_like(x)
    for column in range(N_FEATURES):
        for row in range(N_SAMPLES):
            x[row, column] = np.sin(0.071*(row+1)*(column+1)) + 0.03*np.cos(
                0.11*(row+1)+(column+1)
            )
            x_dot[row, column] = 0.07*np.cos(
                0.037*(row+1)*(column+2)
            )
    y = np.column_stack((
        0.3 + 0.9*x[:, 0] + 0.4*x[:, 1] + 0.2*x[:, 2]
        + 0.02*np.sin(0.13*np.arange(1, N_SAMPLES+1)),
        0.5 + 0.3*x[:, 0] + 0.7*x[:, 1] + 0.6*x[:, 2]
        + 0.02*np.cos(0.09*np.arange(1, N_SAMPLES+1)),
    ))
    weights = 0.5 + (np.arange(1, N_SAMPLES+1) % 5)/5.0
    u = np.column_stack((
        np.sin(0.041*np.arange(1, N_SAMPLES+1)),
        np.cos(0.053*np.arange(1, N_SAMPLES+1)),
    ))
    theta_dot = np.linspace(-0.04, 0.04, (N_FEATURES+1)*N_OUTPUTS)
    return x, y, weights, x_dot, u, theta_dot


def positive_linear(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Independent projected-gradient oracle matching the documented objective."""
    design = np.column_stack((np.ones(x.shape[0]), x))
    mass = float(np.sum(weights))
    curvature = float(np.sum(weights[:, None]*design*design))
    beta = np.zeros((N_FEATURES+1, y.shape[1]), dtype=np.float64)
    beta[0, :] = np.sum(weights[:, None]*y, axis=0)/mass
    if curvature <= np.finfo(np.float64).tiny:
        return beta
    step = mass/curvature
    for _ in range(MAX_ITERATIONS):
        residual = design @ beta-y
        gradient = design.T @ (weights[:, None]*residual)/mass
        candidate = np.maximum(beta-step*gradient, 0.0)
        candidate[0, :] = beta[0, :]-step*gradient[0, :]
        delta = float(np.max(np.abs(candidate-beta)))
        beta = candidate
        if delta <= TOLERANCE*(1.0+float(np.max(np.abs(beta)))):
            return beta
    raise RuntimeError("NumPy positive-linear oracle did not converge")


def expected() -> dict[str, np.ndarray]:
    x, y, weights, x_dot, u, theta_dot = fixture()
    coefficient = positive_linear(x, y, weights)
    design = np.column_stack((np.ones(N_SAMPLES), x))
    design_dot = np.column_stack((np.zeros(N_SAMPLES), x_dot))
    coefficient_dot = theta_dot.reshape(coefficient.shape, order="F")
    prediction = design @ coefficient
    prediction_dot = design_dot @ coefficient + design @ coefficient_dot
    theta_bar = (design.T @ u).reshape(-1, order="F")
    x_bar = u @ coefficient[1:, :].T
    vector_coefficient = positive_linear(x, y[:, :1], weights)
    return {
        "fit_matrix": coefficient,
        "fit_vector": vector_coefficient,
        "predict_matrix": prediction,
        "predict_vector": (design @ vector_coefficient)[:, 0],
        "predict_jvp": prediction_dot,
        "predict_vjp_theta": theta_bar,
        "predict_vjp_x": x_bar,
    }


def parse_app(stdout: str) -> tuple[dict[str, np.ndarray], dict[str, float], str]:
    arrays: dict[str, list[float]] = {}
    times: dict[str, float] = {}
    cuda = ""
    for line in stdout.splitlines():
        fields = line.strip().split(",")
        if len(fields) == 3 and fields[:2] == ["positive_linear", "cuda_status"]:
            cuda = fields[2]
            continue
        if len(fields) != 5 or fields[0] != "positive_linear":
            continue
        _, name, index, raw_value, raw_seconds = fields
        arrays.setdefault(name, []).append(float(raw_value))
        times.setdefault(name, float(raw_seconds))
    return {name: np.asarray(values) for name, values in arrays.items()}, times, cuda


def run_app(fortml: Path) -> tuple[dict[str, np.ndarray], dict[str, float], str, float]:
    env = os.environ.copy()
    env.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, env=env,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    started = time.perf_counter()
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_positive_linear_regression"],
        cwd=fortml, env=env, check=True, capture_output=True, text=True,
    )
    elapsed = time.perf_counter()-started
    return (*parse_app(completed.stdout), elapsed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/positive_linear_regression.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/POSITIVE_LINEAR_REGRESSION.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    target = expected()
    actual, times, cuda, wall_seconds = run_app(fortml)
    missing = sorted(set(target)-set(actual))
    if missing:
        raise RuntimeError(f"release app omitted workloads: {missing}")
    rows: list[dict[str, Any]] = []
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (args.output.resolve(), args.report.resolve())),
        "compiler": "gfortran", "flags": "-O3",
    }
    for name, expected_value in target.items():
        values = actual[name]
        expected_value = np.asarray(expected_value).reshape(-1, order="F")
        error = float(np.max(np.abs(values-expected_value)))
        if values.size != expected_value.size:
            raise RuntimeError(f"{name} size mismatch: {values.size} vs {expected_value.size}")
        if error > ERROR_TOLERANCE:
            raise RuntimeError(f"{name} NumPy mismatch: {error:.3e}")
        rows.append({**details, "workload": "positive_linear", "phase": name,
                     "backend": "fortml", "device": "cpu", "status": "pass",
                     "n_samples": N_SAMPLES, "n_features": N_FEATURES,
                     "n_outputs": N_OUTPUTS if "vector" not in name else 1,
                     "fit_intercept": "true", "repetitions": REPETITIONS,
                     "seconds_per_operation": times.get(name, wall_seconds),
                     "metric": "max_abs_error", "value": float(np.linalg.norm(values)),
                     "max_abs_error": error,
                     "oracle": "independent NumPy projected-gradient nonnegative least squares",
                     "notes": "feature coefficients constrained; fixed-state products"})
    if cuda != "unavailable":
        raise RuntimeError(f"unexpected CUDA contract: {cuda!r}")
    rows.append({**details, "workload": "positive_linear", "phase": "capability_check",
                 "backend": "fortml", "device": "cuda", "status": "unavailable",
                 "n_samples": N_SAMPLES, "n_features": N_FEATURES,
                 "n_outputs": N_OUTPUTS, "fit_intercept": "true", "repetitions": REPETITIONS,
                 "seconds_per_operation": wall_seconds, "metric": "status", "value": 3.0,
                 "max_abs_error": "", "oracle": "declared resident-device contract",
                 "notes": "FORTNUM_NOT_IMPLEMENTED; no host fallback"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    max_error = max(float(row["max_abs_error"]) for row in rows[:-1])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# Positive-constrained linear regression\n\n"
        "This correctness-gated lane compares weighted multi-output least squares "
        "with nonnegative feature coefficients against an independent NumPy "
        "projected-gradient oracle. It checks complete fitted, prediction, JVP, "
        "and VJP arrays, then records the typed CUDA refusal.\n\n"
        f"- FortML revision: `{details['fortml_revision']}`\n"
        f"- Benchmark revision: `{details['benchmark_revision']}`\n"
        f"- Compiler/flags: `{details['compiler']} {details['flags']}`\n"
        f"- Release wall time: `{wall_seconds:.6g}` s\n"
        f"- Maximum oracle error: `{max_error:.6g}`\n"
        f"- Raw record: [`{args.output}`]({args.output.name})\n",
    )
    print(f"wrote {args.output} ({len(rows)} rows)")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
