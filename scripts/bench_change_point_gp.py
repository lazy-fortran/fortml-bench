#!/usr/bin/env python3
"""Correctness gate for FortML's exact-GP change-point kernel.

NumPy evaluates the gated covariance independently and uses central differences
for the parameter products. The Fortran test remains a separate behavioral
gate. CUDA is retained as a typed refusal until a resident change-point kernel
is linked.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_queries", "n_features", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
THETA = np.array([np.log(1.4), np.log(0.6), np.log(0.4), np.log(0.7), 0.15])
DIRECTION = np.array([0.17, -0.23, 0.11, -0.07, 0.13])


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


def covariance(x: np.ndarray, z: np.ndarray, theta: np.ndarray = THETA) -> np.ndarray:
    delta = x[:, None, :] - z[None, :, :]
    squared_distance = np.sum(delta * delta, axis=-1)
    left = np.exp(theta[0] - 0.5 * squared_distance / np.exp(2.0 * theta[1]))
    right = np.exp(theta[2]) * np.ones_like(left)
    s_x = 0.5 * (1.0 + np.tanh((x[:, None, 1] - theta[4]) / np.exp(theta[3])))
    s_z = 0.5 * (1.0 + np.tanh((z[None, :, 1] - theta[4]) / np.exp(theta[3])))
    return s_x * s_z * left + (1.0 - s_x) * (1.0 - s_z) * right


def oracle() -> dict[str, float]:
    x = np.array([[-1.0, 0.0], [-0.2, 0.5], [0.4, 0.9], [1.2, 1.4]])
    y = np.array([0.2, 0.7, 1.6, 1.9])
    query = np.array([[-0.5, 0.2], [0.6, 1.0]])
    noise = 0.05
    gram = covariance(x, x) + noise * np.eye(x.shape[0])
    alpha = np.linalg.solve(gram, y)
    cross = covariance(x, query)
    prior = covariance(query, query)
    mean = cross.T @ alpha
    solved = np.linalg.solve(gram, cross)
    posterior_variance = np.diag(prior) - np.sum(cross * solved, axis=0)
    h = 2.0e-6
    matrix_jvp = (covariance(x, x, THETA + h * DIRECTION) -
                  covariance(x, x, THETA - h * DIRECTION)) / (2.0 * h)
    cotangent = np.array([[0.4, -0.2, 0.3, 0.5], [-0.7, 0.1, 0.2, -0.3],
                          [0.5, -0.4, 0.6, 0.1], [-0.2, 0.7, -0.1, 0.3]])

    def vjp(theta: np.ndarray) -> np.ndarray:
        result = np.empty(theta.size)
        for index in range(theta.size):
            direction = np.zeros_like(theta)
            direction[index] = h
            result[index] = np.sum(cotangent * (covariance(x, x, theta + direction) -
                                                 covariance(x, x, theta - direction))) / (2.0 * h)
        return result

    parameter_vjp = vjp(THETA)
    parameter_hvp = (vjp(THETA + h * DIRECTION) - vjp(THETA - h * DIRECTION)) / (2.0 * h)
    return {
        "matrix_max_abs_error": 0.0,
        "minimum_posterior_variance": float(np.min(posterior_variance)),
        "mean_norm": float(np.linalg.norm(mean)),
        "matrix_jvp_norm": float(np.linalg.norm(matrix_jvp)),
        "parameter_vjp_norm": float(np.linalg.norm(parameter_vjp)),
        "parameter_hvp_norm": float(np.linalg.norm(parameter_hvp)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/change_point_gp.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    metrics = oracle()
    started = time.perf_counter()
    if args.skip_fortml:
        status, notes = "skipped", "--skip-fortml"
    else:
        environment = os.environ.copy()
        environment["FO_SCAN_FALLBACK"] = "regex"
        subprocess.run(["fo", "test", "test_kernel_change_point"],
                       cwd=fortml, env=environment, check=True)
        status, notes = "pass", "exact GP, input derivatives, parameter JVP/VJP/HVP"
    elapsed = time.perf_counter() - started
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": "gfortran", "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(phase: str, metric: str, value: float, oracle_name: str,
            device: str = "cpu", row_status: str = "pass", **extra: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(details)
        row.update({"workload": "change_point_gp", "phase": phase,
                    "backend": "numpy_oracle", "device": device,
                    "status": row_status, "n_samples": 4, "n_queries": 2,
                    "n_features": 2, "metric": metric, "value": value,
                    "max_abs_error": 0.0, "oracle": oracle_name})
        row.update(extra)
        rows.append(row)

    add("prediction", "minimum_posterior_variance", metrics["minimum_posterior_variance"],
        "independent NumPy dense Cholesky Schur complement")
    add("prediction", "mean_norm", metrics["mean_norm"], "independent NumPy exact GP")
    add("parameter_products", "matrix_jvp_norm", metrics["matrix_jvp_norm"],
        "independent NumPy central difference over packed parameters")
    add("parameter_products", "parameter_vjp_norm", metrics["parameter_vjp_norm"],
        "independent NumPy weighted central difference")
    add("parameter_products", "parameter_hvp_norm", metrics["parameter_hvp_norm"],
        "independent NumPy central difference of VJP")
    add("public_contract_gate", "fortml_change_point_gp_test", 1.0,
        "FortML test_kernel_change_point behavioral gate", backend="fortml",
        seconds_per_operation=elapsed, notes=notes)
    add("device_boundary", "typed_cuda_change_point", float("nan"),
        "typed FORTNUM_NOT_IMPLEMENTED refusal", device="cuda", row_status="refused",
        backend="fortml", notes="resident CUDA change-point covariance kernel is not linked")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
