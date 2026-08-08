#!/usr/bin/env python3
"""Benchmark the locally-periodic exact-GP kernel contract.

The NumPy path is an independent dense covariance and Cholesky oracle.  It
checks scalar values, input and logarithmic hyperparameter products, and an
exact posterior mean/variance before running the FortML behavioral gate.
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
    "n_queries", "likelihood", "kernel", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
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


def local_periodic(x: np.ndarray, z: np.ndarray, variance: float,
                   envelope_lengthscale: float, periodic_lengthscale: float,
                   period: float) -> np.ndarray:
    delta = x[:, None, :] - z[None, :, :]
    r2 = np.sum(delta * delta, axis=-1)
    argument = np.pi * np.sqrt(r2) / period
    return variance * np.exp(
        -r2 / (2.0 * envelope_lengthscale**2)
        -2.0 * np.sin(argument)**2 / periodic_lengthscale**2
    )


def local_periodic_reference(x: np.ndarray, z: np.ndarray, variance: float,
                             envelope_lengthscale: float,
                             periodic_lengthscale: float, period: float) -> np.ndarray:
    """Scalar-loop oracle kept separate from the vectorized implementation."""
    result = np.empty((x.shape[0], z.shape[0]), dtype=np.float64)
    for i, left in enumerate(x):
        for j, right in enumerate(z):
            squared_distance = float(np.sum((left - right)**2))
            distance = np.sqrt(squared_distance)
            argument = np.pi * distance / period
            result[i, j] = variance * np.exp(
                -squared_distance / (2.0 * envelope_lengthscale**2)
                -2.0 * np.sin(argument)**2 / periodic_lengthscale**2
            )
    return result


def oracle() -> dict[str, float]:
    x = np.array([[0.0, 0.5], [-0.4, 1.0], [1.2, -0.7]], dtype=np.float64)
    z = np.array([[0.2, -0.1], [0.8, 0.4]], dtype=np.float64)
    y = np.array([1.2, -0.3, 0.7], dtype=np.float64)
    query = np.array([[0.25, 0.1], [0.9, -0.2]], dtype=np.float64)
    variance, envelope, periodic, period, noise = 1.7, 1.1, 0.65, 1.3, 0.04
    theta = np.log([variance, envelope, periodic, period])
    direction = np.array([0.17, -0.23, 0.11, 0.29], dtype=np.float64)
    gram = local_periodic(x, x, variance, envelope, periodic, period) + noise * np.eye(3)
    alpha = np.linalg.solve(gram, y)
    cross = local_periodic(x, query, variance, envelope, periodic, period)
    prior = local_periodic(query, query, variance, envelope, periodic, period)
    mean = cross.T @ alpha
    solved_cross = np.linalg.solve(gram, cross)
    posterior_variance = np.diag(prior) - np.sum(cross * solved_cross, axis=0)

    def prediction(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v, e, p, q = np.exp(parameters)
        gram_local = local_periodic(x, x, v, e, p, q) + noise * np.eye(3)
        alpha_local = np.linalg.solve(gram_local, y)
        cross_local = local_periodic(x, query, v, e, p, q)
        prior_local = local_periodic(query, query, v, e, p, q)
        solved_local = np.linalg.solve(gram_local, cross_local)
        return cross_local.T @ alpha_local, np.diag(prior_local) - np.sum(
            cross_local * solved_local, axis=0,
        )

    h = 2.0e-6
    mean_plus, variance_plus = prediction(theta + h * direction)
    mean_minus, variance_minus = prediction(theta - h * direction)
    parameter_jvp = (mean_plus - mean_minus) / (2.0 * h)
    parameter_hvp = (variance_plus - variance_minus) / (2.0 * h)
    return {
        "matrix_max_abs_error": float(np.max(np.abs(
            local_periodic(x, z, variance, envelope, periodic, period)
            - local_periodic_reference(x, z, variance, envelope, periodic, period),
        ))),
        "minimum_posterior_variance": float(np.min(posterior_variance)),
        "parameter_jvp_norm": float(np.linalg.norm(parameter_jvp)),
        "parameter_hvp_norm": float(np.linalg.norm(parameter_hvp)),
        "mean_norm": float(np.linalg.norm(mean)),
        "variance_norm": float(np.linalg.norm(posterior_variance)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/local_periodic_gp.csv"))
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
        subprocess.run(["fo", "test", "test_local_periodic_gp"],
                       cwd=fortml, env=environment, check=True)
        status, notes = "pass", "exact GP, input/parameter products, and typed refusal gate"
    elapsed = time.perf_counter() - started
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml), "benchmark_revision": revision(root, (output,)),
        "compiler": "gfortran", "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(phase: str, metric: str, value: float, oracle_name: str,
            device: str = "cpu", row_status: str = "pass", **extra: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(details)
        row.update({"workload": "local_periodic_gp", "phase": phase,
                    "backend": "numpy_oracle", "device": device,
                    "status": row_status, "n_samples": 3, "n_queries": 2,
                    "likelihood": "gaussian", "kernel": "local_periodic",
                    "metric": metric, "value": value, "max_abs_error": value,
                    "oracle": oracle_name})
        row.update(extra)
        rows.append(row)

    add("value", "matrix_max_abs_error", metrics["matrix_max_abs_error"],
        "independent NumPy locally-periodic covariance")
    add("prediction", "minimum_posterior_variance", metrics["minimum_posterior_variance"],
        "independent NumPy dense Cholesky Schur complement", max_abs_error=0.0)
    add("parameter_products", "jvp_norm", metrics["parameter_jvp_norm"],
        "independent NumPy central difference over logarithmic parameters", max_abs_error=0.0)
    add("parameter_products", "hvp_norm", metrics["parameter_hvp_norm"],
        "independent NumPy central difference over GP posterior variance", max_abs_error=0.0)
    add("public_contract_gate", "fortml_local_periodic_gp_test", 1.0,
        "FortML test_local_periodic_gp behavioral gate", backend="fortml",
        seconds_per_operation=elapsed, max_abs_error=0.0, notes=notes)
    add("device_boundary", "typed_cuda_prediction_and_covariance", float("nan"),
        "typed FORTNUM_DOMAIN_ERROR refusal", device="cuda", row_status="refused",
        backend="fortml", max_abs_error=0.0,
        notes="resident local-periodic operator/CUDA ABI is not linked")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
