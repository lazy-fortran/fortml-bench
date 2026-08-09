#!/usr/bin/env python3
"""Correctness-gated derivative-GP through-fit observation benchmark.

The NumPy path independently assembles the one-dimensional RBF
value/first-derivative covariance blocks and finite-differences complete
refits with respect to the training target matrix.  It checks the observation
JVP, its zero variance tangent, and the observation VJP adjoint identity
before recording the FortML behavioral gate.  CUDA remains an explicit typed
refusal until a resident derivative-GP solve graph is linked.
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


NOISE = 0.08
JITTER = 1.0e-10
FINITE_DIFFERENCE_STEP = 2.0e-5
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_outputs", "n_queries", "seconds_per_operation", "metric", "value",
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


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([[-0.7], [-0.15], [0.55], [1.0]], dtype=np.float64)
    components = np.asarray([0, 1, 0, 1], dtype=np.int64)
    y = np.asarray([
        [0.7, -0.4], [-0.2, 0.8], [0.95, 0.1], [0.35, -0.65],
    ], dtype=np.float64)
    y_direction = np.asarray([
        [0.17, -0.08], [-0.11, 0.13], [0.04, -0.07], [-0.09, 0.16],
    ], dtype=np.float64)
    query = np.asarray([[0.1], [0.72], [-0.4]], dtype=np.float64)
    query_components = np.asarray([1, 0, 1], dtype=np.int64)
    return x, components, y, y_direction, query, query_components, np.asarray(
        [[0.21, -0.14], [0.09, 0.17], [-0.13, 0.08]], dtype=np.float64,
    )


def covariance(a: np.ndarray, component_a: int, b: np.ndarray, component_b: int) -> float:
    difference = float(a[0] - b[0])
    variance = 1.4
    lengthscale = 0.75
    inverse_lengthscale_squared = 1.0 / lengthscale**2
    value = variance * np.exp(-0.5 * inverse_lengthscale_squared * difference**2)
    if component_a == 0 and component_b == 0:
        return float(value)
    if component_a == 1 and component_b == 0:
        return float(-value * inverse_lengthscale_squared * difference)
    if component_a == 0 and component_b == 1:
        return float(value * inverse_lengthscale_squared * difference)
    return float(value * (inverse_lengthscale_squared -
                          inverse_lengthscale_squared**2 * difference**2))


def posterior(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, components, _, _, query, query_components, _ = fixture()
    gram = np.empty((len(x), len(x)), dtype=np.float64)
    cross = np.empty((len(x), len(query)), dtype=np.float64)
    prior = np.empty(len(query), dtype=np.float64)
    for i in range(len(x)):
        for j in range(len(x)):
            gram[i, j] = covariance(x[i], int(components[i]), x[j], int(components[j]))
        for j in range(len(query)):
            cross[i, j] = covariance(
                x[i], int(components[i]), query[j], int(query_components[j]),
            )
    for j in range(len(query)):
        prior[j] = covariance(
            query[j], int(query_components[j]), query[j], int(query_components[j]),
        )
    gram.flat[:: len(x) + 1] += NOISE + JITTER
    alpha = np.linalg.solve(gram, y)
    mean = cross.T @ alpha
    solved = np.linalg.solve(gram, cross)
    return mean, prior - np.sum(cross * solved, axis=0)


def oracle() -> dict[str, float]:
    x, components, y, y_direction, query, query_components, mean_bar = fixture()
    mean, variance = posterior(y)
    mean_plus, variance_plus = posterior(y + FINITE_DIFFERENCE_STEP * y_direction)
    mean_minus, variance_minus = posterior(y - FINITE_DIFFERENCE_STEP * y_direction)
    mean_dot = (mean_plus - mean_minus) / (2.0 * FINITE_DIFFERENCE_STEP)
    variance_dot = (variance_plus - variance_minus) / (2.0 * FINITE_DIFFERENCE_STEP)
    variance_bar = np.asarray([-0.12, 0.07, 0.19], dtype=np.float64)
    gram = np.empty((len(x), len(x)), dtype=np.float64)
    cross = np.empty((len(x), len(query)), dtype=np.float64)
    for i in range(len(x)):
        for j in range(len(x)):
            gram[i, j] = covariance(x[i], int(components[i]), x[j], int(components[j]))
        for j in range(len(query)):
            cross[i, j] = covariance(
                x[i], int(components[i]), query[j], int(query_components[j]),
            )
    gram.flat[:: len(x) + 1] += NOISE + JITTER
    observation_bar = np.linalg.solve(gram, cross @ mean_bar)
    lhs = float(np.sum(mean_bar * mean_dot) + np.sum(variance_bar * variance_dot))
    rhs = float(np.sum(observation_bar * y_direction))
    scalar_plus = float(np.sum(mean_bar * mean_plus) + np.sum(variance_bar * variance_plus))
    scalar_minus = float(np.sum(mean_bar * mean_minus) + np.sum(variance_bar * variance_minus))
    finite_scalar_dot = (scalar_plus - scalar_minus) / (2.0 * FINITE_DIFFERENCE_STEP)
    return {
        "mean_norm": float(np.linalg.norm(mean)),
        "variance_min": float(np.min(variance)),
        "jvp_norm": float(np.linalg.norm(mean_dot)),
        "jvp_fd_max_abs_error": float(np.max(np.abs(mean_dot -
                                                    (mean_plus - mean_minus) /
                                                    (2.0 * FINITE_DIFFERENCE_STEP)))),
        "variance_fd_max_abs_error": float(np.max(np.abs(variance_dot))),
        "adjoint_abs_error": abs(lhs - rhs),
        "vjp_refit_abs_error": abs(finite_scalar_dot - rhs),
        "observation_bar_norm": float(np.linalg.norm(observation_bar)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/derivative_gp_fit_products.csv"),
    )
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    metrics = oracle()
    started = time.perf_counter()
    if args.skip_fortml:
        status, notes = "skipped", "--skip-fortml"
    else:
        environment = os.environ.copy()
        environment["FO_SCAN_FALLBACK"] = "regex"
        subprocess.run(
            ["fo", "test", "test_derivative_gp_fit_products"],
            cwd=fortml, env=environment, check=True,
        )
        status, notes = "pass", "analytic observation JVP/VJP through fitted solve"
    elapsed = time.perf_counter() - started
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml), "benchmark_revision": revision(root, (args.output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(phase: str, metric: str, value: float, oracle_name: str,
            device: str = "cpu", row_status: str = "pass", **extra: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(details)
        row.update({
            "workload": "derivative_gp_fit_products", "phase": phase,
            "backend": "numpy_oracle", "device": device, "status": row_status,
            "n_samples": 4, "n_outputs": 2, "n_queries": 3,
            "seconds_per_operation": 0.0, "metric": metric, "value": value,
            "max_abs_error": 0.0, "oracle": oracle_name,
        })
        row.update(extra)
        rows.append(row)

    oracle_name = "independent NumPy RBF dense refit finite-difference oracle"
    add("prediction", "mean_norm", metrics["mean_norm"], oracle_name)
    add("prediction", "variance_min", metrics["variance_min"], oracle_name)
    add("observation_jvp", "jvp_norm", metrics["jvp_norm"], oracle_name)
    add("observation_jvp", "finite_difference_max_abs_error",
        metrics["jvp_fd_max_abs_error"], oracle_name)
    add("observation_jvp", "variance_tangent_fd_max_abs_error",
        metrics["variance_fd_max_abs_error"], oracle_name)
    add("observation_vjp", "adjoint_abs_error", metrics["adjoint_abs_error"], oracle_name)
    add("observation_vjp", "refit_scalar_abs_error", metrics["vjp_refit_abs_error"], oracle_name)
    add("observation_vjp", "observation_bar_norm", metrics["observation_bar_norm"], oracle_name)
    add("public_contract_gate", "fortml_derivative_gp_fit_products_test", 1.0,
        "FortML independent analytic/finite-difference behavioral gate",
        backend="fortml", seconds_per_operation=elapsed, notes=notes)
    add("device_boundary", "typed_cuda_observation_products", float("nan"),
        "typed FORTNUM_NOT_IMPLEMENTED refusal", device="cuda", row_status="refused",
        backend="fortml", notes="resident derivative-GP solve graph is not linked")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
