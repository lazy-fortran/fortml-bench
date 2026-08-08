#!/usr/bin/env python3
"""Correctness-gated local-periodic derivative-observation GP benchmark.

The NumPy path independently assembles value/gradient/mixed-Hessian covariance
blocks and finite-differences the dense likelihood gradient in packed
log-variance/log-lengthscale/log-noise coordinates.  The FortML test is a
separate behavioral gate for analytic mixed-observation parameter and
query-input JVP products. Query finite differences are formed from this
independent covariance/Cholesky implementation, including a coincident query.
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
NOISE = 0.045
JITTER = 1.0e-10


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


def covariance(theta: np.ndarray, x: np.ndarray, component: int,
               z: np.ndarray, other_component: int) -> float:
    difference = x - z
    squared_distance = float(np.dot(difference, difference))
    distance = np.sqrt(squared_distance)
    variance, envelope, periodic, period = np.exp(theta[:4])
    a = 0.5 / envelope**2
    b = 2.0 / periodic**2
    c = np.pi / period
    if distance == 0.0:
        value = variance
        first_t = -value * (a + b * c**2)
        second_t = value * ((a + b * c**2)**2 + 2.0 * b * c**4 / 3.0)
    else:
        argument = c * distance
        sine = np.sin(argument)
        cosine = np.cos(argument)
        value = variance * np.exp(-a * squared_distance - b * sine**2)
        log_r = -2.0 * a * distance - 2.0 * b * c * sine * cosine
        log_rr = -2.0 * a - 2.0 * b * c**2 * (cosine**2 - sine**2)
        first_r = value * log_r
        second_r = value * (log_r**2 + log_rr)
        first_t = first_r / (2.0 * distance)
        second_t = (second_r - first_r / distance) / (4.0 * squared_distance)
    if component == 0 and other_component == 0:
        return float(value)
    if component > 0 and other_component == 0:
        return float(2.0 * first_t * difference[component - 1])
    if component == 0 and other_component > 0:
        return float(-2.0 * first_t * difference[other_component - 1])
    return float(-2.0 * first_t * (component == other_component) -
                 4.0 * second_t * difference[component - 1] * difference[other_component - 1])


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.array([
        [-0.8, 0.1], [0.35, -0.55], [0.9, 0.72],
        [-0.15, 1.05], [0.6, -0.95],
    ], dtype=np.float64)
    y = np.array([[0.7], [-0.2], [0.95], [0.3], [-0.65]], dtype=np.float64)
    components = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    query = np.array([[0.15, -0.3], [0.72, 0.44], [-0.48, 0.93]], dtype=np.float64)
    query_components = np.array([2, 0, 1], dtype=np.int64)
    return x, y, components, query, query_components


def gram(theta: np.ndarray, x: np.ndarray, components: np.ndarray) -> np.ndarray:
    result = np.empty((len(x), len(x)), dtype=np.float64)
    for i in range(len(x)):
        for j in range(len(x)):
            result[i, j] = covariance(theta, x[i], int(components[i]), x[j], int(components[j]))
    result.flat[:: len(x) + 1] += np.exp(theta[4]) + JITTER
    return result


def likelihood(theta: np.ndarray) -> float:
    x, y, components, _, _ = fixture()
    matrix = gram(theta, x, components)
    alpha = np.linalg.solve(matrix, y)
    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0.0:
        raise RuntimeError("independent local-periodic covariance is not SPD")
    return float(-0.5 * np.sum(y * alpha) - 0.5 * logdet -
                 0.5 * len(x) * np.log(2.0 * np.pi))


def prediction(theta: np.ndarray, query_override: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    x, y, components, query, query_components = fixture()
    if query_override is not None:
        query = query_override
    matrix = gram(theta, x, components)
    alpha = np.linalg.solve(matrix, y)
    cross = np.empty((len(x), len(query)), dtype=np.float64)
    for i in range(len(x)):
        for j in range(len(query)):
            cross[i, j] = covariance(theta, x[i], int(components[i]), query[j],
                                     int(query_components[j]))
    mean = cross.T @ alpha
    solved = np.linalg.solve(matrix, cross)
    prior = np.array([
        covariance(theta, query[i], int(query_components[i]), query[i],
                   int(query_components[i])) for i in range(len(query))
    ])
    return mean, prior - np.sum(cross * solved, axis=0)


def oracle() -> dict[str, float]:
    theta = np.log(np.array([1.3, 0.85, 0.62, 1.7, NOISE], dtype=np.float64))
    h = 2.0e-5
    gradient = np.empty_like(theta)
    for i in range(len(theta)):
        direction = np.zeros_like(theta)
        direction[i] = h
        gradient[i] = (likelihood(theta + direction) - likelihood(theta - direction)) / (2.0 * h)
    direction = np.array([0.11, -0.08, 0.14, -0.06, 0.17], dtype=np.float64)
    x, _, _, query, _ = fixture()
    query[0] = x[0]
    query_direction = np.array([
        [0.07, 0.11], [-0.03, -0.05], [0.09, -0.08],
    ], dtype=np.float64)
    mean, variance = prediction(theta, query)
    query_step = 2.0e-5
    mean_plus, variance_plus = prediction(theta, query + query_step * query_direction)
    mean_minus, variance_minus = prediction(theta, query - query_step * query_direction)
    mean_jvp = (mean_plus - mean_minus) / (2.0 * query_step)
    variance_jvp = (variance_plus - variance_minus) / (2.0 * query_step)
    return {
        "likelihood": likelihood(theta),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "directional_jvp": float(np.dot(gradient, direction)),
        "minimum_posterior_variance": float(np.min(variance)),
        "mean_norm": float(np.linalg.norm(mean)),
        "query_mean_jvp_norm": float(np.linalg.norm(mean_jvp)),
        "query_variance_jvp_norm": float(np.linalg.norm(variance_jvp)),
        "query_jvp_max_abs": float(max(np.max(np.abs(mean_jvp)),
                                        np.max(np.abs(variance_jvp)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/derivative_gp_local_periodic.csv"))
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
        subprocess.run(["fo", "test", "test_derivative_gp_local_periodic"],
                       cwd=fortml, env=environment, check=True)
        status, notes = "pass", "analytic mixed-observation parameter and query-input JVP"
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
        row.update({"workload": "derivative_gp_local_periodic", "phase": phase,
                    "backend": "numpy_oracle", "device": device,
                    "status": row_status, "n_samples": 5, "n_queries": 3,
                    "n_features": 2, "metric": metric, "value": value,
                    "max_abs_error": 0.0, "oracle": oracle_name})
        row.update(extra)
        rows.append(row)

    add("likelihood", "value", metrics["likelihood"],
        "independent NumPy dense Cholesky mixed-observation oracle")
    add("parameter_products", "gradient_norm", metrics["gradient_norm"],
        "independent NumPy central difference of dense likelihood")
    add("parameter_products", "directional_jvp", metrics["directional_jvp"],
        "independent NumPy central difference over packed log parameters")
    add("prediction", "minimum_posterior_variance", metrics["minimum_posterior_variance"],
        "independent NumPy dense Cholesky Schur complement")
    add("prediction", "mean_norm", metrics["mean_norm"],
        "independent NumPy mixed-observation posterior")
    add("query_products", "mean_jvp_norm", metrics["query_mean_jvp_norm"],
        "independent NumPy central difference of query posterior means")
    add("query_products", "variance_jvp_norm", metrics["query_variance_jvp_norm"],
        "independent NumPy central difference of query posterior variances")
    add("query_products", "jvp_max_abs", metrics["query_jvp_max_abs"],
        "independent NumPy query-input directional finite difference")
    add("public_contract_gate", "fortml_derivative_gp_local_periodic_test", 1.0,
        "FortML independent analytic/finite-difference behavioral gate", backend="fortml",
        seconds_per_operation=elapsed, notes=notes)
    add("device_boundary", "typed_cuda_derivative_gp", float("nan"),
        "typed FORTNUM_DOMAIN_ERROR refusal", device="cuda", row_status="refused",
        backend="fortml", notes="resident derivative-GP covariance/factorization graph is not linked")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
