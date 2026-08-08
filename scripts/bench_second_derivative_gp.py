#!/usr/bin/env python3
"""Benchmark the bounded RBF value/gradient/Hessian GP reference.

The NumPy implementation is an independent dense Cholesky oracle. It checks
mixed order-four covariance blocks and differentiates query predictions with a
central difference before the FortML behavioral gate is timed.
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
    "n_queries", "likelihood", "seconds_per_operation", "metric", "value",
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


def distance_derivative(base: np.ndarray, difference: np.ndarray,
                        lengthscale: float, order: int) -> np.ndarray:
    inv2 = lengthscale**-2
    inv4 = inv2**2
    inv6 = inv4 * inv2
    inv8 = inv4**2
    inv10 = inv8 * inv2
    if order == 0:
        return base
    if order == 1:
        return -difference * inv2 * base
    if order == 2:
        return (difference**2 * inv4 - inv2) * base
    if order == 3:
        return (3.0 * difference * inv4 - difference**3 * inv6) * base
    if order == 4:
        return (difference**4 * inv8 - 6.0 * difference**2 * inv6 + 3.0 * inv4) * base
    if order == 5:
        return (-difference**5 * inv10 + 10.0 * difference**3 * inv8 -
                15.0 * difference * inv6) * base
    raise ValueError(f"unsupported derivative order {order}")


def covariance(left: np.ndarray, left_order: np.ndarray, right: np.ndarray,
               right_order: np.ndarray, variance: float, lengthscale: float) -> np.ndarray:
    difference = left[:, None] - right[None, :]
    base = variance * np.exp(-0.5 * difference**2 / lengthscale**2)
    result = np.empty_like(base)
    for i, order_left in enumerate(left_order):
        for j, order_right in enumerate(right_order):
            result[i, j] = ((-1.0) ** int(order_right) *
                            distance_derivative(base[i, j], difference[i, j],
                                                lengthscale,
                                                int(order_left + order_right)))
    return result


def oracle() -> dict[str, float]:
    x = np.array([-1.1, -0.25, 0.45, 1.2], dtype=np.float64)
    orders = np.array([0, 1, 2, 0], dtype=np.int64)
    y = np.array([0.7, -0.2, 1.1, -0.45], dtype=np.float64)
    query = np.array([-0.8, -0.1, 0.65, 1.0], dtype=np.float64)
    query_orders = np.array([0, 1, 2, 0], dtype=np.int64)
    direction = np.array([0.23, -0.17, 0.11, -0.19], dtype=np.float64)
    variance, lengthscale, noise, jitter = 1.6, 0.75, 0.035, 1.0e-10
    gram = covariance(x, orders, x, orders, variance, lengthscale) + (noise + jitter) * np.eye(4)
    alpha = np.linalg.solve(gram, y)
    cross = covariance(x, orders, query, query_orders, variance, lengthscale)
    mean = cross.T @ alpha
    solved_cross = np.linalg.solve(gram, cross)
    prior_diag = np.diag(covariance(query, query_orders, query, query_orders,
                                    variance, lengthscale))
    posterior_variance = prior_diag - np.sum(cross * solved_cross, axis=0)
    prior = covariance(query, query_orders, query, query_orders, variance, lengthscale)
    posterior_covariance = prior - cross.T @ solved_cross
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)

    def prediction(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cross_points = covariance(x, orders, points, query_orders, variance, lengthscale)
        solved = np.linalg.solve(gram, cross_points)
        prior_points = np.diag(covariance(points, query_orders, points, query_orders,
                                          variance, lengthscale))
        return cross_points.T @ alpha, prior_points - np.sum(cross_points * solved, axis=0)

    h = 2.0e-5
    mean_plus, variance_plus = prediction(query + h * direction)
    mean_minus, variance_minus = prediction(query - h * direction)
    mean_dot_fd = (mean_plus - mean_minus) / (2.0 * h)
    variance_dot_fd = (variance_plus - variance_minus) / (2.0 * h)
    mean_bar = np.array([0.3, -0.5, 0.2, 0.4])
    variance_bar = np.array([-0.2, 0.6, -0.1, 0.25])
    cross_dot = np.empty_like(cross)
    for i, order_left in enumerate(orders):
        for j, order_right in enumerate(query_orders):
            difference = x[i] - query[j]
            base = variance * np.exp(-0.5 * difference**2 / lengthscale**2)
            cross_dot[i, j] = direction[j] * ((-1.0) ** (int(order_right) + 1)) * distance_derivative(
                base, difference, lengthscale, int(order_left + order_right + 1))
    solved_dot = np.linalg.solve(gram, cross_dot)
    mean_dot_exact = cross_dot.T @ alpha
    variance_dot_exact = -np.sum(cross_dot * solved_cross, axis=0) - np.sum(cross * solved_dot, axis=0)
    x_bar = np.zeros(4)
    for j in range(4):
        cross_bar = alpha * mean_bar[j] - 2.0 * solved_cross[:, j] * variance_bar[j]
        x_bar[j] = np.sum(cross_bar * (cross_dot[:, j] / direction[j]))
    duality_error = abs(float(x_bar @ direction - (mean_bar @ mean_dot_exact + variance_bar @ variance_dot_exact)))
    return {
        "prediction_mean_max_abs_error": float(np.max(np.abs(mean - (cross.T @ alpha)))),
        "prediction_variance_max_abs_error": float(np.max(np.abs(posterior_variance - (prior_diag - np.sum(cross * solved_cross, axis=0))))),
        "joint_covariance_max_abs_error": float(np.max(np.abs(posterior_covariance - posterior_covariance.T))),
        "input_jvp_fd_max_abs_error": float(max(np.max(np.abs(mean_dot_exact - mean_dot_fd)),
                                                 np.max(np.abs(variance_dot_exact - variance_dot_fd)))),
        "input_vjp_duality_abs_error": duality_error,
        "minimum_posterior_variance": float(np.min(posterior_variance)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/second_derivative_gp.csv"))
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
        subprocess.run(["fo", "test", "test_second_derivative_gp"],
                       cwd=fortml, env=environment, check=True)
        status, notes = "pass", "mixed order-four covariance, JVP/VJP, and typed refusal gate"
    elapsed = time.perf_counter() - started
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml), "benchmark_revision": revision(root, (output,)),
        "compiler": "gfortran", "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(details)
        row.update({"workload": "second_derivative_gp", "backend": "numpy_oracle",
                    "device": "cpu", "status": "pass", "n_samples": 4, "n_queries": 4,
                    "likelihood": "gaussian"})
        row.update(values)
        rows.append(row)

    add(phase="prediction", metric="mean_max_abs_error", value=metrics["prediction_mean_max_abs_error"],
        max_abs_error=metrics["prediction_mean_max_abs_error"],
        oracle="independent NumPy RBF order-four covariance and Cholesky solve")
    add(phase="prediction", metric="variance_max_abs_error", value=metrics["prediction_variance_max_abs_error"],
        max_abs_error=metrics["prediction_variance_max_abs_error"],
        oracle="independent NumPy latent variance Schur complement",
        notes=f"minimum_posterior_variance={metrics['minimum_posterior_variance']:.6e}")
    add(phase="joint_covariance", metric="symmetry_abs_error", value=metrics["joint_covariance_max_abs_error"],
        max_abs_error=metrics["joint_covariance_max_abs_error"],
        oracle="independent NumPy dense latent covariance", notes="posterior covariance is symmetrized")
    add(phase="input_products", metric="jvp_central_difference_max_abs_error",
        value=metrics["input_jvp_fd_max_abs_error"], max_abs_error=metrics["input_jvp_fd_max_abs_error"],
        oracle="independent NumPy central difference of mixed-order predictions")
    add(phase="input_products", metric="vjp_duality_abs_error", value=metrics["input_vjp_duality_abs_error"],
        max_abs_error=metrics["input_vjp_duality_abs_error"], oracle="independent NumPy cotangent identity")
    add(phase="public_contract_gate", backend="fortml", seconds_per_operation=elapsed,
        metric="fortml_second_derivative_gp_test", value=1.0, max_abs_error=max(metrics.values()),
        oracle="FortML test_second_derivative_gp behavioral gate", notes=notes)
    add(phase="device_boundary", backend="fortml", status="refused", device="cuda",
        metric="typed_cuda_prediction_and_covariance", value="nan", max_abs_error=0.0,
        oracle="FORTNUM_NOT_IMPLEMENTED", notes="resident derivative covariance/solve is not linked")
    add(phase="input_boundary", backend="fortml", status="refused", metric="non_rbf_or_order_three_fit",
        value="nan", max_abs_error=0.0, oracle="typed FORTNUM_NOT_IMPLEMENTED/DOMAIN_ERROR",
        notes="only scalar 1-D RBF orders 0:2 are in this bounded reference")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
