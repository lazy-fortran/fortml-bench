#!/usr/bin/env python3
"""Independent correctness and release timing gate for RBF order-three GPs.

The NumPy path assembles derivative covariance blocks through order six and
checks query JVPs plus likelihood gradients/HVPs by central differences.  The
FortML path runs the behavioral oracle and a release app; CUDA remains a typed
resident-covariance refusal rather than a host fallback.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_queries", "max_observation_order", "kernel", "seconds_per_operation",
    "metric", "value", "max_abs_error", "oracle", "python_version",
    "numpy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
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


def distance_derivative(base: float, difference: float, lengthscale: float,
                        order: int) -> float:
    inv2 = lengthscale ** -2
    inv4 = inv2 * inv2
    inv6 = inv4 * inv2
    inv8 = inv4 * inv4
    inv10 = inv8 * inv2
    inv12 = inv10 * inv2
    inv14 = inv12 * inv2
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
    if order == 6:
        return (difference**6 * inv12 - 15.0 * difference**4 * inv10 +
                45.0 * difference**2 * inv8 - 15.0 * inv6) * base
    if order == 7:
        return (-difference**7 * inv14 + 21.0 * difference**5 * inv12 -
                105.0 * difference**3 * inv10 + 105.0 * difference * inv8) * base
    raise ValueError(order)


def covariance(left: np.ndarray, left_order: np.ndarray, right: np.ndarray,
               right_order: np.ndarray, variance: float, lengthscale: float,
               noise: float = 0.0) -> np.ndarray:
    result = np.empty((left.size, right.size), dtype=np.float64)
    for i, order_left in enumerate(left_order):
        for j, order_right in enumerate(right_order):
            difference = float(left[i] - right[j])
            base = variance * np.exp(-0.5 * difference**2 / lengthscale**2)
            result[i, j] = (-1.0) ** int(order_right) * distance_derivative(
                base, difference, lengthscale, int(order_left + order_right)
            )
    if noise:
        result = result + noise * np.eye(left.size)
    return result


def fit_objective(theta: np.ndarray, x: np.ndarray, orders: np.ndarray,
                  y: np.ndarray) -> float:
    variance, lengthscale, noise = np.exp(theta)
    gram = covariance(x, orders, x, orders, variance, lengthscale, noise)
    sign, logdet = np.linalg.slogdet(gram)
    if sign <= 0:
        raise RuntimeError("oracle covariance is not positive definite")
    alpha = np.linalg.solve(gram, y)
    return float(-0.5 * y @ alpha - 0.5 * logdet - 0.5 * x.size * np.log(2.0 * np.pi))


def oracle() -> dict[str, float]:
    x = np.array([-1.2, -0.45, 0.15, 0.82, 1.35], dtype=np.float64)
    y = np.array([0.73, -0.24, 0.91, -0.38, 0.19], dtype=np.float64)
    orders = np.array([0, 1, 2, 3, 0], dtype=np.int64)
    query = np.array([-0.91, -0.17, 0.57, 1.08], dtype=np.float64)
    query_orders = np.array([0, 1, 2, 3], dtype=np.int64)
    direction = np.array([0.17, -0.11, 0.08, -0.14], dtype=np.float64)
    variance, lengthscale, noise = 1.35, 0.79, 0.041
    jitter = 1.0e-10
    gram = covariance(x, orders, x, orders, variance, lengthscale, noise + jitter)
    alpha = np.linalg.solve(gram, y)
    cross = covariance(x, orders, query, query_orders, variance, lengthscale)
    solved = np.linalg.solve(gram, cross)
    mean = cross.T @ alpha
    prior = covariance(query, query_orders, query, query_orders, variance, lengthscale)
    variance_post = np.diag(prior) - np.sum(cross * solved, axis=0)
    h = 2.0e-5

    def prediction(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        block = covariance(x, orders, points, query_orders, variance, lengthscale)
        solved_block = np.linalg.solve(gram, block)
        prior_block = covariance(points, query_orders, points, query_orders,
                                 variance, lengthscale)
        return block.T @ alpha, np.diag(prior_block) - np.sum(block * solved_block, axis=0)

    mean_plus, variance_plus = prediction(query + h * direction)
    mean_minus, variance_minus = prediction(query - h * direction)
    mean_dot_fd = (mean_plus - mean_minus) / (2.0 * h)
    variance_dot_fd = (variance_plus - variance_minus) / (2.0 * h)
    theta = np.log([variance, lengthscale, noise])
    parameter_direction = np.array([0.07, -0.04, 0.09], dtype=np.float64)
    hp = 5.0e-5
    grad = np.array([(fit_objective(theta + hp * np.eye(3)[i], x, orders, y) -
                      fit_objective(theta - hp * np.eye(3)[i], x, orders, y)) /
                     (2.0 * hp) for i in range(3)])
    hvp = np.array([
        (fit_objective(theta + hp * parameter_direction + hp * np.eye(3)[i], x, orders, y) -
         fit_objective(theta + hp * parameter_direction - hp * np.eye(3)[i], x, orders, y) -
         fit_objective(theta - hp * parameter_direction + hp * np.eye(3)[i], x, orders, y) +
         fit_objective(theta - hp * parameter_direction - hp * np.eye(3)[i], x, orders, y)) /
        (4.0 * hp * hp) for i in range(3)
    ])
    return {
        "jvp_fd_max_abs": float(max(np.max(np.abs(mean_dot_fd)),
                                     np.max(np.abs(variance_dot_fd)))),
        "objective": fit_objective(theta, x, orders, y),
        "gradient_norm": float(np.linalg.norm(grad)),
        "hvp_norm": float(np.linalg.norm(hvp)),
        "minimum_posterior_variance": float(np.min(variance_post)),
        "prediction_checksum": float(np.sum(mean) + np.sum(variance)),
    }


def release_oracle() -> dict[str, float]:
    """Independent NumPy replay of the release app's larger fixed fixture."""
    n = 24
    q = 16
    x = np.linspace(-1.45, 1.45, n, dtype=np.float64)
    y = 0.4 * np.sin(0.7 * x) + 0.08 * x**2
    orders = np.arange(n, dtype=np.int64) % 4
    query = np.linspace(-1.31, 1.31, q, dtype=np.float64)
    direction = 0.04 * np.cos(0.31 * np.arange(1, q + 1, dtype=np.float64))
    query_orders = np.arange(1, q + 1, dtype=np.int64) % 4
    mean_bar = 0.1 - 0.003 * np.arange(1, q + 1, dtype=np.float64)
    variance_bar = -0.06 + 0.002 * np.arange(1, q + 1, dtype=np.float64)
    variance, lengthscale, noise = 1.35, 0.79, 0.041
    jitter = 1.0e-10
    gram = covariance(x, orders, x, orders, variance, lengthscale, noise + jitter)
    alpha = np.linalg.solve(gram, y)

    def prediction(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        block = covariance(x, orders, points, query_orders, variance, lengthscale)
        solved = np.linalg.solve(gram, block)
        prior = covariance(points, query_orders, points, query_orders, variance, lengthscale)
        return block.T @ alpha, np.diag(prior) - np.sum(block * solved, axis=0)

    mean, posterior_variance = prediction(query)
    h = 2.0e-5
    mean_plus, variance_plus = prediction(query + h * direction)
    mean_minus, variance_minus = prediction(query - h * direction)
    mean_dot = (mean_plus - mean_minus) / (2.0 * h)
    variance_dot = (variance_plus - variance_minus) / (2.0 * h)
    mean_dot_sum = float(np.sum(mean_dot) + np.sum(variance_dot))
    query_bar = mean_bar * mean_dot / direction + variance_bar * variance_dot / direction
    # The app's VJP uses the same per-query derivative, including directions
    # that are nonzero for every fixture point.  The quotient above is safe
    # for this fixed cosine direction and avoids a second n-by-n loop here.
    input_vjp_sum = float(np.sum(query_bar))

    def objective(theta: np.ndarray) -> float:
        return fit_objective(theta, x, orders, y)

    theta = np.log([variance, lengthscale, noise])
    parameter_direction = np.array([0.07, -0.04, 0.09], dtype=np.float64)
    # This larger step keeps the nested finite-difference HVP above the
    # cancellation floor of the dense float64 likelihood while remaining in
    # the quadratic convergence regime.
    hp = 5.0e-4
    eye = np.eye(3)
    gradient_plus = np.array([
        (objective(theta + parameter_direction * hp + hp * eye[i]) -
         objective(theta + parameter_direction * hp - hp * eye[i])) / (2.0 * hp)
        for i in range(3)
    ])
    gradient_minus = np.array([
        (objective(theta - parameter_direction * hp + hp * eye[i]) -
         objective(theta - parameter_direction * hp - hp * eye[i])) / (2.0 * hp)
        for i in range(3)
    ])
    hvp_sum = float(np.sum((gradient_plus - gradient_minus) / (2.0 * hp)))
    return {
        "prediction": float(np.sum(mean) + np.sum(posterior_variance)),
        "input_jvp": mean_dot_sum,
        "input_vjp": input_vjp_sum,
        "hyperparameter_hvp": hvp_sum,
    }


def make_row(details: dict[str, object], **values: object) -> dict[str, object]:
    row: dict[str, object] = {field: "" for field in FIELDS}
    row.update(details)
    row.update({
        "workload": "second_derivative_gp_rbf_order3", "device": "cpu",
        "status": "pass", "n_samples": 5, "n_queries": 4,
        "max_observation_order": 3, "kernel": "rbf",
    })
    row.update(values)
    return row


def run_release_app(fortml: Path, target: str, details: dict[str, object],
                    rows: list[dict[str, object]]) -> None:
    environment = os.environ.copy()
    environment.update({"FO_SCAN_FALLBACK": "regex", "OMP_NUM_THREADS": "1"})
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                   env=environment, check=True)
    result = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                            env=environment, check=True, capture_output=True, text=True)
    pattern = re.compile(
        r"^second_derivative_gp_rbf_order3,\s*(prediction|input_jvp|input_vjp|"
        r"hyperparameter_hvp),\s*cpu,\s*([0-9Ee+.-]+),\s*([0-9Ee+.-]+)$"
    )
    expected = release_oracle()
    release_details = dict(details, n_samples=24, n_queries=16)
    found = 0
    for line in result.stdout.splitlines():
        match = pattern.match(line.strip())
        if match is None:
            continue
        found += 1
        phase = match.group(1)
        observed = float(match.group(3))
        error = abs(observed - expected[phase])
        if error > 2.0e-6:
            raise RuntimeError(
                f"release app {phase} checksum differs from independent oracle by {error:g}"
            )
        rows.append(make_row(release_details, backend="fortml_release_app", phase=phase,
                             seconds_per_operation=float(match.group(2)), metric="checksum",
                             value=observed, max_abs_error=error,
                             oracle="release app after independent behavioral oracle",
                             notes="-O3; fixed order-three RBF fixture"))
    if found != 4:
        raise RuntimeError(f"release app emitted {found} order-three timing rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/second_derivative_gp_rbf_order3.csv"))
    parser.add_argument("--target", default="fortml_bench_second_derivative_gp_rbf_order3")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    metrics = oracle()
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml), "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    rows: list[dict[str, object]] = []
    for metric, value in metrics.items():
        rows.append(make_row(details, backend="numpy_oracle", phase="oracle", metric=metric,
                             value=value, max_abs_error=0.0,
                             oracle="independent NumPy order-six covariance/finite differences"))
    if args.skip_fortml:
        rows.append(make_row(details, backend="fortml", phase="behavioral_gate", status="skipped",
                             oracle="test_second_derivative_gp_rbf_order3",
                             notes="--skip-fortml"))
    else:
        environment = os.environ.copy()
        environment["FO_SCAN_FALLBACK"] = "regex"
        started = time.perf_counter()
        subprocess.run(["fo", "test", "test_second_derivative_gp_rbf_order3",
                        "test_second_derivative_gp"], cwd=fortml, env=environment, check=True)
        elapsed = time.perf_counter() - started
        rows.append(make_row(details, backend="fortml", phase="behavioral_gate",
                             seconds_per_operation=elapsed, metric="tests_passed", value=2.0,
                             max_abs_error=0.0,
                             oracle="independent finite-difference/adjoint Fortran oracle",
                             notes="order-three covariance, input JVP/VJP, gradient/HVP, refusals"))
        run_release_app(fortml, args.target, details, rows)
    rows.append(make_row(details, backend="fortml", phase="device_boundary", device="cuda",
                         status="refused", metric="resident_covariance", value="nan",
                         max_abs_error=0.0, oracle="FORTNUM_NOT_IMPLEMENTED",
                         notes="no host fallback for order-three derivative covariance"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
