#!/usr/bin/env python3
"""Correctness-gated Matérn-5/2 order-two GP hyperproduct lane.

The NumPy path independently assembles value/first/second-derivative blocks
and central-differences the dense objective and query functional.  FortML is
timed only after every CPU checksum agrees; CUDA remains an explicit typed
resident-covariance refusal.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
from pathlib import Path

import numpy as np


N, Q, REPETITIONS = 24, 16, 8
NOISE, JITTER = 0.041, 1.0e-10
VARIANCE, LENGTHSCALE = 1.35, 0.79
THETA = np.log([VARIANCE, LENGTHSCALE, NOISE])
PARAMETER_DIRECTION = np.array([0.07, -0.04, 0.09], dtype=np.float64)
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
    ignored_names = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            pass
    dirty = [
        line for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True,
        ).splitlines()
        if line[3:].strip() not in ignored_names
    ]
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-1.45, 1.45, N, dtype=np.float64)
    y = 0.4*np.sin(0.7*x) + 0.08*x**2
    orders = np.arange(N, dtype=np.int64) % 3
    query = np.linspace(-1.31, 1.31, Q, dtype=np.float64)
    direction = 0.04*np.cos(0.31*np.arange(1, Q + 1, dtype=np.float64))
    query_orders = np.arange(1, Q + 1, dtype=np.int64) % 3
    return x, y, orders, query, direction, query_orders


def matern52_derivative(variance: float, lengthscale: float, x1: float,
                        order1: int, x2: float, order2: int) -> float:
    """Independent d^(order1+order2)/dx1^... covariance block."""
    tau = x1 - x2
    radius = abs(tau)/lengthscale
    root_five = np.sqrt(5.0)
    base = variance*np.exp(-root_five*radius)
    radial = (
        1.0 + root_five*radius + 5.0*radius**2/3.0,
        -(5.0/3.0)*radius*(1.0 + root_five*radius),
        (5.0/3.0)*(5.0*radius**2 - root_five*radius - 1.0),
        (25.0/3.0)*radius*(3.0 - root_five*radius),
        (25.0/3.0)*(3.0 - 5.0*root_five*radius + 5.0*radius**2),
        root_five**5/3.0*(-8.0 + 7.0*root_five*radius - 5.0*radius**2),
    )
    total = order1 + order2
    if total >= len(radial):
        raise ValueError(f"unsupported derivative order {total}")
    sign_tau = -1.0 if tau < 0.0 else 1.0
    sign = sign_tau if total % 2 else 1.0
    value = base*radial[total]/lengthscale**total*sign
    if order2 % 2:
        value = -value
    return float(value)


def covariance(theta: np.ndarray, left: np.ndarray, left_orders: np.ndarray,
               right: np.ndarray, right_orders: np.ndarray, noise: bool = False) -> np.ndarray:
    variance, lengthscale, noise_variance = np.exp(theta)
    result = np.empty((left.size, right.size), dtype=np.float64)
    for i, order_left in enumerate(left_orders):
        for j, order_right in enumerate(right_orders):
            result[i, j] = matern52_derivative(
                variance, lengthscale, float(left[i]), int(order_left),
                float(right[j]), int(order_right),
            )
    if noise:
        result[np.diag_indices(left.size)] += noise_variance + JITTER
    return result


def predict(theta: np.ndarray, x: np.ndarray, y: np.ndarray, orders: np.ndarray,
            query: np.ndarray, query_orders: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gram = covariance(theta, x, orders, x, orders, noise=True)
    cross = covariance(theta, x, orders, query, query_orders)
    alpha = np.linalg.solve(gram, y)
    solved = np.linalg.solve(gram, cross)
    prior = covariance(theta, query, query_orders, query, query_orders)
    return cross.T @ alpha, np.diag(prior) - np.sum(cross*solved, axis=0)


def likelihood(theta: np.ndarray, x: np.ndarray, y: np.ndarray,
               orders: np.ndarray) -> float:
    gram = covariance(theta, x, orders, x, orders, noise=True)
    chol = np.linalg.cholesky(gram)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
    return float(-0.5*np.dot(y, alpha) - np.sum(np.log(np.diag(chol))) -
                 0.5*x.size*np.log(2.0*np.pi))


def objective_gradient(theta: np.ndarray, x: np.ndarray, y: np.ndarray,
                       orders: np.ndarray) -> np.ndarray:
    step = 3.0e-6
    result = np.empty(3, dtype=np.float64)
    for index in range(3):
        probe = np.zeros(3, dtype=np.float64)
        probe[index] = step
        result[index] = (likelihood(theta + probe, x, y, orders) -
                         likelihood(theta - probe, x, y, orders))/(2.0*step)
    return result


def oracle() -> dict[str, float]:
    x, y, orders, query, direction, query_orders = fixture()
    mean, variance = predict(THETA, x, y, orders, query, query_orders)
    h = 2.0e-5
    mean_plus, variance_plus = predict(THETA, x, y, orders, query + h*direction, query_orders)
    mean_minus, variance_minus = predict(THETA, x, y, orders, query - h*direction, query_orders)
    mean_dot = (mean_plus - mean_minus)/(2.0*h)
    variance_dot = (variance_plus - variance_minus)/(2.0*h)

    mean_bar = 0.1 - 0.003*np.arange(1, Q + 1, dtype=np.float64)
    variance_bar = -0.06 + 0.002*np.arange(1, Q + 1, dtype=np.float64)
    query_bar = np.empty(Q, dtype=np.float64)
    for index in range(Q):
        perturbation = np.zeros(Q, dtype=np.float64)
        perturbation[index] = h
        mean_p, variance_p = predict(THETA, x, y, orders, query + perturbation, query_orders)
        mean_m, variance_m = predict(THETA, x, y, orders, query - perturbation, query_orders)
        query_bar[index] = (np.dot(mean_bar, mean_p - mean_m) +
                            np.dot(variance_bar, variance_p - variance_m))/(2.0*h)

    gradient = objective_gradient(THETA, x, y, orders)
    hp = 5.0e-4
    gradient_plus = objective_gradient(THETA + hp*PARAMETER_DIRECTION, x, y, orders)
    gradient_minus = objective_gradient(THETA - hp*PARAMETER_DIRECTION, x, y, orders)
    hvp = (gradient_plus - gradient_minus)/(2.0*hp)
    return {
        "prediction": float(np.sum(mean) + np.sum(variance)),
        "input_jvp": float(np.sum(mean_dot) + np.sum(variance_dot)),
        "input_vjp": float(np.sum(query_bar)),
        "hyperparameter_gradient": float(np.sum(gradient)),
        "hyperparameter_hvp": float(np.sum(hvp)),
    }


def make_row(metadata: dict[str, str], **values: object) -> dict[str, object]:
    row: dict[str, object] = {field: "" for field in FIELDS}
    row.update({
        "workload": "second_derivative_gp_matern52_hyperparameters",
        "device": "cpu", "status": "pass", "n_samples": N, "n_queries": Q,
        "max_observation_order": 2, "kernel": "matern52",
    })
    row.update(metadata)
    row.update(values)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/second_derivative_gp_matern52_hyperparameters.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/SECOND_DERIVATIVE_GP_MATERN52_HYPERPARAMETERS.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ignored = (args.output.resolve(), args.report.resolve())
    metadata = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(args.fortml), "benchmark_revision": revision(root, ignored),
        "compiler": "gfortran", "flags": "-O3",
    }
    expected = oracle()
    oracle_description = "independent NumPy dense Matérn-5/2 covariance and central-difference oracle"
    rows: list[dict[str, object]] = []
    for phase, value in expected.items():
        rows.append(make_row(metadata, backend="numpy_oracle", phase=phase,
                             metric="checksum", value=value, max_abs_error=0.0,
                             oracle=oracle_description,
                             notes="central differences only in the independent behavioral oracle"))

    environment = os.environ.copy()
    environment.update({"FO_SCAN_FALLBACK": "regex", "OMP_NUM_THREADS": "1"})
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=args.fortml,
                   env=environment, check=True, capture_output=True, text=True)
    result = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_second_derivative_gp_matern52_hyperparameters"],
        cwd=args.fortml, env=environment, check=True, capture_output=True, text=True,
    )
    pattern = re.compile(
        r"^second_derivative_gp_matern52_hyperparameters,\s*(prediction|input_jvp|input_vjp|"
        r"hyperparameter_gradient|hyperparameter_hvp),\s*cpu,\s*([0-9Ee+.-]+),\s*([0-9Ee+.-]+)$"
    )
    refusal_pattern = re.compile(
        r"^second_derivative_gp_matern52_hyperparameters,\s*device,\s*cuda,\s*refused,\s*(\d+)$"
    )
    records: dict[str, tuple[float, float]] = {}
    refusal_codes: list[int] = []
    for line in result.stdout.splitlines():
        match = pattern.match(line.strip())
        refusal = refusal_pattern.match(line.strip())
        if refusal is not None:
            refusal_codes.append(int(refusal.group(1)))
        elif match is not None:
            records[match.group(1)] = (float(match.group(2)), float(match.group(3)))
    if set(records) != set(expected):
        raise RuntimeError(f"release app emitted phases {sorted(records)}, expected {sorted(expected)}")
    if refusal_codes != [3]:
        raise RuntimeError(f"release app emitted unexpected CUDA refusal codes: {refusal_codes}")
    errors: dict[str, float] = {}
    for phase, reference in expected.items():
        seconds, observed = records[phase]
        error = abs(observed - reference)
        errors[phase] = error
        if error > 3.0e-5*max(1.0, abs(reference)):
            raise RuntimeError(f"FortML {phase} checksum mismatch: {error:.3e}")
        rows.append(make_row(metadata, backend="fortml_release_app", phase=phase,
                             metric="checksum", value=observed, max_abs_error=error,
                             seconds_per_operation=seconds,
                             oracle="release app after independent NumPy behavioral oracle",
                             notes="CPU reference; exact Matérn-5/2 order-four parameter recurrence"))
    rows.append(make_row(metadata, backend="fortml", phase="device", device="cuda",
                         status="unavailable", metric="status", value="FORTNUM_NOT_IMPLEMENTED",
                         oracle="typed_device_contract",
                         notes="selected CUDA is refused before output writes"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# Matérn-5/2 second-derivative GP hyperproducts\n\n"
        f"FortML revision: `{metadata['fortml_revision']}`  \n"
        f"Benchmark revision: `{metadata['benchmark_revision']}`  \n\n"
        "The independent NumPy fixture assembles value, first-derivative, and "
        "second-derivative covariance blocks through order four. It central "
        "differences the dense prediction functional, likelihood gradient, and "
        "query functional, then compares every checksum with the CPU release "
        "app. The largest CPU checksum error is "
        f"{max(errors.values()):.3e}. Timings are recorded in the CSV.\n\n"
        "The production Matérn-5/2 parameter jet uses the exact scaling identity "
        "for log lengthscale and a finite coincidence limit. No finite-difference "
        "fallback is used in the Fortran path. CUDA is recorded as the typed "
        "`FORTNUM_NOT_IMPLEMENTED` refusal until resident derivative covariance "
        "and factorization kernels are linked.\n",
    )
    print(f"wrote {args.output} and {args.report} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
