#!/usr/bin/env python3
"""Correctness gate for weighted Laplace-GP classification fits.

The NumPy oracle implements the binary Laplace Newton recurrence independently
of FortML.  It checks the weighted mode log posterior and its envelope
gradient with respect to log kernel variance/length scale against refitted
central differences.  The Fortran gate additionally checks OVR composition,
logistic/probit fits, malformed weights, and the typed CUDA boundary.
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_parameters", "seconds_per_operation", "metric", "value",
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
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fit_oracle(theta: np.ndarray, x: np.ndarray, signed: np.ndarray,
               weights: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    variance, length = np.exp(theta)
    distance = (x[:, None] - x[None, :]) ** 2
    kernel = variance * np.exp(-0.5 * distance / length**2)
    kernel[np.diag_indices_from(kernel)] += 1.0e-7
    mode = np.zeros(x.size, dtype=np.float64)
    for _ in range(100):
        margin = signed * mode
        probability = 1.0 / (1.0 + np.exp(-margin))
        gradient = 1.0 - probability
        curvature = np.maximum(probability * (1.0 - probability), 1.0e-12)
        active = weights > 0.0
        sqrt_w = np.zeros_like(weights)
        sqrt_w[active] = np.sqrt(np.maximum(weights[active] * curvature[active], 1.0e-12))
        b = weights * curvature * mode + signed * weights * gradient
        system = np.eye(x.size) + sqrt_w[:, None] * kernel * sqrt_w[None, :]
        rhs = np.linalg.solve(system, sqrt_w * (kernel @ b))
        new_mode = kernel @ (b - sqrt_w * rhs)
        scale = max(1.0, np.max(np.abs(mode)))
        if np.max(np.abs(new_mode - mode)) / scale <= 1.0e-9:
            mode = new_mode
            break
        mode = new_mode
    alpha = np.linalg.solve(kernel, mode)
    margin = signed * mode
    log_likelihood = -np.logaddexp(0.0, -margin)
    value = float(-0.5 * mode @ alpha + np.sum(weights * log_likelihood))
    dvariance = kernel.copy()
    dlength = kernel * distance / length**2
    dvariance[np.diag_indices_from(dvariance)] = variance
    dlength[np.diag_indices_from(dlength)] = 0.0
    envelope = np.array([0.5 * alpha @ dvariance @ alpha,
                         0.5 * alpha @ dlength @ alpha])
    return value, envelope, mode


def oracle() -> tuple[float, float, int]:
    x = np.array([-1.5, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 1.5], dtype=np.float64)
    signed = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    weights = np.array([0.5, 0.0, 1.5, 2.0, 0.7, 1.2, 2.3, 0.8], dtype=np.float64)
    theta = np.log(np.array([1.4, 0.7], dtype=np.float64))
    value, envelope, _ = fit_oracle(theta, x, signed, weights)
    step = 1.0e-5
    finite_difference = np.empty(2)
    for index in range(2):
        plus = theta.copy()
        minus = theta.copy()
        plus[index] += step
        minus[index] -= step
        finite_difference[index] = (
            fit_oracle(plus, x, signed, weights)[0] -
            fit_oracle(minus, x, signed, weights)[0]
        ) / (2.0 * step)
    error = float(np.max(np.abs(envelope - finite_difference)))
    if error > 3.0e-5 or not np.isfinite(value):
        raise RuntimeError(f"weighted Laplace-GP oracle failed: {error:.3e}")
    return value, error, theta.size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gp_classification_sample_weights.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml, output = args.fortml.resolve(), args.output.resolve()
    value, oracle_error, n_parameters = oracle()
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": "gfortran", "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(details)
        row.update({"workload": "gp_classification_sample_weights",
                    "backend": "fortml", "device": "cpu", "n_samples": 8,
                    "n_features": 1, "n_parameters": n_parameters})
        row.update(values)
        rows.append(row)

    add(phase="independent_oracle", status="pass", metric="weighted_log_posterior",
        value=value, max_abs_error=oracle_error,
        oracle="independent NumPy weighted Laplace Newton recurrence and envelope FD",
        notes="logistic binary objective with zero-weight row")
    started = time.perf_counter()
    if args.skip_fortml:
        status, notes = "skipped", "--skip-fortml"
    else:
        subprocess.run(["fo", "test", "test_gp_classification_sample_weights"],
                       cwd=fortml, check=True)
        status, notes = "pass", "weighted binary/OVR logistic-probit and refusal gate"
    elapsed = time.perf_counter() - started
    add(phase="public_contract_gate", status=status, seconds_per_operation=elapsed,
        metric="envelope_gradient_max_abs_error", value=oracle_error,
        max_abs_error=oracle_error,
        oracle="FortML test_gp_classification_sample_weights", notes=notes)
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_weighted_laplace_graph", value="nan", max_abs_error="",
        oracle="typed FortML CUDA refusal",
        notes="weighted covariance/Laplace state is not resident; no host fallback")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
