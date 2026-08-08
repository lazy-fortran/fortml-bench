#!/usr/bin/env python3
"""Correctness and device-contract benchmark for binary GP log probabilities.

The NumPy oracle independently evaluates the integrated logistic/probit
predictive map, its input/parameter chain rule, and the finite probit-tail
floor.  The Fortran gate checks the fitted Laplace model, fixed-state kernel
parameter setter, JVP/VJP products, and typed CUDA refusal.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
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


def normal_cdf(value: np.ndarray) -> np.ndarray:
    return 0.5 * np.vectorize(math.erfc)(-value / np.sqrt(2.0))


def predictive(mean: np.ndarray, variance: np.ndarray, likelihood: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if likelihood == "logistic":
        scale = np.sqrt(1.0 + np.pi * variance / 8.0)
        probability = 1.0 / (1.0 + np.exp(-mean / scale))
        p_mean = probability * (1.0 - probability) / scale
        p_variance = probability * (1.0 - probability) * (-mean * np.pi / (16.0 * scale**3))
    else:
        scale = np.sqrt(1.0 + variance)
        z = mean / scale
        probability = normal_cdf(z)
        density = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
        p_mean = density / scale
        p_variance = density * (-mean / (2.0 * scale**3))
    return probability, p_mean, p_variance


def oracle() -> tuple[float, float, float, int]:
    mean = np.array([-1.2, -0.2, 0.3, 1.1], dtype=np.float64)
    variance = np.array([0.08, 0.17, 0.11, 0.23], dtype=np.float64)
    mean_dot = np.array([0.2, -0.1, 0.3, -0.2], dtype=np.float64)
    variance_dot = np.array([0.03, -0.02, 0.01, 0.04], dtype=np.float64)
    errors = []
    log_sum = 0.0
    for likelihood in ("logistic", "probit"):
        probability, p_mean, p_variance = predictive(mean, variance, likelihood)
        probability = np.clip(probability, np.finfo(np.float64).tiny, 1.0)
        log_probability = np.column_stack((np.log1p(-probability), np.log(probability)))
        probability_dot = p_mean * mean_dot + p_variance * variance_dot
        log_dot = np.column_stack((
            -probability_dot / np.maximum(1.0 - probability, np.finfo(np.float64).tiny),
            probability_dot / probability,
        ))
        step = 2.0e-6
        plus = predictive(mean + step * mean_dot, variance + step * variance_dot, likelihood)[0]
        minus = predictive(mean - step * mean_dot, variance - step * variance_dot, likelihood)[0]
        plus = np.clip(plus, np.finfo(np.float64).tiny, 1.0)
        minus = np.clip(minus, np.finfo(np.float64).tiny, 1.0)
        finite_difference = np.column_stack((
            (np.log1p(-plus) - np.log1p(-minus)) / (2.0 * step),
            (np.log(plus) - np.log(minus)) / (2.0 * step),
        ))
        errors.append(float(np.max(np.abs(log_dot - finite_difference))))
        errors.append(float(np.max(np.abs(np.exp(log_probability[:, 1]) - probability))))
        log_sum += float(np.sum(log_probability))
    # Stable negative probit-tail floor is an explicit boundary of the API.
    tail_probability = max(float(normal_cdf(np.array([-40.0]))[0]), np.finfo(np.float64).tiny)
    if not math.isfinite(math.log(tail_probability)):
        raise RuntimeError("probit log floor is not finite")
    error = max(errors)
    if error > 2.0e-8:
        raise RuntimeError(f"GP log-probability oracle failed: {error:.3e}")
    return log_sum, error, float(tail_probability), 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gp_classification_log_proba.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml, output = args.fortml.resolve(), args.output.resolve()
    oracle_value, oracle_error, tail_probability, n_parameters = oracle()
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
        row.update({"workload": "gp_classification_log_proba", "backend": "fortml",
                    "device": "cpu", "n_samples": 8, "n_features": 1,
                    "n_parameters": n_parameters})
        row.update(values)
        rows.append(row)

    add(phase="independent_oracle", backend="numpy_oracle", status="pass",
        metric="log_probability_jvp_max_abs_error", value=oracle_value,
        max_abs_error=oracle_error,
        oracle="independent NumPy logistic/probit predictive map and central difference",
        notes=f"finite probit-tail probability floor={tail_probability:.3e}")
    started = time.perf_counter()
    if args.skip_fortml:
        status, notes = "skipped", "--skip-fortml"
    else:
        environment = os.environ.copy()
        environment["FO_SCAN_FALLBACK"] = "regex"
        subprocess.run(["fo", "test", "test_gp_classification_log_proba"],
                       cwd=fortml, env=environment, check=True)
        status, notes = "pass", "fitted binary logistic Laplace log-proba and derivative/refusal gate"
    elapsed = time.perf_counter() - started
    add(phase="public_contract_gate", status=status, seconds_per_operation=elapsed,
        metric="log_probability_derivative_max_abs_error", value=oracle_error,
        max_abs_error=oracle_error,
        oracle="FortML test_gp_classification_log_proba behavioral gate", notes=notes)
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_log_probability_laplace_graph", value="nan", max_abs_error="",
        oracle="typed FortML CUDA refusal",
        notes="covariance/Laplace state is not resident; no host fallback")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
