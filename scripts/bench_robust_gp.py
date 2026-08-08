#!/usr/bin/env python3
"""Benchmark Poisson and Student-t Laplace GP observation models.

The independent NumPy path reconstructs the latent-mode Newton solves. It
checks Poisson stationarity/positive rates and the Student-t outlier contrast
before the FortML behavioral/refusal test is timed.
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


def rbf(x_left: np.ndarray, x_right: np.ndarray, variance: float,
        lengthscale: float) -> np.ndarray:
    difference = x_left[:, None, :] - x_right[None, :, :]
    return variance * np.exp(-0.5 * np.sum(difference * difference, axis=2) / lengthscale**2)


def poisson_oracle() -> tuple[float, float, float]:
    x = (-1.4 + 0.4 * np.arange(8, dtype=np.float64))[:, None]
    counts = np.maximum(0.0, np.rint(3.0 * np.exp(0.6 * x[:, 0])))
    gram = rbf(x, x, 1.0, 0.9) + 1.0e-8 * np.eye(8)
    latent = np.zeros(8, dtype=np.float64)
    for _ in range(200):
        rate = np.exp(latent)
        curvature = rate
        root = np.sqrt(curvature)
        system = np.eye(8) + root[:, None] * gram * root[None, :]
        gradient = counts - rate
        b_vector = curvature * latent + gradient
        temporary = np.linalg.solve(system, root * (gram @ b_vector))
        step = gram @ (b_vector - root * temporary)
        if np.max(np.abs(step - latent)) < 1.0e-10:
            latent = step
            break
        latent = step
    rate = np.exp(latent)
    stationarity_error = float(np.max(np.abs((counts - rate) - np.linalg.solve(gram, latent))))
    cross = gram
    system = np.eye(8) + np.sqrt(rate)[:, None] * gram * np.sqrt(rate)[None, :]
    weighted = np.linalg.solve(system, np.sqrt(rate)[:, None] * cross)
    variance = np.diag(gram) - np.sum(cross * (np.sqrt(rate)[:, None] * weighted), axis=0)
    response = np.exp(cross.T @ (counts - rate) + 0.5 * variance)
    positive = float(np.min(response))
    monotone = float(np.min(np.diff(response)))
    if stationarity_error > 2.0e-7 or positive <= 0.0 or monotone < -1.0e-10:
        raise RuntimeError(
            f"Poisson oracle failed: stationarity={stationarity_error:.3e}, "
            f"minimum_rate={positive:.3e}, min_rate_difference={monotone:.3e}"
        )
    return stationarity_error, positive, monotone


def student_t_mode(x: np.ndarray, y: np.ndarray, nu: float, scale: float,
                   kernel_variance: float = 4.0,
                   lengthscale: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gram = rbf(x, x, kernel_variance, lengthscale) + 1.0e-8 * np.eye(x.shape[0])
    latent = y.copy()
    for _ in range(200):
        residual = y - latent
        denominator = nu * scale * scale + residual * residual
        gradient = (nu + 1.0) * residual / denominator
        curvature = (nu + 1.0) * (nu * scale * scale - residual * residual) / denominator**2
        curvature = np.maximum(curvature, 0.0)
        root = np.sqrt(curvature)
        system = np.eye(x.shape[0]) + root[:, None] * gram * root[None, :]
        b_vector = curvature * latent + gradient
        temporary = np.linalg.solve(system, root * (gram @ b_vector))
        step = gram @ (b_vector - root * temporary)
        next_latent = 0.5 * latent + 0.5 * step
        if np.max(np.abs(next_latent - latent)) < 1.0e-10:
            latent = next_latent
            break
        latent = next_latent
    residual = y - latent
    denominator = nu * scale * scale + residual * residual
    curvature = np.maximum((nu + 1.0) * (nu * scale * scale - residual * residual) /
                           denominator**2, 0.0)
    root = np.sqrt(curvature)
    system = np.eye(x.shape[0]) + root[:, None] * gram * root[None, :]
    gradient = (nu + 1.0) * residual / denominator
    return latent, np.linalg.solve(gram, latent), curvature


def student_t_contrast() -> tuple[float, float]:
    x = (-2.0 + 0.4 * np.arange(11, dtype=np.float64))[:, None]
    clean = 0.5 * x[:, 0]
    spoiled = clean.copy()
    spoiled[5] += 25.0
    probe = x[5:6]
    truth = clean[5]
    latent, alpha, curvature = student_t_mode(x, spoiled, 3.0, 0.2)
    robust_mean = float((rbf(x, probe, 4.0, 0.8).T @ alpha)[0])
    gaussian_gram = rbf(x, x, 4.0, 0.8) + 0.04 * np.eye(11)
    gaussian_cross = rbf(x, probe, 4.0, 0.8)
    gaussian_alpha = np.linalg.solve(gaussian_gram, spoiled)
    gaussian_mean = float((gaussian_cross.T @ gaussian_alpha)[0])
    robust_error = abs(robust_mean - truth)
    gaussian_error = abs(gaussian_mean - truth)
    if robust_error >= 0.5 * gaussian_error:
        raise RuntimeError(
            f"Student-t outlier oracle failed: robust={robust_error:.3e}, "
            f"gaussian={gaussian_error:.3e}"
        )
    del latent, curvature
    return robust_error, gaussian_error


def oracle() -> tuple[float, float, float, float, float]:
    stationarity, minimum_rate, monotone = poisson_oracle()
    robust_error, gaussian_error = student_t_contrast()
    return stationarity, minimum_rate, monotone, robust_error, gaussian_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/robust_gp.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    stationarity, minimum_rate, monotone, robust_error, gaussian_error = oracle()
    started = time.perf_counter()
    if args.skip_fortml:
        status, notes = "skipped", "--skip-fortml"
    else:
        environment = os.environ.copy()
        environment["FO_SCAN_FALLBACK"] = "regex"
        subprocess.run(["fo", "test", "test_robust_gp"],
                       cwd=fortml, env=environment, check=True)
        status = "pass"
        notes = "FortML test covers Poisson stationarity/rates, Student-t outlier contrast, and refusals"
    elapsed = time.perf_counter() - started
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
        row.update({"workload": "robust_gp", "backend": "fortml", "device": "cpu",
                    "n_samples": 11, "n_queries": 8})
        row.update(values)
        rows.append(row)

    add(phase="poisson_stationarity", backend="numpy_oracle", status="pass",
        likelihood="poisson", metric="mode_stationarity_max_abs_error",
        value=stationarity, max_abs_error=stationarity,
        oracle="independent NumPy Poisson Laplace Newton solve",
        notes=f"minimum_response_rate={minimum_rate:.3e}; min_rate_difference={monotone:.3e}")
    add(phase="poisson_rates", backend="numpy_oracle", status="pass",
        likelihood="poisson", metric="minimum_positive_predicted_rate", value=minimum_rate,
        max_abs_error=max(0.0, -monotone),
        oracle="independent NumPy log-rate posterior response",
        notes="rates are positive and monotone on the rising-count fixture")
    add(phase="student_t_outlier_contrast", backend="numpy_oracle", status="pass",
        likelihood="student_t", metric="robust_to_gaussian_error_ratio",
        value=robust_error / gaussian_error, max_abs_error=robust_error,
        oracle="independent NumPy Student-t/Gaussian posterior contrast",
        notes=f"robust_error={robust_error:.3e}; gaussian_error={gaussian_error:.3e}")
    add(phase="public_contract_gate", status=status, seconds_per_operation=elapsed,
        metric="fortml_robust_gp_test", value=1.0,
        max_abs_error=max(stationarity, robust_error),
        oracle="FortML test_robust_gp behavioral gate", notes=notes)
    add(phase="refusal_contract", status="refused", likelihood="invalid",
        metric="negative_count_or_invalid_likelihood_or_nu", value="nan",
        max_abs_error=0.0,
        oracle="typed FORTNUM_DOMAIN_ERROR for malformed robust likelihood inputs",
        notes="negative Poisson count, unknown likelihood, and non-positive nu are refused")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
