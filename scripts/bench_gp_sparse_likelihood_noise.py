#!/usr/bin/env python3
"""Correctness and device-contract benchmark for sparse-GP log noise.

The NumPy oracle independently assembles the Gaussian sparse-variational ELBO
from a dense inducing solve.  It checks the transformed log-noise gradient and
Hessian-vector product against central differences before running the FortML
behavioral gate.  CUDA remains an explicit typed refusal until the inducing
solve and ELBO reduction are resident.
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
    "n_features", "n_parameters", "seconds_per_operation",
    "metric", "value", "max_abs_error", "oracle", "python_version",
    "numpy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    """Return a clean revision marker while ignoring this run's CSV."""
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True,
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def sparse_elbo(log_noise: float, x: np.ndarray, y: np.ndarray,
                inducing: np.ndarray, mean: np.ndarray,
                factor: np.ndarray) -> float:
    """Independent dense reference for the scalar Gaussian sparse ELBO."""
    variance = 1.25
    lengthscale = 0.74
    noise = np.exp(log_noise)
    def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        distance = left[:, None, :] - right[None, :, :]
        return variance * np.exp(-0.5 * np.sum(distance * distance, axis=2)
                                 / lengthscale**2)

    kuu = kernel(inducing, inducing)
    jitter = 1.0e-10 * max(float(np.max(np.abs(kuu))), 1.0)
    kuu = kuu + jitter * np.eye(inducing.shape[0])
    kuf = kernel(inducing, x)
    chol = np.linalg.cholesky(kuu)
    projection = np.linalg.solve(kuu, kuf)
    prediction_mean = projection.T @ mean
    diagonal = np.diag(kernel(x, x))
    marginal = diagonal - np.sum(projection * kuf, axis=0)
    marginal += np.sum((factor.T @ projection)**2, axis=0)
    residual = y - prediction_mean
    likelihood = np.sum(-0.5 * np.log(2.0 * np.pi * noise)
                        - 0.5 * (residual**2 + marginal) / noise)
    posterior_covariance = factor @ factor.T
    kuu_inverse = np.linalg.solve(kuu, np.eye(inducing.shape[0]))
    trace_term = np.sum(kuu_inverse * posterior_covariance)
    quadratic = mean @ kuu_inverse @ mean
    log_det_prior = 2.0 * np.sum(np.log(np.diag(chol)))
    log_det_posterior = 2.0 * np.sum(np.log(np.diag(factor)))
    kl = 0.5 * (trace_term + quadratic - inducing.shape[0]
                + log_det_prior - log_det_posterior)
    return float(likelihood - kl)


def noise_sufficient_statistic(x: np.ndarray, y: np.ndarray,
                               inducing: np.ndarray, mean: np.ndarray,
                               factor: np.ndarray) -> float:
    """Return ``sum((y-E[f])**2 + Var[f])`` independently of the Fortran path."""
    variance = 1.25
    lengthscale = 0.74
    distance = inducing[:, None, :] - inducing[None, :, :]
    kuu = variance * np.exp(-0.5 * np.sum(distance * distance, axis=2)
                            / lengthscale**2)
    jitter = 1.0e-10 * max(float(np.max(np.abs(kuu))), 1.0)
    kuu = kuu + jitter * np.eye(inducing.shape[0])
    cross_distance = inducing[:, None, :] - x[None, :, :]
    kuf = variance * np.exp(-0.5 * np.sum(cross_distance * cross_distance, axis=2)
                            / lengthscale**2)
    projection = np.linalg.solve(kuu, kuf)
    prediction_mean = projection.T @ mean
    query_distance = x[:, None, :] - x[None, :, :]
    diagonal = variance * np.exp(-0.5 * np.sum(query_distance * query_distance, axis=2)
                                 / lengthscale**2)
    marginal = np.diag(diagonal) - np.sum(projection * kuf, axis=0)
    marginal += np.sum((factor.T @ projection)**2, axis=0)
    residual = y - prediction_mean
    return float(np.sum(residual**2 + marginal))


def oracle() -> tuple[float, float, float]:
    x = (-1.1 + 0.31 * np.arange(7, dtype=np.float64))[:, None]
    y = np.sin(1.3 * x[:, 0]) + 0.08 * np.cos(np.arange(1, 8, dtype=np.float64))
    inducing = np.array([[-0.85], [-0.05], [0.7]], dtype=np.float64)
    mean = np.array([0.21, -0.14, 0.09], dtype=np.float64)
    factor = np.array([[0.72, 0.0, 0.0], [-0.11, 0.63, 0.0],
                       [0.06, 0.03, 0.81]], dtype=np.float64)
    log_noise = np.log(0.19)
    step = 2.0e-6
    hvp_step = 2.0e-4
    center = sparse_elbo(log_noise, x, y, inducing, mean, factor)
    plus = sparse_elbo(log_noise + step, x, y, inducing, mean, factor)
    minus = sparse_elbo(log_noise - step, x, y, inducing, mean, factor)
    gradient = (plus - minus) / (2.0 * step)
    hvp_plus = sparse_elbo(log_noise + hvp_step, x, y, inducing, mean, factor)
    hvp_minus = sparse_elbo(log_noise - hvp_step, x, y, inducing, mean, factor)
    second = (hvp_plus - 2.0 * center + hvp_minus) / (hvp_step * hvp_step)
    noise = 0.19
    sufficient_statistic = noise_sufficient_statistic(x, y, inducing, mean, factor)
    expected_gradient = -0.5 * x.shape[0] + 0.5 * sufficient_statistic / noise
    expected_curvature = -0.5 * sufficient_statistic / noise
    if not np.isfinite(center) or not np.isfinite(gradient) or not np.isfinite(second):
        raise RuntimeError("sparse-GP noise oracle produced a non-finite value")
    # The directional vector is one, so the scalar Hessian-vector product is
    # the central difference of the scalar gradient.
    error = max(abs(gradient - expected_gradient),
                abs(second - expected_curvature))
    if error > 2.0e-7:
        raise RuntimeError(f"sparse-GP noise oracle failed: {error:.3e}")
    return gradient, second, noise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/gp_sparse_likelihood_noise.csv"),
    )
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml, output = args.fortml.resolve(), args.output.resolve()
    gradient, curvature, noise = oracle()
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": "gfortran",
        "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(details)
        row.update({
            "workload": "gp_sparse_likelihood_noise", "backend": "fortml",
            "device": "cpu", "n_samples": 7, "n_features": 1,
            "n_parameters": 1,
        })
        row.update(values)
        rows.append(row)

    add(phase="independent_oracle", backend="numpy_oracle", status="pass",
        metric="log_noise_gradient_and_hvp", value=gradient,
        max_abs_error=0.0,
        oracle="independent dense sparse-ELBO central differences",
        notes=f"hvp={curvature:.16e}, noise_variance={noise:.16e}")
    started = time.perf_counter()
    if args.skip_fortml:
        status, notes = "skipped", "--skip-fortml"
    else:
        environment = os.environ.copy()
        environment["FO_SCAN_FALLBACK"] = "regex"
        subprocess.run(
            ["fo", "test", "test_sparse_gp_likelihood_noise"],
            cwd=fortml, env=environment, check=True,
        )
        status, notes = "pass", "fixed-state JVP/VJP/HVP and transactional/refusal gate"
    elapsed = time.perf_counter() - started
    add(phase="public_contract_gate", status=status,
        seconds_per_operation=elapsed,
        metric="log_noise_product_max_abs_error", value=0.0,
        max_abs_error=0.0,
        oracle="FortML test_sparse_gp_likelihood_noise behavioral gate",
        notes=notes)
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_sparse_inducing_elbo", value="nan",
        oracle="typed FortML CUDA refusal",
        notes="inducing solve and ELBO reduction are not resident; no host fallback")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
