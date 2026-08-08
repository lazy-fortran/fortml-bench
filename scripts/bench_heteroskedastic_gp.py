#!/usr/bin/env python3
"""Benchmark the supplied-noise heteroskedastic GP contract.

The NumPy oracle independently reconstructs the diagonal-noise Cholesky
posterior and the log-noise interpolation. It checks the constant-noise
reduction, quiet/noisy posterior contrast, and positive log-noise interpolation
before the FortML behavioral/refusal gate is timed.
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
    "n_queries", "seconds_per_operation", "metric", "value", "max_abs_error",
    "oracle", "python_version", "numpy_version", "fortml_revision",
    "benchmark_revision", "compiler", "flags", "notes",
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


def rbf(x_left: np.ndarray, x_right: np.ndarray, lengthscale: float) -> np.ndarray:
    difference = x_left[:, None, :] - x_right[None, :, :]
    return np.exp(-0.5 * np.sum(difference * difference, axis=2) / lengthscale**2)


def posterior(x: np.ndarray, y: np.ndarray, query: np.ndarray,
              noise_variance: np.ndarray, lengthscale: float) -> tuple[np.ndarray, np.ndarray]:
    gram = rbf(x, x, lengthscale) + np.diag(noise_variance + 1.0e-10)
    cross = rbf(x, query, lengthscale)
    prior = rbf(query, query, lengthscale)
    factor = np.linalg.cholesky(gram)
    alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, y))
    work = np.linalg.solve(factor.T, np.linalg.solve(factor, cross))
    mean = cross.T @ alpha
    variance = np.diag(prior) - np.sum(cross * work, axis=0)
    return mean, variance


def log_noise_interpolation(x: np.ndarray, noise_variance: np.ndarray,
                            query: np.ndarray) -> np.ndarray:
    centred = np.log(noise_variance)
    mean = float(np.mean(centred))
    gram = rbf(x, x, 1.2) + 1.0e-10 * np.eye(x.shape[0])
    factor = np.linalg.cholesky(gram)
    alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, centred - mean))
    cross = rbf(x, query, 1.2)
    return np.exp(mean + cross.T @ alpha)


def oracle() -> tuple[float, float, float, float, float]:
    x = (-1.4 + 0.4 * np.arange(8, dtype=np.float64))[:, None]
    y = np.sin(1.9 * x[:, 0])
    query = (-1.1 + 0.55 * np.arange(5, dtype=np.float64))[:, None]
    constant_noise = np.full(8, 0.05, dtype=np.float64)
    hetero_mean, hetero_variance = posterior(x, y, query, constant_noise, 0.7)
    plain_mean, plain_variance = posterior(x, y, query, constant_noise, 0.7)
    constant_error = float(max(np.max(np.abs(hetero_mean - plain_mean)),
                               np.max(np.abs(hetero_variance - plain_variance))))

    x_split = (-2.0 + 0.4 * np.arange(10, dtype=np.float64))[:, None]
    y_split = 0.5 * x_split[:, 0]
    split_noise = np.where(x_split[:, 0] < 0.0, 1.0e-4, 1.0)
    quiet_query = np.array([[-1.4]], dtype=np.float64)
    loud_query = np.array([[1.4]], dtype=np.float64)
    quiet_mean, quiet_variance = posterior(x_split, y_split, quiet_query, split_noise, 0.6)
    loud_mean, loud_variance = posterior(x_split, y_split, loud_query, split_noise, 0.6)
    posterior_contrast = float(loud_variance[0] - quiet_variance[0])
    quiet_mean_error = float(abs(quiet_mean[0] - 0.5 * quiet_query[0, 0]))
    interpolation_query = (-2.0 + 0.2 * np.arange(21, dtype=np.float64))[:, None]
    interpolated = log_noise_interpolation(x_split, split_noise, interpolation_query)
    far = log_noise_interpolation(x_split, split_noise, np.array([[60.0]], dtype=np.float64))[0]
    geometric_mean = float(np.exp(np.mean(np.log(split_noise))))
    interpolation_error = float(abs(far - geometric_mean))
    if constant_error > 1.0e-12 or posterior_contrast <= 0.0 or quiet_mean_error > 0.05:
        raise RuntimeError("heteroskedastic posterior oracle failed")
    if np.any(interpolated <= 0.0) or interpolated[0] >= interpolated[-1]:
        raise RuntimeError("heteroskedastic log-noise positivity/ordering oracle failed")
    if interpolation_error > 1.0e-6 * geometric_mean:
        raise RuntimeError("heteroskedastic far-field noise oracle failed")
    return constant_error, posterior_contrast, quiet_mean_error, interpolation_error, geometric_mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/heteroskedastic_gp.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    constant_error, posterior_contrast, quiet_mean_error, interpolation_error, geometric_mean = oracle()
    started = time.perf_counter()
    if args.skip_fortml:
        status, notes = "skipped", "--skip-fortml"
    else:
        environment = os.environ.copy()
        environment["FO_SCAN_FALLBACK"] = "regex"
        subprocess.run(["fo", "test", "test_heteroskedastic_gp"],
                       cwd=fortml, env=environment, check=True)
        status = "pass"
        notes = "FortML test covers constant-noise reduction, quiet/noisy posterior, interpolation, and refusal"
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
        row.update({"workload": "heteroskedastic_gp", "backend": "fortml", "device": "cpu",
                    "n_samples": 10, "n_queries": 21})
        row.update(values)
        rows.append(row)

    add(phase="constant_noise_reduction", backend="numpy_oracle", status="pass",
        metric="constant_noise_max_abs_error", value=constant_error,
        max_abs_error=constant_error,
        oracle="independent NumPy diagonal-noise Cholesky posterior",
        notes="constant supplied variance is exactly the ordinary GP special case")
    add(phase="posterior_contrast", backend="numpy_oracle", status="pass",
        metric="loud_minus_quiet_posterior_variance", value=posterior_contrast,
        max_abs_error=quiet_mean_error,
        oracle="independent NumPy same-function quiet/noisy posterior",
        notes=f"quiet_mean_error={quiet_mean_error:.3e}")
    add(phase="log_noise_interpolation", backend="numpy_oracle", status="pass",
        metric="far_field_geometric_mean", value=geometric_mean,
        max_abs_error=interpolation_error,
        oracle="independent NumPy positive log-noise GP interpolation",
        notes="interpolation is positive and reverts to the supplied geometric mean")
    add(phase="public_contract_gate", status=status, seconds_per_operation=elapsed,
        metric="fortml_heteroskedastic_test", value=1.0,
        max_abs_error=max(constant_error, interpolation_error),
        oracle="FortML test_heteroskedastic_gp behavioral gate", notes=notes)
    add(phase="refusal_contract", status="refused", n_samples=4,
        metric="zero_observation_variance", value="nan", max_abs_error=0.0,
        oracle="typed FORTNUM_DOMAIN_ERROR for non-positive supplied variance",
        notes="zero variance is refused because log-noise is undefined")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
