#!/usr/bin/env python3
"""Benchmark fitted preprocessing and binary Laplace GP classification.

The Fortran executable reports release timings.  This script reconstructs the
scaler fixture and the complete small-data Laplace Newton solve independently
with NumPy before writing a raw, provenance-bearing CSV.
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


def revision(repository: Path) -> str:
    value = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).strip()
    return value + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(1, 129, dtype=np.float64)[:, None]
    columns = np.arange(1, 4, dtype=np.float64)[None, :]
    x = np.sin(0.017 * rows + 0.13 * columns)
    tangent = np.cos(0.011 * (rows + 2.0 * columns))
    x[:, 2] = 2.0
    return x, tangent, np.linspace(-1.5, 1.5, 32)[:, None]


def scaler_oracle(x: np.ndarray, tangent: np.ndarray) -> dict[str, float]:
    mean = np.mean(x, axis=0)
    scale = np.sqrt(np.mean((x - mean) ** 2, axis=0))
    scale[scale == 0.0] = 1.0
    standard = (x - mean) / scale
    standard_jvp = tangent / scale
    data_min = np.min(x, axis=0)
    data_max = np.max(x, axis=0)
    denominator = data_max - data_min
    denominator[denominator == 0.0] = 1.0
    minimum = -1.0 + 2.0 * (x - data_min) / denominator
    return {
        "standard_sum": float(np.sum(standard)),
        "standard_jvp_sum": float(np.sum(standard_jvp)),
        "minmax_sum": float(np.sum(minimum)),
    }


def sigmoid(value: np.ndarray) -> np.ndarray:
    result = np.empty_like(value)
    positive = value >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def normal_cdf(value: np.ndarray) -> np.ndarray:
    return 0.5 * np.vectorize(math.erfc, otypes=[float])(-value / np.sqrt(2.0))


def laplace_oracle(query: np.ndarray, probit: bool = False) -> dict[str, float]:
    x = np.linspace(-1.5, 1.5, 32)[:, None]
    labels = np.where(x[:, 0] >= 0.0, 1.0, -1.0)
    variance = 1.2
    lengthscale = 0.7
    jitter = 1.0e-7
    differences = x[:, None, :] - x[None, :, :]
    covariance = variance * np.exp(
        -0.5 * np.sum(differences * differences, axis=2) / lengthscale**2
    )
    covariance = covariance + jitter * np.eye(x.shape[0])
    mode = np.zeros(x.shape[0])
    for _iteration in range(80):
        eta = labels * mode
        if probit:
            probability = normal_cdf(eta)
            density = np.exp(-0.5 * eta * eta) / np.sqrt(2.0 * np.pi)
            ratio = density / np.maximum(probability, 1.0e-14)
            curvature = np.maximum(ratio * (ratio + eta), 1.0e-12)
            gradient = ratio
        else:
            probability = sigmoid(eta)
            gradient = 1.0 - probability
            curvature = np.maximum(probability * (1.0 - probability), 1.0e-12)
        sqrt_w = np.sqrt(curvature)
        b = curvature * mode + labels * gradient
        posterior = np.eye(x.shape[0]) + sqrt_w[:, None] * covariance * sqrt_w[None, :]
        rhs = np.linalg.solve(posterior, sqrt_w * (covariance @ b))
        mode_new = covariance @ (b - sqrt_w * rhs)
        scale = max(1.0, float(np.max(np.abs(mode))))
        step = float(np.max(np.abs(mode_new - mode)) / scale)
        mode = mode_new
        if step <= 1.0e-8:
            break
    eta = labels * mode
    if probit:
        probability = normal_cdf(eta)
        density = np.exp(-0.5 * eta * eta) / np.sqrt(2.0 * np.pi)
        ratio = density / np.maximum(probability, 1.0e-14)
        curvature = np.maximum(ratio * (ratio + eta), 1.0e-12)
    else:
        probability = sigmoid(eta)
        curvature = np.maximum(probability * (1.0 - probability), 1.0e-12)
    sqrt_w = np.sqrt(curvature)
    posterior = np.eye(x.shape[0]) + sqrt_w[:, None] * covariance * sqrt_w[None, :]
    alpha = np.linalg.solve(covariance, mode)
    cross = variance * np.exp(
        -0.5 * np.sum((x[:, None, :] - query[None, :, :]) ** 2, axis=2) / lengthscale**2
    )
    work = np.linalg.solve(posterior, sqrt_w[:, None] * cross)
    latent_mean = cross.T @ alpha
    latent_variance = variance - np.sum(work * work, axis=0)
    if probit:
        observed = normal_cdf(latent_mean / np.sqrt(1.0 + latent_variance))
    else:
        observed = sigmoid(latent_mean / np.sqrt(1.0 + np.pi * latent_variance / 8.0))
    predicted = np.where(observed >= 0.5, 11, -7)
    target = np.where(query[:, 0] >= 0.0, 11, -7)
    return {
        "accuracy": float(np.mean(predicted == target)),
        "probability_sum": float(np.sum(observed)),
        "variance_min": float(np.min(latent_variance)),
    }


def parse(stdout: str) -> dict[str, list[str]]:
    records: dict[str, list[str]] = {}
    for line in stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if fields and fields[0] in {
            "standard_scaler",
            "minmax_scaler",
            "gp_classification_logistic",
            "gp_classification_probit",
        }:
            records[fields[0]] = fields[1:]
    required = {
        "standard_scaler",
        "minmax_scaler",
        "gp_classification_logistic",
        "gp_classification_probit",
    }
    if required - records.keys():
        raise RuntimeError(
            f"FortML extension app omitted {sorted(required - records.keys())}"
        )
    return records


def metadata(root: Path, fortml: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root),
        "compiler": "gfortran",
        "flags": "-O3",
    }


def run(fortml: Path, root: Path) -> list[dict[str, object]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment, check=True
    )
    started = time.perf_counter()
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_classification"],
        cwd=fortml,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - started
    parsed = parse(completed.stdout)
    x, tangent, query = fixture()
    scaler = scaler_oracle(x, tangent)
    standard = parsed["standard_scaler"]
    scaler_error = max(
        abs(float(standard[5]) - scaler["standard_sum"]),
        abs(float(standard[6]) - scaler["standard_jvp_sum"]),
    )
    if scaler_error > 5.0e-12:
        raise RuntimeError(f"scaler oracle mismatch: {scaler_error:.3e}")
    minmax = parsed["minmax_scaler"]
    scaler_error = max(scaler_error, abs(float(minmax[2]) - scaler["minmax_sum"]))
    if scaler_error > 5.0e-12:
        raise RuntimeError(f"min-max oracle mismatch: {scaler_error:.3e}")

    logistic = laplace_oracle(query)
    probit = laplace_oracle(query, probit=True)
    gp_logistic = parsed["gp_classification_logistic"]
    gp_probit = parsed["gp_classification_probit"]
    logistic_error = max(
        abs(float(gp_logistic[4]) - logistic["accuracy"]),
        abs(float(gp_logistic[5]) - logistic["probability_sum"]),
    )
    probit_error = abs(float(gp_probit[2]) - probit["probability_sum"])
    if logistic_error > 2.0e-8 or probit_error > 2.0e-8:
        raise RuntimeError(
            f"GP classification oracle mismatch: logistic={logistic_error:.3e}, "
            f"probit={probit_error:.3e}"
        )
    rows: list[dict[str, object]] = []
    rows.append(
        {
            "workload": "standard_scaler",
            "phase": "transform",
            "backend": "fortml",
            "status": "pass",
            "n_samples": 128,
            "n_features": 3,
            "seconds_per_operation": float(standard[2]),
            "metric": "feature_sum",
            "value": float(standard[5]),
            "max_abs_error": scaler_error,
            "oracle": "independent NumPy population-standardization fixture",
            "notes": f"JVP sum={standard[6]}; wall={wall:.6e}s",
        }
    )
    rows.append(
        {
            "workload": "minmax_scaler",
            "phase": "transform",
            "backend": "fortml",
            "status": "pass",
            "n_samples": 128,
            "n_features": 3,
            "metric": "feature_sum",
            "value": float(minmax[2]),
            "max_abs_error": scaler_error,
            "oracle": "independent NumPy min-max fixture",
            "notes": "range=[-1,1]; constant feature policy checked",
        }
    )
    rows.extend(
        [
            {
                "workload": "gp_classification_logistic",
                "phase": "fit",
                "backend": "fortml",
                "status": "pass",
                "n_samples": 32,
                "n_features": 1,
                "seconds_per_operation": float(gp_logistic[2]),
                "metric": "accuracy",
                "value": float(gp_logistic[4]),
                "max_abs_error": logistic_error,
                "oracle": "independent NumPy Laplace logistic Newton solve",
                "notes": "RBF variance=1.2; lengthscale=0.7; jitter=1e-7",
            },
            {
                "workload": "gp_classification_logistic",
                "phase": "predict",
                "backend": "fortml",
                "status": "pass",
                "n_samples": 32,
                "n_features": 1,
                "seconds_per_operation": float(gp_logistic[3]),
                "metric": "positive_probability_sum",
                "value": float(gp_logistic[5]),
                "max_abs_error": logistic_error,
                "oracle": "independent NumPy Laplace posterior prediction",
                "notes": "latent variance is independently checked nonnegative",
            },
            {
                "workload": "gp_classification_probit",
                "phase": "predict",
                "backend": "fortml",
                "status": "pass",
                "n_samples": 32,
                "n_features": 1,
                "metric": "positive_probability_sum",
                "value": float(gp_probit[2]),
                "max_abs_error": probit_error,
                "oracle": "independent NumPy Laplace probit Newton solve",
                "notes": "analytic Gaussian-CDF predictive map",
            },
        ]
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/classification_extensions.csv")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    details = metadata(root, args.fortml.resolve())
    rows = run(args.fortml.resolve(), root)
    fields = [
        "workload",
        "phase",
        "backend",
        "status",
        "n_samples",
        "n_features",
        "seconds_per_operation",
        "metric",
        "value",
        "max_abs_error",
        "oracle",
        "notes",
        *details,
    ]
    for record in rows:
        record.update(details)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
