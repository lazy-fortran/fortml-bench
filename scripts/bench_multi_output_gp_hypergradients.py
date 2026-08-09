#!/usr/bin/env python3
"""Correctness-gated benchmark for exact ICM GP likelihood products.

NumPy independently assembles the dense output-major ICM covariance and uses
central differences of the scalar log marginal likelihood as gradient/HVP
oracles.  The Fortran release app reports checksums and timings; the focused
Fortran test checks every coordinate and the transactional/device contracts.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_outputs", "repetitions", "seconds_per_operation",
    "metric", "value", "max_abs_error", "oracle", "python_version",
    "numpy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
)
N, D, P, RANK, REPETITIONS = 40, 2, 3, 2, 5


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


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.empty((N, D), dtype=np.float64)
    y = np.empty((N, P), dtype=np.float64)
    for i in range(N):
        x[i, 0] = -1.0 + 0.05 * i
        x[i, 1] = np.sin(0.13 * (i + 1))
        y[i, 0] = np.sin(0.8 * x[i, 0]) + 0.1 * x[i, 1]
        y[i, 1] = np.cos(0.6 * x[i, 0]) - 0.2 * x[i, 1]
        y[i, 2] = x[i, 0] * x[i, 1]
    weights = np.array([[0.8, 0.3], [-0.4, 0.6], [-0.2, 0.5]])
    independent = np.array([0.25, 0.35, 0.18])
    return x, y, weights, independent


def lml(coordinates: np.ndarray) -> float:
    x, y, _, _ = fixture()
    weights = coordinates[3:3 + P * RANK].reshape(P, RANK)
    independent = coordinates[3 + P * RANK:3 + P * RANK + P]
    kernel = np.empty((N, N))
    delta = x[:, None, :] - x[None, :, :]
    kernel[:, :] = 1.2 * np.exp(-0.5 * np.sum(delta * delta, axis=2) / 0.65**2)
    b = weights @ weights.T + np.diag(independent)
    covariance = np.kron(b, kernel)
    covariance.flat[:: covariance.shape[0] + 1] += np.exp(coordinates[2])
    alpha = np.linalg.solve(covariance, y.T.reshape(-1))
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise RuntimeError("NumPy ICM oracle covariance is not positive definite")
    return float(-0.5 * np.dot(y.T.reshape(-1), alpha) - 0.5 * logdet -
                 0.5 * N * P * np.log(2.0 * np.pi))


def gradient_oracle(coordinates: np.ndarray, step: float = 2.0e-5) -> np.ndarray:
    gradient = np.empty_like(coordinates)
    for i in range(coordinates.size):
        plus, minus = coordinates.copy(), coordinates.copy()
        plus[i] += step
        minus[i] -= step
        gradient[i] = (lml(plus) - lml(minus)) / (2.0 * step)
    return gradient


def products_oracle() -> tuple[float, float, float]:
    parameters = np.array([
        np.log(1.2), np.log(0.65), np.log(0.12),
        0.8, 0.3, -0.4, 0.6, -0.2, 0.5,
        0.25, 0.35, 0.18,
    ])
    direction = 0.03 * np.sin(0.17 * np.arange(1, parameters.size + 1))
    gradient = gradient_oracle(parameters)
    h = 2.0e-4
    hvp = (gradient_oracle(parameters + h * direction) -
           gradient_oracle(parameters - h * direction)) / (2.0 * h)
    return float(lml(parameters)), float(np.sum(gradient)), float(np.sum(hvp))


def parse_probe(stdout: str) -> dict[str, tuple[float, float]]:
    records: dict[str, tuple[float, float]] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if not fields or fields[0] != "multi_output_gp":
            continue
        if fields[1] == "likelihood_value_jvp" and len(fields) == 4:
            records[fields[1]] = (float(fields[2]), float(fields[3]))
        elif len(fields) == 7:
            records[fields[1]] = (float(fields[5]), float(fields[6]))
    required = {"hyperparameter_gradient", "hyperparameter_hvp", "likelihood_value_jvp"}
    missing = required - records.keys()
    if missing:
        raise RuntimeError(f"ICM release app omitted {sorted(missing)}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/multi_output_gp_hypergradients.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/MULTI_OUTPUT_GP_HYPERGRADIENTS.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    ignored_fortml = tuple((fortml / path).resolve() for path in (
        "test_lightgbm_ranking.snapshot", "test_mlp_amsgrad_checkpoint.txt",
        "test_mlp_loss_scaling_checkpoint.txt", "test_mlp_radam_checkpoint.txt",
    ))
    ignored_bench = (args.output.resolve(), args.report.resolve(),
                     (root / "results/OVR_LOGISTIC_PARTIAL_FIT.md").resolve(),
                     (root / "results/ovr_logistic_partial_fit.csv").resolve())
    environment = os.environ.copy()
    environment["FO_SCAN_FALLBACK"] = "regex"
    environment["FO_FC"] = "gfortran"
    subprocess.run(["fo", "test", "test_multi_output_gp_hypergradients"],
                   cwd=fortml, env=environment, check=True)
    probe = subprocess.run(
        ["fo", "exec", "fortml_bench_multi_output_gp_hypergradients"],
        cwd=fortml, env=environment, check=True, text=True, capture_output=True,
    )
    records = parse_probe(probe.stdout)
    expected_value, expected_gradient_sum, expected_hvp_sum = products_oracle()
    value, value_dot = records["likelihood_value_jvp"]
    value_time = ""
    gradient_time, gradient_sum = records["hyperparameter_gradient"]
    hvp_time, hvp_sum = records["hyperparameter_hvp"]
    oracle_parameters = np.array([
        np.log(1.2), np.log(0.65), np.log(0.12),
        0.8, 0.3, -0.4, 0.6, -0.2, 0.5, 0.25, 0.35, 0.18,
    ])
    oracle_direction = 0.03 * np.sin(0.17 * np.arange(1, 13))
    expected_jvp = float(np.dot(gradient_oracle(oracle_parameters), oracle_direction))
    value_error = max(abs(value - expected_value), abs(value_dot - expected_jvp))
    gradient_error = abs(gradient_sum - expected_gradient_sum)
    hvp_error = abs(hvp_sum - expected_hvp_sum)
    if gradient_error > 2.0e-4 or hvp_error > 3.0e-3:
        raise RuntimeError(f"ICM product oracle failed: gradient={gradient_error:.3e}, "
                           f"HVP={hvp_error:.3e}")
    metadata = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml, ignored_fortml),
        "benchmark_revision": revision(root, ignored_bench),
        "compiler": "gfortran", "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(phase: str, device: str, status: str, seconds: object, metric: str,
            value: object, error: object, notes: str) -> None:
        row = {field: "" for field in FIELDS}
        row.update(metadata)
        row.update({"workload": "multi_output_gp_hypergradients", "phase": phase,
                    "backend": "fortml_cpu" if device == "cpu" else "fortml_cuda",
                    "device": device, "status": status, "n_samples": N,
                    "n_features": D, "n_outputs": P, "repetitions": REPETITIONS,
                    "seconds_per_operation": seconds, "metric": metric,
                    "value": value, "max_abs_error": error,
                    "oracle": "independent NumPy dense ICM finite-difference oracle",
                    "notes": notes})
        rows.append(row)

    add("likelihood_jvp", "cpu", "pass", value_time, "jvp_value", value_dot,
        value_error, "scalar likelihood JVP composition")
    add("hyperparameter_gradient", "cpu", "pass", gradient_time, "gradient_sum",
        gradient_sum, gradient_error, "dense ICM central-difference oracle")
    add("hyperparameter_hvp", "cpu", "pass", hvp_time, "hvp_sum", hvp_sum,
        hvp_error, "dense ICM directional finite-difference oracle")
    add("hyperparameter_products", "cuda", "unavailable", "", "typed_refusal", "", "",
        "FORTNUM_NOT_IMPLEMENTED until resident ICM covariance and solve kernels are linked")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"oracle likelihood={expected_value:.12e} gradient_sum={expected_gradient_sum:.12e} "
          f"hvp_sum={expected_hvp_sum:.12e}")


if __name__ == "__main__":
    main()
