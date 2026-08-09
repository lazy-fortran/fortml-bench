#!/usr/bin/env python3
"""Correctness-gated independent multilabel Laplace-GP HPO benchmark.

The NumPy oracle keeps each fitted Laplace mode fixed while changing one RBF
log-variance/log-lengthscale block per label. It independently solves the
candidate prior systems and checks the exact value, gradient, JVP, VJP, and
central directional finite difference reported by the FortML probe.
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
    "n_labels", "n_features", "n_parameters", "seconds_per_operation",
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
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fit_head(x: np.ndarray, labels: np.ndarray, weights: np.ndarray,
             variance: float, lengthscale: float) -> tuple[np.ndarray, np.ndarray]:
    d2 = (x[:, None] - x[None, :]) ** 2
    covariance = variance * np.exp(-0.5 * d2 / lengthscale**2)
    covariance[np.diag_indices_from(covariance)] += 1.0e-7
    signed = 2.0 * labels.astype(np.float64) - 1.0
    mode = np.zeros(x.size, dtype=np.float64)
    for _ in range(100):
        margin = signed * mode
        probability = 1.0 / (1.0 + np.exp(-margin))
        likelihood_gradient = 1.0 - probability
        curvature = np.maximum(probability * (1.0 - probability), 1.0e-12)
        sqrt_w = np.sqrt(np.maximum(weights * curvature, 1.0e-12))
        b = weights * curvature * mode + signed * weights * likelihood_gradient
        system = np.eye(x.size) + sqrt_w[:, None] * covariance * sqrt_w[None, :]
        rhs = np.linalg.solve(system, sqrt_w * (covariance @ b))
        new_mode = covariance @ (b - sqrt_w * rhs)
        scale = max(1.0, float(np.max(np.abs(mode))))
        if np.max(np.abs(new_mode - mode)) / scale <= 1.0e-9:
            mode = new_mode
            break
        mode = new_mode
    return mode, signed


def fixed_objective_gradient(theta: np.ndarray, x: np.ndarray,
                             labels: np.ndarray, weights: np.ndarray,
                             modes: list[np.ndarray],
                             signed: list[np.ndarray]) -> tuple[float, np.ndarray]:
    d2 = (x[:, None] - x[None, :]) ** 2
    value = 0.0
    gradient = np.zeros(theta.size, dtype=np.float64)
    for index in range(labels.shape[1]):
        variance, lengthscale = np.exp(theta[2 * index:2 * index + 2])
        base = variance * np.exp(-0.5 * d2 / lengthscale**2)
        covariance = base.copy()
        covariance[np.diag_indices_from(covariance)] += 1.0e-7
        mode = modes[index]
        alpha = np.linalg.solve(covariance, mode)
        margin = signed[index] * mode
        log_likelihood = np.sum(weights * (-np.logaddexp(0.0, -margin)))
        value += 0.5 * float(mode @ alpha) - float(log_likelihood)
        d_covariance = (base, base * d2 / lengthscale**2)
        gradient[2 * index:2 * index + 2] = -0.5 * np.array([
            alpha @ d_covariance[0] @ alpha,
            alpha @ d_covariance[1] @ alpha,
        ])
    return value, gradient


def oracle() -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    x = np.array([-2.0, -1.5, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 1.5, 2.0])
    labels = np.array([[0, 1], [0, 1], [0, 1], [0, 0], [0, 0],
                       [1, 0], [1, 0], [1, 1], [1, 1], [1, 1]], dtype=np.int64)
    weights = np.array([1.0, 0.9, 1.1, 1.0, 0.8, 1.2, 1.0, 1.1, 0.9, 1.0])
    theta = np.log([1.3, 0.75, 1.3, 0.75])
    modes, signed = [], []
    for index in range(labels.shape[1]):
        mode, sign = fit_head(x, labels[:, index], weights, 1.3, 0.75)
        modes.append(mode)
        signed.append(sign)
    value, gradient = fixed_objective_gradient(theta, x, labels, weights, modes, signed)
    direction = np.array([0.17, -0.11, -0.13, 0.07])
    step = 1.0e-5
    value_plus, _ = fixed_objective_gradient(theta + step * direction, x, labels,
                                             weights, modes, signed)
    value_minus, _ = fixed_objective_gradient(theta - step * direction, x, labels,
                                              weights, modes, signed)
    return theta, gradient, value, float((value_plus - value_minus) / (2.0 * step)), direction


def parse_probe(output: str) -> dict[str, list[list[str]]]:
    records: dict[str, list[list[str]]] = {}
    for line in output.splitlines():
        if not line.startswith("gp_multilabel_independent_"):
            continue
        fields = next(csv.reader([line]))
        records.setdefault(fields[0], []).append(fields[1:])
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gp_multilabel_independent_optimizer.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml, output = args.fortml.resolve(), args.output.resolve()
    theta, expected_gradient, expected_value, expected_jvp, direction = oracle()
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)), "compiler": "gfortran",
        "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(details)
        row.update({"workload": "gp_multilabel_independent_optimizer", "backend": "fortml",
                    "device": "cpu", "n_samples": 10, "n_labels": 2,
                    "n_features": 1, "n_parameters": 4})
        row.update(values)
        rows.append(row)

    add(phase="independent_oracle", backend="numpy_oracle", status="pass",
        metric="objective_gradient_jvp", value=expected_jvp, max_abs_error=0.0,
        oracle="independent NumPy fixed-mode prior solves and analytic RBF contractions",
        notes="negative summed mode posterior; per-label [log variance, log lengthscale]")
    started = time.perf_counter()
    if args.skip_fortml:
        status, records, notes = "skipped", {}, "--skip-fortml"
    else:
        environment = os.environ.copy()
        environment["FO_SCAN_FALLBACK"] = "regex"
        completed = subprocess.run(
            ["fo", "exec", "--no-build", "fortml_bench_gp_multilabel_independent_optimizer"],
            cwd=fortml, env=environment, check=True, capture_output=True, text=True,
        )
        status, records, notes = "pass", parse_probe(completed.stdout), "release probe"
    elapsed = time.perf_counter() - started
    if status == "pass":
        observed_value = float(records["gp_multilabel_independent_objective"][0][0])
        observed_gradient = np.array(
            [float(v) for v in records["gp_multilabel_independent_gradient"][0]])
        observed_jvp = float(records["gp_multilabel_independent_jvp"][0][0])
        observed_vjp = np.array(
            [float(v) for v in records["gp_multilabel_independent_vjp"][0]])
        fd_values = np.array(
            [float(v) for v in records["gp_multilabel_independent_fd_values"][0]])
        error = float(max(abs(observed_value - expected_value),
                          np.max(abs(observed_gradient - expected_gradient)),
                          abs(observed_jvp - expected_jvp),
                          np.max(abs(observed_vjp - expected_gradient)),
                          abs((fd_values[0] - fd_values[1]) / (2.0e-5) - expected_jvp)))
        if error > 3.0e-6:
            raise RuntimeError(f"independent multilabel GP oracle mismatch: {error:.3e}")
        result = records["gp_multilabel_independent_result"][0]
        result_status, converged = int(result[0]), result[1] == "T"
        result_objective = float(result[3])
        if result_status != 0 or not converged or result_objective >= expected_value:
            raise RuntimeError("FortOpt independent multilabel GP result did not converge")
        cuda_code = int(records["gp_multilabel_independent_cuda"][0][0])
        if cuda_code != 0:
            raise RuntimeError("unexpected independent multilabel GP CUDA capability")
    else:
        error = float("nan")
        cuda_code = 0
        result_objective = expected_value
    add(phase="fixed_state_products", status=status, seconds_per_operation=elapsed,
        metric="objective_gradient_max_abs_error", value=expected_value,
        max_abs_error=error, oracle="independent NumPy fixed-mode gradient/JVP/VJP oracle",
        notes=notes)
    add(phase="lbfgsb", status=status, seconds_per_operation=elapsed,
        metric="negative_objective_decrease", value=expected_value - result_objective,
        max_abs_error=error, oracle="FortOpt bounded L-BFGS-B independent label blocks",
        notes="fixed-state outer objective; bounds=[-1,1]")
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_multilabel_gp_hpo", value="nan", max_abs_error="nan",
        oracle="typed CPU-only capability refusal", notes=f"device_supported(CUDA)={cuda_code}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
