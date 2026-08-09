#!/usr/bin/env python3
"""Correctness-gated finite-feature GP posterior variance benchmark.

The NumPy reference independently forms the hidden feature map and the
regularized precision matrix.  It checks both the posterior predictive
variance ``diag(Z A^-1 Z^T)`` and its analytic regularization JVP.  CUDA is a
typed capability row: CPU feature-map and solve timings are never relabeled.
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
from typing import Any

import numpy as np


N_SAMPLES = 256
N_FEATURES = 8
N_HIDDEN = 16
N_OUTPUTS = 2
REGULARIZATION = 0.1
REPETITIONS = 8
TOLERANCE = 3.0e-11
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_hidden", "n_outputs", "regularization", "repetitions",
    "seconds_per_operation", "metric", "value", "max_abs_error", "oracle",
    "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(1, N_SAMPLES + 1, dtype=np.float64)[:, None]
    columns = np.arange(1, N_FEATURES + 1, dtype=np.float64)[None, :]
    x = np.sin(0.017 * rows + 0.053 * columns) + np.cos(0.011 * rows * columns)
    target = np.empty((N_SAMPLES, N_OUTPUTS), dtype=np.float64)
    indices = np.arange(1, N_SAMPLES + 1, dtype=np.float64)
    for j in range(1, N_OUTPUTS + 1):
        target[:, j - 1] = 0.4 * np.sin(0.013 * indices * j) + 0.2 * np.cos(0.019 * (indices + j))
    return x, target


def layer(seed: int, layer_index: int, n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    scale = np.sqrt(6.0 / (n_in + n_out))
    indices = np.arange(1, n_in * n_out + 1, dtype=np.float64).reshape((n_in, n_out), order="F")
    weights = scale * np.sin(seed + 1009 * layer_index + 9176 * indices)
    biases = 0.01 * scale * np.sin(seed + 1009 * layer_index + 7919 * np.arange(1, n_out + 1, dtype=np.float64))
    return weights, biases


def oracle(x: np.ndarray, target: np.ndarray) -> dict[str, float]:
    weights, biases = layer(29, 1, N_FEATURES, N_HIDDEN)
    hidden = np.tanh(x @ weights + biases)
    design = np.concatenate([hidden, np.ones((N_SAMPLES, 1))], axis=1)
    started = time.perf_counter()
    precision = design.T @ design + REGULARIZATION * np.eye(N_HIDDEN + 1)
    coefficients = np.linalg.solve(precision, design.T @ target)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(REPETITIONS):
        prediction = design @ coefficients
    predict_seconds = (time.perf_counter() - started) / REPETITIONS
    started = time.perf_counter()
    solve = np.linalg.solve(precision, design.T)
    solve2 = np.linalg.solve(precision, solve)
    variance = np.einsum("ij,ji->i", design, solve)
    dvariance = -np.einsum("ij,ji->i", design, solve2)
    variance_seconds = time.perf_counter() - started
    return {
        "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
        "variance_seconds": variance_seconds,
        "mse": float(np.mean((prediction - target) ** 2)),
        "variance_checksum": float(np.sum(variance)),
        "derivative_checksum": float(np.sum(dvariance)),
    }


def row(details: dict[str, Any], **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {field: "" for field in FIELDS}
    result.update(details)
    result.update({
        "workload": "mlp_last_layer_gp_posterior", "phase": "posterior_variance", "backend": "",
        "device": "cpu", "status": "", "n_samples": N_SAMPLES, "n_features": N_FEATURES,
        "n_hidden": N_HIDDEN, "n_outputs": N_OUTPUTS, "regularization": REGULARIZATION,
        "repetitions": REPETITIONS, "seconds_per_operation": "", "metric": "", "value": "",
        "max_abs_error": "", "oracle": "", "notes": "",
    })
    result.update(values)
    return result


def run_fortml(fortml: Path, target: str, details: dict[str, Any], expected: dict[str, float]) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment, capture_output=True, text=True)
    if build.returncode:
        return row(details, backend="fortml", status="unavailable", oracle="FortML release-app protocol", notes="fo build failed")
    run = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml, env=environment, capture_output=True, text=True)
    if run.returncode:
        return row(details, backend="fortml", status="unavailable", oracle="FortML release-app protocol", notes="release app failed")
    pattern = re.compile(r"^mlp_last_layer_gp_posterior,\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*([0-9Ee+.-]+),\s*([0-9Ee+.-]+),\s*([0-9Ee+.-]+),\s*([0-9Ee+.-]+),\s*([0-9Ee+.-]+),\s*([0-9Ee+.-]+)$")
    match = next((pattern.match(line.strip()) for line in run.stdout.splitlines() if pattern.match(line.strip())), None)
    if match is None:
        return row(details, backend="fortml", status="unavailable", oracle="FortML release-app protocol", notes=f"record not found: {run.stdout!r}")
    values = [float(match.group(index)) for index in range(5, 11)]
    errors = [abs(values[index] - expected[name]) for index, name in enumerate(("fit_seconds", "predict_seconds", "variance_seconds", "mse", "variance_checksum", "derivative_checksum"))]
    error = max(errors[3:])
    if error > TOLERANCE:
        raise RuntimeError(f"FortML posterior variance mismatch: {error:.3e}")
    return row(details, backend="fortml", status="pass", seconds_per_operation=values[2], metric="variance_checksum", value=values[4], max_abs_error=error, oracle="independent NumPy precision solve", notes=f"mse={values[3]:.16e}; derivative_checksum={values[5]:.16e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/mlp_last_layer_gp_posterior.csv"))
    parser.add_argument("--target", default="fortml_bench_mlp_last_layer_gp_posterior")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    fortml = args.fortml.resolve()
    details = {"python_version": platform.python_version(), "numpy_version": np.__version__, "fortml_revision": revision(fortml), "benchmark_revision": revision(root, (output,)), "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3"}
    expected = oracle(*fixture())
    rows = [row(details, backend="numpy_oracle", status="pass", seconds_per_operation=expected["variance_seconds"], metric="variance_checksum", value=expected["variance_checksum"], max_abs_error=0.0, oracle="independent NumPy precision solve", notes=f"mse={expected['mse']:.16e}; derivative_checksum={expected['derivative_checksum']:.16e}")]
    rows.append(row(details, backend="fortml", status="skipped", oracle="FortML release-app protocol", notes="--skip-fortml") if args.skip_fortml else run_fortml(fortml, args.target, details, expected))
    rows.append(row(details, phase="device_capability", backend="fortml", device="cuda", status="unavailable", oracle="typed_device_contract", notes="resident CUDA feature-map/precision kernels are not implemented"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
