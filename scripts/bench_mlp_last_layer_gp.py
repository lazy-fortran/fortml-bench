#!/usr/bin/env python3
"""Correctness-gated finite-feature last-layer GP/NTK initializer benchmark.

The NumPy row independently reproduces the deterministic hidden MLP feature
map and solves the augmented kernel-ridge normal equations.  The FortML row
is retained only when its reported MSE agrees with that oracle.  A CUDA row is
always explicit: CPU feature-map work is never relabeled as resident GPU.
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
MSE_TOLERANCE = 2.0e-12
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_hidden", "n_outputs", "regularization", "repetitions",
    "seconds_per_operation", "metric", "value", "max_abs_error", "oracle",
    "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(1, N_SAMPLES + 1, dtype=np.float64)[:, None]
    columns = np.arange(1, N_FEATURES + 1, dtype=np.float64)[None, :]
    x = np.sin(0.017 * rows + 0.053 * columns) + np.cos(0.011 * rows * columns)
    target = np.empty((N_SAMPLES, N_OUTPUTS), dtype=np.float64)
    for j in range(1, N_OUTPUTS + 1):
        indices = np.arange(1, N_SAMPLES + 1, dtype=np.float64)
        target[:, j - 1] = 0.4 * np.sin(0.013 * indices * j) + 0.2 * np.cos(
            0.019 * (indices + j)
        )
    return x, target


def layer(seed: int, layer_index: int, n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    scale = np.sqrt(6.0 / (n_in + n_out))
    indices = np.arange(1, n_in * n_out + 1, dtype=np.float64).reshape(
        (n_in, n_out), order="F"
    )
    phases = seed + 1009 * layer_index + 9176 * indices
    weights = scale * np.sin(phases)
    bias_indices = np.arange(1, n_out + 1, dtype=np.float64)
    biases = 0.01 * scale * np.sin(seed + 1009 * layer_index + 7919 * bias_indices)
    return weights, biases


def oracle(x: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    weights, biases = layer(29, 1, N_FEATURES, N_HIDDEN)
    hidden = np.tanh(x @ weights + biases)
    design = np.concatenate([hidden, np.ones((N_SAMPLES, 1))], axis=1)
    started = time.perf_counter()
    coefficients = np.linalg.solve(
        design.T @ design + REGULARIZATION * np.eye(N_HIDDEN + 1), design.T @ target
    )
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(REPETITIONS):
        prediction = design @ coefficients
    predict_seconds = (time.perf_counter() - started) / REPETITIONS
    mse = float(np.mean((prediction - target) ** 2))
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("NumPy posterior mean is nonfinite")
    return fit_seconds, predict_seconds, mse


def row(details: dict[str, Any], **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {field: "" for field in FIELDS}
    result.update(details)
    result.update({
        "workload": "mlp_last_layer_gp", "phase": "predict", "backend": "",
        "device": "cpu", "status": "", "n_samples": N_SAMPLES,
        "n_features": N_FEATURES, "n_hidden": N_HIDDEN, "n_outputs": N_OUTPUTS,
        "regularization": REGULARIZATION, "repetitions": REPETITIONS,
        "seconds_per_operation": "", "metric": "", "value": "",
        "max_abs_error": "", "oracle": "", "notes": "",
    })
    result.update(values)
    return result


def run_fortml(
    fortml: Path, target: str, details: dict[str, Any], expected: float
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    build = subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml,
        env=environment, capture_output=True, text=True,
    )
    if build.returncode:
        note = build.stderr.strip().splitlines()[-1] if build.stderr.strip() else "fo build failed"
        return row(details, backend="fortml", status="unavailable",
                   oracle="FortML release-app protocol", notes=note)
    run = subprocess.run(
        ["fo", "exec", "--no-build", target], cwd=fortml,
        env=environment, capture_output=True, text=True,
    )
    if run.returncode:
        note = run.stderr.strip().splitlines()[-1] if run.stderr.strip() else "release app failed"
        return row(details, backend="fortml", status="unavailable",
                   oracle="FortML release-app protocol", notes=note)
    pattern = re.compile(
        r"^mlp_last_layer_gp_predict,\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*"
        r"([0-9Ee+.-]+),\s*([0-9Ee+.-]+),\s*([0-9Ee+.-]+)$"
    )
    match = next((pattern.match(line.strip()) for line in run.stdout.splitlines()
                  if pattern.match(line.strip())), None)
    if match is None:
        return row(details, backend="fortml", status="unavailable",
                   oracle="FortML release-app protocol",
                   notes=f"timing record not found: {run.stdout!r}")
    value = float(match.group(7))
    error = abs(value - expected)
    if error > MSE_TOLERANCE:
        raise RuntimeError(f"FortML last-layer GP MSE mismatch: {error:.3e}")
    return row(details, backend="fortml", status="pass",
               phase="predict", seconds_per_operation=float(match.group(6)),
               metric="posterior_mse", value=value, max_abs_error=error,
               oracle="independent NumPy finite-feature kernel-ridge solve",
               notes=f"MSE tolerance={MSE_TOLERANCE:.1e}; fit_seconds={match.group(5)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/mlp_last_layer_gp.csv"))
    parser.add_argument("--target", default="fortml_bench_mlp_last_layer_gp")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    fortml = args.fortml.resolve()
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml), "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    numpy_fit, numpy_predict, expected_mse = oracle(*fixture())
    rows = [row(details, backend="numpy_oracle", status="pass", phase="fit",
                seconds_per_operation=numpy_fit, metric="posterior_mse", value=expected_mse,
                max_abs_error=0.0,
                oracle="independent NumPy finite-feature kernel-ridge solve",
                notes=f"predict_seconds={numpy_predict:.16e}; fixed-feature approximation")]
    if args.skip_fortml:
        rows.append(row(details, backend="fortml", status="skipped",
                        oracle="FortML release-app protocol", notes="--skip-fortml"))
    else:
        rows.append(run_fortml(fortml, args.target, details, expected_mse))
    rows.append(row(details, phase="device_capability", backend="fortml", device="cuda",
                    status="unavailable", oracle="typed_device_contract",
                    notes="resident CUDA feature-map/solve kernels are not implemented"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
