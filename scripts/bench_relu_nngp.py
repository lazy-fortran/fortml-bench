#!/usr/bin/env python3
"""Correctness-gated analytic ReLU NNGP covariance benchmark."""

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


N_LEFT, N_RIGHT, N_FEATURES, DEPTH, REPETITIONS = 192, 160, 8, 3, 8
TOLERANCE = 3.0e-10
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_left", "n_right",
    "n_features", "hidden_depth", "weight_variance", "bias_variance", "repetitions",
    "seconds_per_operation", "metric", "value", "max_abs_error", "oracle",
    "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
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
    left_rows = np.arange(1, N_LEFT + 1, dtype=np.float64)[:, None]
    right_rows = np.arange(1, N_RIGHT + 1, dtype=np.float64)[:, None]
    columns = np.arange(1, N_FEATURES + 1, dtype=np.float64)[None, :]
    x_left = np.sin(0.017 * left_rows + 0.053 * columns) + np.cos(0.011 * left_rows * columns)
    x_right = np.sin(0.019 * right_rows + 0.041 * columns) + np.cos(0.007 * right_rows * columns)
    return x_left, x_right


def oracle(x_left: np.ndarray, x_right: np.ndarray) -> tuple[float, float]:
    covariance = x_left @ x_right.T / N_FEATURES
    left_variance = np.sum(x_left**2, axis=1) / N_FEATURES
    right_variance = np.sum(x_right**2, axis=1) / N_FEATURES
    started = time.perf_counter()
    for _ in range(REPETITIONS):
        covariance = x_left @ x_right.T / N_FEATURES
        left = left_variance.copy()
        right = right_variance.copy()
        for _ in range(DEPTH):
            denominator = np.sqrt(left[:, None] * right[None, :])
            rho = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator != 0.0)
            rho = np.clip(rho, -1.0, 1.0)
            theta = np.arccos(rho)
            covariance = denominator * (np.sin(theta) + (np.pi - theta) * rho) / np.pi
            left = 0.5 * 2.0 * left
            right = 0.5 * 2.0 * right
    seconds = (time.perf_counter() - started) / REPETITIONS
    if not np.all(np.isfinite(covariance)):
        raise RuntimeError("NumPy ReLU NNGP covariance is nonfinite")
    return seconds, float(np.sum(covariance))


def row(details: dict[str, Any], **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {field: "" for field in FIELDS}
    result.update(details)
    result.update({
        "workload": "relu_nngp_covariance", "phase": "covariance", "backend": "",
        "device": "cpu", "status": "", "n_left": N_LEFT, "n_right": N_RIGHT,
        "n_features": N_FEATURES, "hidden_depth": DEPTH, "weight_variance": 2.0,
        "bias_variance": 0.0, "repetitions": REPETITIONS,
        "seconds_per_operation": "", "metric": "", "value": "", "max_abs_error": "",
        "oracle": "", "notes": "",
    })
    result.update(values)
    return result


def run_fortml(fortml: Path, target: str, details: dict[str, Any], expected: float) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment, capture_output=True, text=True)
    if build.returncode:
        return row(details, backend="fortml", status="unavailable", oracle="FortML release-app protocol",
                   notes=build.stderr.strip().splitlines()[-1] if build.stderr.strip() else "fo build failed")
    run = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml, env=environment,
                         capture_output=True, text=True)
    if run.returncode:
        return row(details, backend="fortml", status="unavailable", oracle="FortML release-app protocol",
                   notes=run.stderr.strip().splitlines()[-1] if run.stderr.strip() else "release app failed")
    pattern = re.compile(r"^relu_nngp_covariance,\s*(\d+),\s*(\d+),\s*(\d+),\s*([0-9Ee+.-]+),\s*([0-9Ee+.-]+)$")
    match = next((pattern.match(line.strip()) for line in run.stdout.splitlines() if pattern.match(line.strip())), None)
    if match is None:
        return row(details, backend="fortml", status="unavailable", oracle="FortML release-app protocol",
                   notes=f"timing record not found: {run.stdout!r}")
    seconds, actual = float(match.group(4)), float(match.group(5))
    error = abs(actual - expected)
    if error > TOLERANCE:
        raise RuntimeError(f"FortML ReLU NNGP checksum mismatch: {error:.3e}")
    return row(details, backend="fortml", status="pass", seconds_per_operation=seconds,
               metric="covariance_checksum", value=actual, max_abs_error=error,
               oracle="independent NumPy arc-cosine recurrence",
               notes=f"checksum tolerance={TOLERANCE:.1e}; exact infinite-width kernel")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/relu_nngp.csv"))
    parser.add_argument("--target", default="fortml_bench_relu_nngp")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root, output, fortml = Path(__file__).resolve().parents[1], args.output.resolve(), args.fortml.resolve()
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml), "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    numpy_seconds, expected = oracle(*fixture())
    rows = [row(details, backend="numpy_oracle", status="pass", seconds_per_operation=numpy_seconds,
                metric="covariance_checksum", value=expected, max_abs_error=0.0,
                oracle="independent NumPy arc-cosine recurrence",
                notes="exact infinite-width ReLU covariance")]
    if args.skip_fortml:
        rows.append(row(details, backend="fortml", status="skipped", oracle="FortML release-app protocol",
                        notes="--skip-fortml"))
    else:
        rows.append(run_fortml(fortml, args.target, details, expected))
    rows.append(row(details, phase="device_capability", backend="fortml", device="cuda", status="unavailable",
                    oracle="typed_device_contract", notes="resident CUDA ReLU NNGP kernel is not implemented"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
