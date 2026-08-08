#!/usr/bin/env python3
"""Correctness-gated PCA-seeded linear-MLP benchmark.

NumPy's centered thin SVD is the independent oracle.  The FortML row is
retained only when the two-layer MLP reconstruction agrees with the same
rank-truncated PCA map; CUDA is recorded as an explicit refusal.
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


N_SAMPLES = 512
N_FEATURES = 16
N_COMPONENTS = 8
REPETITIONS = 16
RMSE_TOLERANCE = 2.0e-10
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_components", "repetitions", "seconds_per_operation",
    "metric", "value", "max_abs_error", "oracle", "python_version",
    "numpy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
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


def fixture() -> np.ndarray:
    rows = np.arange(1, N_SAMPLES + 1, dtype=np.float64)[:, None]
    columns = np.arange(1, N_FEATURES + 1, dtype=np.float64)[None, :]
    return np.sin(0.013 * rows + 0.071 * columns) + np.cos(0.009 * rows * columns)


def oracle(x: np.ndarray) -> tuple[float, float]:
    center = x.mean(axis=0)
    centered = x - center
    components = np.linalg.svd(centered, full_matrices=False)[2][:N_COMPONENTS]
    started = time.perf_counter()
    for _ in range(REPETITIONS):
        projected = (x - center) @ components.T
        reconstructed = projected @ components + center
    elapsed = (time.perf_counter() - started) / REPETITIONS
    rmse = float(np.sqrt(np.mean((reconstructed - x) ** 2)))
    if not np.all(np.isfinite(reconstructed)):
        raise RuntimeError("NumPy PCA reconstruction is nonfinite")
    return elapsed, rmse


def row(details: dict[str, Any], **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {field: "" for field in FIELDS}
    result.update(details)
    result.update({
        "workload": "mlp_pca_initializer", "phase": "predict", "backend": "",
        "device": "cpu", "status": "", "n_samples": N_SAMPLES,
        "n_features": N_FEATURES, "n_components": N_COMPONENTS,
        "repetitions": REPETITIONS, "seconds_per_operation": "", "metric": "",
        "value": "", "max_abs_error": "", "oracle": "", "notes": "",
    })
    result.update(values)
    return result


def run_fortml(fortml: Path, target: str, details: dict[str, Any], expected: float) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                           env=environment, capture_output=True, text=True)
    if build.returncode:
        note = build.stderr.strip().splitlines()[-1] if build.stderr.strip() else "fo build failed"
        return row(details, backend="fortml", status="unavailable", oracle="FortML release-app protocol", notes=note)
    run = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                         env=environment, capture_output=True, text=True)
    if run.returncode:
        note = run.stderr.strip().splitlines()[-1] if run.stderr.strip() else "release app failed"
        return row(details, backend="fortml", status="unavailable", oracle="FortML release-app protocol", notes=note)
    pattern = re.compile(
        r"^mlp_pca_initializer_predict,\s*(\d+),\s*(\d+),\s*(\d+),\s*([0-9Ee+.-]+),\s*([0-9Ee+.-]+)$"
    )
    match = next((pattern.match(line.strip()) for line in run.stdout.splitlines()
                  if pattern.match(line.strip())), None)
    if match is None:
        return row(details, backend="fortml", status="unavailable",
                   oracle="FortML release-app protocol", notes=f"timing record not found: {run.stdout!r}")
    seconds = float(match.group(4))
    value = float(match.group(5))
    error = abs(value - expected)
    if error > RMSE_TOLERANCE:
        raise RuntimeError(f"FortML PCA MLP RMSE mismatch: {error:.3e}")
    return row(details, backend="fortml", status="pass", seconds_per_operation=seconds,
               metric="reconstruction_rmse", value=value, max_abs_error=error,
               oracle="independent NumPy centered thin-SVD reconstruction",
               notes=f"RMSE tolerance={RMSE_TOLERANCE:.1e}; CUDA is an explicit refusal")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/mlp_pca_initializer.csv"))
    parser.add_argument("--target", default="fortml_bench_mlp_pca_initializer")
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
    numpy_seconds, expected_rmse = oracle(fixture())
    rows = [row(details, backend="numpy_oracle", status="pass",
                seconds_per_operation=numpy_seconds, metric="reconstruction_rmse",
                value=expected_rmse, max_abs_error=0.0,
                oracle="independent NumPy centered thin-SVD reconstruction",
                notes="finite two-layer linear/PCA optimum")]
    if args.skip_fortml:
        rows.append(row(details, backend="fortml", status="skipped",
                        oracle="FortML release-app protocol", notes="--skip-fortml"))
    else:
        rows.append(run_fortml(fortml, args.target, details, expected_rmse))
    rows.append(row(details, phase="device_capability", backend="fortml", device="cuda",
                    status="unavailable", oracle="typed_device_contract",
                    notes="no resident CUDA MLP/PCA initializer; CPU timing is not relabeled"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
