#!/usr/bin/env python3
"""Correctness-gated PCA-initialized tied linear-autoencoder benchmark.

NumPy's centered thin SVD is the independent oracle.  The FortML row is
accepted only when its rank-truncated reconstruction RMSE agrees with the
oracle; CUDA is reported as a refusal because the current implementation has
no resident matrix-product lowering.
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
    "numpy_version", "sklearn_version", "fortml_revision",
    "benchmark_revision", "compiler", "flags", "notes",
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


def package_version(name: str) -> str:
    try:
        module = __import__(name)
    except ImportError:
        return "unavailable"
    return str(getattr(module, "__version__", "installed"))


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    ignored = (output, root / "results" / "linear_autoencoder.csv")
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": package_version("sklearn"),
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def fixture() -> np.ndarray:
    rows = np.arange(1, N_SAMPLES + 1, dtype=np.float64)[:, None]
    columns = np.arange(1, N_FEATURES + 1, dtype=np.float64)[None, :]
    return np.sin(0.013 * rows + 0.071 * columns) + np.cos(0.009 * rows * columns)


def oracle(x: np.ndarray) -> tuple[float, float]:
    centered = x - x.mean(axis=0)
    started = time.perf_counter()
    for _ in range(REPETITIONS):
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:N_COMPONENTS]
        reconstructed = centered @ components.T @ components + x.mean(axis=0)
    elapsed = (time.perf_counter() - started) / REPETITIONS
    rmse = float(np.sqrt(np.mean((reconstructed - x) ** 2)))
    orthogonality = float(np.max(np.abs(components @ components.T - np.eye(N_COMPONENTS))))
    if orthogonality > 1.0e-12:
        raise RuntimeError(f"NumPy PCA loading orthogonality failed: {orthogonality:.3e}")
    return elapsed, rmse


def base_row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = dict(details)
    row.update({
        "workload": "linear_autoencoder", "phase": "reconstruct", "backend": "",
        "device": "cpu", "status": "", "n_samples": N_SAMPLES,
        "n_features": N_FEATURES, "n_components": N_COMPONENTS,
        "repetitions": REPETITIONS, "seconds_per_operation": "", "metric": "",
        "value": "", "max_abs_error": "", "oracle": "", "notes": "",
    })
    row.update(values)
    return row


def run_fortml(fortml: Path, target: str, details: dict[str, str], expected: float) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                           env=environment, capture_output=True, text=True)
    if build.returncode != 0:
        note = build.stderr.strip().splitlines()[-1] if build.stderr.strip() else "fo build failed"
        return base_row(details, backend="fortml", status="unavailable",
                        oracle="FortML release-app protocol", notes=note)
    run = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                         env=environment, capture_output=True, text=True)
    if run.returncode != 0:
        note = run.stderr.strip().splitlines()[-1] if run.stderr.strip() else "release app failed"
        return base_row(details, backend="fortml", status="unavailable",
                        oracle="FortML release-app protocol", notes=note)
    pattern = re.compile(
        r"^linear_autoencoder_reconstruct,\s*(\d+),\s*(\d+),\s*(\d+),\s*([0-9Ee+.-]+),\s*([0-9Ee+.-]+)$"
    )
    match = next((pattern.match(line.strip()) for line in run.stdout.splitlines()
                  if pattern.match(line.strip())), None)
    if match is None:
        return base_row(details, backend="fortml", status="unavailable",
                        oracle="FortML release-app protocol",
                        notes=f"timing record not found: {run.stdout!r}")
    seconds = float(match.group(4))
    value = float(match.group(5))
    error = abs(value - expected)
    if error > RMSE_TOLERANCE:
        raise RuntimeError(f"FortML linear-autoencoder RMSE mismatch: {error:.3e}")
    return base_row(details, backend="fortml", status="pass",
                    seconds_per_operation=seconds, metric="reconstruction_rmse",
                    value=value, max_abs_error=error,
                    oracle="independent NumPy centered thin-SVD reconstruction",
                    notes=f"RMSE tolerance={RMSE_TOLERANCE:.1e}; CUDA is an explicit refusal")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/linear_autoencoder.csv"))
    parser.add_argument("--target", default="fortml_bench_linear_autoencoder")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    fortml = args.fortml.resolve()
    details = metadata(root, fortml, output)
    numpy_seconds, expected_rmse = oracle(fixture())
    rows = [base_row(details, backend="numpy_oracle", status="pass",
                     seconds_per_operation=numpy_seconds, metric="reconstruction_rmse",
                     value=expected_rmse, max_abs_error=0.0,
                     oracle="independent NumPy centered thin-SVD reconstruction",
                     notes="rank-truncated tied linear optimum")]
    if args.skip_fortml:
        rows.append(base_row(details, backend="fortml", status="skipped",
                             oracle="FortML release-app protocol", notes="--skip-fortml"))
    else:
        rows.append(run_fortml(fortml, args.target, details, expected_rmse))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
