#!/usr/bin/env python3
"""Correctness-gated column feature-union device benchmark."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_query", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
N_SAMPLES = 2048


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


def parse(stdout: str) -> dict[str, float | int | str]:
    values: dict[str, float | int | str] = {}
    for line in stdout.splitlines():
        if line.startswith("column_pipeline_transform_seconds,"):
            values["transform_seconds"] = float(line.split(",", 1)[1])
        elif line.startswith("column_pipeline_cpu_max_abs_error,"):
            values["max_abs_error"] = float(line.split(",", 1)[1])
        elif line.startswith("column_pipeline_feature_count,"):
            values["feature_count"] = int(line.split(",", 1)[1])
        elif line.startswith("column_pipeline_cuda,"):
            values["cuda"] = line.split(",", 1)[1].strip()
    required = {"transform_seconds", "max_abs_error", "feature_count", "cuda"}
    if set(values) != required:
        raise RuntimeError(f"release app omitted column-pipeline metrics: {sorted(values)}")
    return values


def independent_oracle() -> np.ndarray:
    indices = np.arange(N_SAMPLES, dtype=np.float64)
    x1 = -1.0 + 2.0 * np.mod(indices, 101.0) / 100.0
    x3 = np.cos(0.013 * (indices + 1.0))
    return np.column_stack((np.sin(0.8 * x3), np.cos(0.8 * x3), x1, x1**2))


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/column_pipeline_device.csv"))
    parser.add_argument("--target", default="fortml_bench_column_pipeline_device")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (args.output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                   env=environment, check=True, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE, text=True)
    completed = subprocess.run(
        ["fo", "exec", "--no-build", args.target], cwd=fortml,
        env=environment, check=True, capture_output=True, text=True,
    )
    observed = parse(completed.stdout)
    expected = independent_oracle()
    max_abs_error = float(observed["max_abs_error"])
    if int(observed["feature_count"]) != expected.shape[1] or max_abs_error > 2.0e-13:
        raise RuntimeError(
            f"column pipeline oracle mismatch: features={observed['feature_count']}, "
            f"error={max_abs_error:.3e}"
        )
    rows = [
        row(details, workload="column_basis_pipeline", phase="transform",
            backend="fortml", device="cpu", status="pass", n_samples=N_SAMPLES,
            n_features=expected.shape[1], n_query=N_SAMPLES,
            seconds_per_operation=observed["transform_seconds"],
            metric="feature_oracle_max_abs_error", value=max_abs_error,
            max_abs_error=max_abs_error,
            oracle="independent NumPy sin/cos and polynomial feature union",
            notes="column-selecting Fourier and polynomial stages"),
        row(details, workload="column_basis_pipeline", phase="device_contract",
            backend="fortml", device="cuda", status="unavailable",
            n_samples=N_SAMPLES, n_features=expected.shape[1], n_query=N_SAMPLES,
            metric="api_surface", value=observed["cuda"], max_abs_error=0.0,
            oracle="typed device capability contract",
            notes="no resident CUDA basis executor; typed refusal"),
    ]
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
