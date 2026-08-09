#!/usr/bin/env python3
"""Correctness-gated weighted ordinary-least-squares workload."""

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
    "workload", "phase", "backend", "device", "status", "metric", "value",
    "max_abs_error", "seconds", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
TOLERANCE = 3.0e-11


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    """Return the repository revision, marking unrelated working-tree edits."""
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names: set[str] = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            continue
    dirty = any(
        line[3:].split(" -> ")[-1].strip() not in ignored_names
        for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True,
        ).splitlines()
    )
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the deterministic release fixture."""
    x = np.empty((64, 3), dtype=np.float64)
    y = np.empty((64, 2), dtype=np.float64)
    weights = np.empty(64, dtype=np.float64)
    for i in range(64):
        x[i, 0] = i / 16.0
        x[i, 1] = np.sin(x[i, 0])
        x[i, 2] = np.cos(0.5 * x[i, 0])
        y[i, 0] = 0.3 + 1.1*x[i, 0] - 0.2*x[i, 1] + 0.5*x[i, 2]
        y[i, 1] = -0.4 + 0.7*x[i, 0] + 0.4*x[i, 1] - 0.6*x[i, 2]
        weights[i] = 0.5 + (i + 1) % 5 / 5.0
    return x, y, weights


def oracle() -> tuple[np.ndarray, float]:
    """Solve the weighted normal equations independently in NumPy."""
    x, y, weights = fixture()
    design = np.column_stack((np.ones(x.shape[0]), x))
    gram = (design.T * weights) @ design
    rhs = (design.T * weights) @ y
    coefficients = np.linalg.solve(gram, rhs)
    prediction_mean = float(np.mean(design @ coefficients))
    return coefficients, prediction_mean


def run_app(fortml: Path) -> tuple[dict[str, float | str], float]:
    env = os.environ.copy()
    env.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, env=env,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    started = time.perf_counter()
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_weighted_ols"],
        cwd=fortml, env=env, check=True, capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - started
    values: dict[str, float | str] = {}
    for line in completed.stdout.splitlines():
        if line.strip() == "weighted_ols_cuda,unavailable":
            values["weighted_ols_cuda_status"] = "unavailable"
            continue
        if line.startswith("weighted_ols_") and "," in line:
            name, raw = line.split(",", 1)
            if name.startswith("weighted_ols_"):
                values[name.strip()] = float(raw.strip())
    required = {
        "weighted_ols_prediction_mean", "weighted_ols_parameter_checksum",
        "weighted_ols_parameter_count", "weighted_ols_cuda_status",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise RuntimeError(f"release app omitted {missing}")
    return values, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/weighted_ols.csv"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("results/WEIGHTED_OLS.md"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    expected_coefficients, expected_prediction_mean = oracle()
    expected_checksum = float(np.sum(expected_coefficients))
    values, seconds = run_app(fortml)
    prediction_error = abs(
        float(values["weighted_ols_prediction_mean"]) - expected_prediction_mean,
    )
    checksum_error = abs(
        float(values["weighted_ols_parameter_checksum"]) - expected_checksum,
    )
    if prediction_error > TOLERANCE or checksum_error > TOLERANCE:
        raise RuntimeError(
            "NumPy weighted OLS mismatch: "
            f"prediction={prediction_error:g}, checksum={checksum_error:g}",
        )
    if int(values["weighted_ols_parameter_count"]) != 8:
        raise RuntimeError("unexpected weighted OLS parameter dimension")
    if values["weighted_ols_cuda_status"] != "unavailable":
        raise RuntimeError("weighted OLS CUDA refusal changed")
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (args.output.resolve(), args.report.resolve())),
        "compiler": "gfortran", "flags": "-O3",
    }
    rows = [
        {**details, "workload": "weighted_ols", "phase": "fit_predict",
         "backend": "fortml", "device": "cpu", "status": "pass",
         "metric": "prediction_mean",
         "value": float(values["weighted_ols_prediction_mean"]),
         "max_abs_error": prediction_error, "seconds": seconds,
         "oracle": "independent NumPy weighted normal equations",
         "notes": "dense 64x3, two outputs, intercept, nonnegative weights"},
        {**details, "workload": "weighted_ols", "phase": "packed_state",
         "backend": "fortml", "device": "cpu", "status": "pass",
         "metric": "parameter_checksum",
         "value": float(values["weighted_ols_parameter_checksum"]),
         "max_abs_error": checksum_error, "seconds": seconds,
         "oracle": "independent NumPy weighted normal equations",
         "notes": "Fortran column-major packed coefficient state"},
        {**details, "workload": "weighted_ols", "phase": "capability_check",
         "backend": "fortml", "device": "cuda", "status": "unavailable",
         "metric": "status", "value": 3.0, "max_abs_error": "",
         "seconds": seconds, "oracle": "declared resident-device contract",
         "notes": "FORTNUM_NOT_IMPLEMENTED; no host fallback"},
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# Weighted ordinary least squares\n\n"
        "This lane compares deterministic weighted multi-output ordinary least "
        "squares with an independent NumPy weighted normal-equation oracle. "
        "It checks the packed fitted state and records the typed CUDA refusal; "
        "positive constraints, derivative-through-fit, and resident GPU solves "
        "are not claimed.\n\n"
        f"- FortML revision: `{details['fortml_revision']}`\n"
        f"- Benchmark revision: `{details['benchmark_revision']}`\n"
        f"- Compiler/flags: `{details['compiler']} {details['flags']}`\n"
        f"- Release wall time: `{seconds:.6g}` s\n"
        f"- Prediction-mean error: `{prediction_error:.6g}`\n"
        f"- Packed-checksum error: `{checksum_error:.6g}`\n"
        f"- Raw record: [`{args.output}`]({args.output.name})\n",
    )
    print(f"wrote {args.output} ({len(rows)} rows)")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
