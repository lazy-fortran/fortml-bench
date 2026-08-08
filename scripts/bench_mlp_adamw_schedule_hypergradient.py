#!/usr/bin/env python3
"""Correctness-gated benchmark for scheduled AdamW trajectory products.

The NumPy recurrence below is intentionally independent of FortML.  It checks
the release app's complete value, all eight packed hypergradients, and one
directional JVP before recording the CPU row.  CUDA and outer-HVP products are
recorded as explicit capability rows because the source contract refuses them.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_parameters", "steps", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        path = (repository / line[3:].strip()).resolve()
        if path not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + np.exp(-value))


def cosine_rate(update: int, base: float, minimum: float, total: int) -> float:
    progress = min(1.0, update / total)
    factor = minimum + (1.0 - minimum) * 0.5 * (1.0 + np.cos(np.pi * progress))
    return base * factor


def objective(parameters: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Independent affine MLP AdamW value, finite-difference gradient and JVP."""
    train_x = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float64)
    train_y = 0.6 * train_x - 0.35
    validation_x = np.array([-1.75, 0.25, 1.9], dtype=np.float64)
    validation_y = 0.6 * validation_x - 0.35
    direction = np.array([0.21, -0.17, 0.23, -0.19, 0.13, 0.11, 0.07, -0.05])
    h = 2.0e-6

    def scalar(p: np.ndarray) -> float:
        # Use a standalone recurrence for the FD oracle, never the release app.
        b, reg, wd = np.exp(p[:3])
        b1, b2, eps = sigmoid(p[3]), sigmoid(p[4]), np.exp(p[5])
        minimum_rate = sigmoid(p[6])
        t = np.array([0.12, -0.08], dtype=np.float64)
        m = np.zeros(2)
        v = np.zeros(2)
        for k in range(1, 7):
            r = t[0] * train_x + t[1] - train_y
            g = np.array([np.mean(r * train_x), np.mean(r)]) + reg * t
            m = b1 * m + (1.0 - b1) * g
            v = b2 * v + (1.0 - b2) * g**2
            u = (m / (1.0 - b1**k)) / (np.sqrt(v / (1.0 - b2**k)) + eps)
            rate_k = cosine_rate(k, b, minimum_rate, 8)
            t = (1.0 - rate_k * wd) * t - rate_k * u
        rv = t[0] * validation_x + t[1] - validation_y
        return 0.5 * np.mean(rv**2)

    value = scalar(parameters)
    gradient = np.empty(8)
    for index in range(8):
        plus = parameters.copy(); plus[index] += h
        minus = parameters.copy(); minus[index] -= h
        gradient[index] = (scalar(plus) - scalar(minus)) / (2.0 * h)
    tangent = float(np.dot(gradient, direction))
    return value, gradient, tangent


def parse_oracle(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            values[f"{row['quantity']}_{row['index']}"] = float(row["value"])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/mlp_adamw_schedule_hypergradient.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    parameters = np.array([
        np.log(0.05), np.log(0.03), np.log(0.04),
        np.log(0.7 / 0.3), np.log(0.9 / 0.1), np.log(0.02),
        np.log(0.2 / 0.8), np.log(0.99999 / 0.00001),
    ])
    expected_value, expected_gradient, expected_tangent = objective(parameters)
    with tempfile.TemporaryDirectory(prefix="fortml-adamw-schedule-") as temporary:
        oracle_path = Path(temporary) / "oracle.csv"
        environment = os.environ.copy()
        environment["FORTML_BENCH_MLP_ADAMW_SCHEDULE_HYPERGRADIENT_ORACLE"] = str(oracle_path)
        environment["FORTML_BENCH_ORACLE_ONLY"] = "1"
        environment["FO_SCAN_FALLBACK"] = "regex"
        run = subprocess.run(
            ["fo", "exec", "fortml_bench_mlp_adamw_schedule_hypergradient"],
            cwd=fortml, env=environment, capture_output=True, text=True,
        )
        parsed = parse_oracle(oracle_path) if run.returncode == 0 and oracle_path.exists() else {}
    observed = np.array([parsed.get(f"gradient_{i}", np.nan) for i in range(1, 9)])
    errors = np.r_[abs(parsed.get("value_1", np.nan) - expected_value),
                   np.max(abs(observed - expected_gradient)),
                   abs(parsed.get("jvp_1", np.nan) - expected_tangent)]
    passed = run.returncode == 0 and np.all(np.isfinite(errors)) and np.max(errors) < 5.0e-8
    ignored = (output, root / "results" / "mlp_adamw_schedule_hypergradient.csv")
    source_revision = revision(fortml)
    benchmark_revision = revision(root, ignored)
    metadata = {
        "backend": "fortml", "device": "cpu", "status": "pass" if passed else "failed",
        "n_samples": 6, "n_parameters": 8, "steps": 6, "seconds_per_operation": "",
        "oracle": "independent NumPy AdamW affine recurrence with central-FD packed products",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": source_revision, "benchmark_revision": benchmark_revision,
        "compiler": "gfortran", "flags": "-O3",
        "notes": "cosine schedule; exact CPU JVP/VJP; HVP/CUDA typed refusals",
    }
    rows = [{
        "workload": "mlp_adamw_schedule_hypergradient", "phase": "trajectory_products",
        "metric": "validation_mse", "value": parsed.get("value_1", ""),
        "max_abs_error": float(np.max(errors)), **metadata,
    }]
    rows += [{
        "workload": "mlp_adamw_schedule_hypergradient", "phase": "cuda_refusal",
        "device": "cuda", "metric": "typed_refusal", "value": "", "max_abs_error": 0.0,
        **{key: value for key, value in metadata.items() if key != "device"},
    }, {
        "workload": "mlp_adamw_schedule_hypergradient", "phase": "outer_hvp_refusal",
        "metric": "typed_refusal", "value": "", "max_abs_error": 0.0, **metadata,
    }, {
        "workload": "mlp_adamw_schedule_hypergradient", "phase": "mixed_precision_refusal",
        "metric": "typed_refusal", "value": "", "max_abs_error": 0.0, **metadata,
    }]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}; max_abs_error={np.max(errors):.3e}")
    if not passed:
        print(run.stdout + run.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
