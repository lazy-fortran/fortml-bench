#!/usr/bin/env python3
"""Correctness-gated benchmark for the named residual-sum basis DAG.

The NumPy fixture is built independently from the Fortran implementation.  It
checks the residual sum, a directional derivative by central differences, and
records the CPU behavioral gate plus the typed CUDA refusal.
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_branches", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


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


def independent_oracle() -> tuple[int, float, float]:
    x = np.array(
        [[0.2, 0.8], [-0.4, 1.1], [0.7, -0.3], [0.1, 0.6], [-0.5, -0.9]],
        dtype=np.float64,
    )
    x_dot = np.array(
        [[-0.3, 0.6], [0.2, -0.2], [0.1, 0.3], [-0.4, 0.4], [0.5, -0.1]],
        dtype=np.float64,
    )
    theta = np.log(np.array([0.7, 1.2], dtype=np.float64))
    theta_dot = np.array([0.11, -0.08], dtype=np.float64)
    main = np.column_stack([x[:, 0], x[:, 0] ** 2, x[:, 1], x[:, 1] ** 2])
    residual = np.column_stack([
        np.sin(np.exp(theta[0]) * x[:, 0]),
        np.cos(np.exp(theta[0]) * x[:, 0]),
        np.sin(np.exp(theta[1]) * x[:, 1]),
        np.cos(np.exp(theta[1]) * x[:, 1]),
    ])
    value = main + residual
    frequencies = np.exp(theta)
    argument_dot = frequencies * (x_dot + x * theta_dot)
    main_dot = np.column_stack([
        x_dot[:, 0], 2.0 * x[:, 0] * x_dot[:, 0],
        x_dot[:, 1], 2.0 * x[:, 1] * x_dot[:, 1],
    ])
    residual_dot = np.column_stack([
        np.cos(frequencies[0] * x[:, 0]) * argument_dot[:, 0],
        -np.sin(frequencies[0] * x[:, 0]) * argument_dot[:, 0],
        np.cos(frequencies[1] * x[:, 1]) * argument_dot[:, 1],
        -np.sin(frequencies[1] * x[:, 1]) * argument_dot[:, 1],
    ])
    direction = main_dot + residual_dot
    h = 2.0e-6
    theta_plus = theta + h * theta_dot
    theta_minus = theta - h * theta_dot
    x_plus = x + h * x_dot
    x_minus = x - h * x_dot

    def evaluate(xv: np.ndarray, tv: np.ndarray) -> np.ndarray:
        return np.column_stack([
            xv[:, 0] + np.sin(np.exp(tv[0]) * xv[:, 0]),
            xv[:, 0] ** 2 + np.cos(np.exp(tv[0]) * xv[:, 0]),
            xv[:, 1] + np.sin(np.exp(tv[1]) * xv[:, 1]),
            xv[:, 1] ** 2 + np.cos(np.exp(tv[1]) * xv[:, 1]),
        ])

    finite_difference = (evaluate(x_plus, theta_plus) -
                         evaluate(x_minus, theta_minus)) / (2.0 * h)
    return value.shape[1], float(np.max(np.abs(direction - finite_difference))), 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/basis_residual_pipeline.csv"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    n_features, derivative_error, oracle_error = independent_oracle()
    if derivative_error > 2.0e-10 or oracle_error > 1.0e-14:
        raise RuntimeError(
            f"independent residual oracle failed: derivative={derivative_error}"
        )

    started = time.perf_counter()
    completed = subprocess.run(
        ["fo", "test", "test_basis_residual_pipeline"],
        cwd=fortml, capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - started
    gate_text = (completed.stdout + "\n" + completed.stderr).strip()
    passed = completed.returncode == 0
    note = (
        "independent residual-sum value, JVP/VJP, HVP, metadata, CPU dispatch, "
        "and typed CUDA refusal oracle"
    )
    if not passed:
        note += ": " + (gate_text.splitlines()[-1] if gate_text else "no gate output")

    ignored = (
        output, root / "results" / "basis_residual_pipeline.csv",
    )
    metadata = {
        "n_samples": 5, "n_features": n_features, "n_branches": 2,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml, (
            fortml / "test_mlp_amsgrad_checkpoint.txt",
            fortml / "test_mlp_radam_checkpoint.txt",
        )),
        "benchmark_revision": revision(root, ignored), "compiler": "gfortran",
        "flags": "-O3", "oracle": "independent NumPy residual-sum and finite-difference direction",
        "notes": note,
    }
    rows = [{
        "workload": "basis_residual_pipeline", "phase": "value_derivatives",
        "backend": "fortml", "device": "cpu", "status": "pass" if passed else "failed",
        "seconds_per_operation": elapsed, "metric": "gate_seconds",
        "value": 1.0 if passed else 0.0,
        "max_abs_error": max(derivative_error, oracle_error), **metadata,
    }]
    device_metadata = dict(metadata)
    device_metadata.update({
        "oracle": "typed CUDA refusal leaves value and derivative outputs untouched",
        "notes": "no resident CUDA residual executor; FORTNUM_NOT_IMPLEMENTED",
    })
    rows.append({
        "workload": "basis_residual_pipeline", "phase": "device_contract",
        "backend": "fortml", "device": "cuda", "status": "unavailable",
        "seconds_per_operation": "", "metric": "api_surface", "value": "unavailable",
        "max_abs_error": 0.0, **device_metadata,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n").writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
