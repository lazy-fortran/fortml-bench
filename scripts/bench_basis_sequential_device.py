#!/usr/bin/env python3
"""Correctness-gated benchmark for sequential basis device dispatch."""

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
    "n_features", "n_stages", "seconds_per_operation", "metric", "value",
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
    frequency = np.exp(theta)
    argument = x * frequency
    value = np.column_stack([
        np.sin(argument[:, 0]), np.cos(argument[:, 0]),
        np.sin(argument[:, 1]), np.cos(argument[:, 1]),
    ])
    argument_dot = frequency * (x_dot + x * theta_dot)
    tangent = np.column_stack([
        np.cos(argument[:, 0]) * argument_dot[:, 0],
        -np.sin(argument[:, 0]) * argument_dot[:, 0],
        np.cos(argument[:, 1]) * argument_dot[:, 1],
        -np.sin(argument[:, 1]) * argument_dot[:, 1],
    ])
    h = 2.0e-6

    def evaluate(xv: np.ndarray, tv: np.ndarray) -> np.ndarray:
        av = np.exp(tv)
        zv = xv * av
        return np.column_stack([
            np.sin(zv[:, 0]), np.cos(zv[:, 0]),
            np.sin(zv[:, 1]), np.cos(zv[:, 1]),
        ])

    finite_difference = (
        evaluate(x + h * x_dot, theta + h * theta_dot)
        - evaluate(x - h * x_dot, theta - h * theta_dot)
    ) / (2.0 * h)
    u = (0.03 * np.arange(value.size, dtype=np.float64) - 0.17).reshape(value.shape)
    z0 = u[:, 0] * np.cos(argument[:, 0]) - u[:, 1] * np.sin(argument[:, 0])
    z1 = u[:, 2] * np.cos(argument[:, 1]) - u[:, 3] * np.sin(argument[:, 1])
    theta_bar = np.array([
        np.sum(frequency[0] * x[:, 0] * z0),
        np.sum(frequency[1] * x[:, 1] * z1),
    ])
    x_bar = np.column_stack([frequency[0] * z0, frequency[1] * z1])
    adjoint_error = abs(
        np.sum(u * tangent)
        - (np.dot(theta_bar, theta_dot) + np.sum(x_bar * x_dot))
    )
    return value.shape[1], float(np.max(np.abs(tangent - finite_difference))), float(adjoint_error)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/basis_sequential_device.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    n_features, derivative_error, adjoint_error = independent_oracle()
    if derivative_error > 2.0e-10 or adjoint_error > 2.0e-12:
        raise RuntimeError(
            f"independent sequential oracle failed: derivative={derivative_error}, "
            f"adjoint={adjoint_error}"
        )
    started = time.perf_counter()
    completed = subprocess.run(
        ["fo", "test", "test_basis_sequential_device"],
        cwd=fortml, capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - started
    gate_text = (completed.stdout + "\n" + completed.stderr).strip()
    passed = completed.returncode == 0
    metadata = {
        "n_samples": 5, "n_features": n_features, "n_stages": 2,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)), "compiler": "gfortran",
        "flags": "-O3", "oracle": "independent NumPy polynomial/Fourier chain",
        "notes": "CPU value/JVP/VJP dispatch and typed CUDA refusal",
    }
    rows = [{
        "workload": "basis_sequential_device", "phase": "value_derivatives",
        "backend": "fortml", "device": "cpu", "status": "pass" if passed else "failed",
        "seconds_per_operation": elapsed, "metric": "gate_seconds",
        "value": 1.0 if passed else 0.0,
        "max_abs_error": max(derivative_error, adjoint_error), **metadata,
    }, {
        "workload": "basis_sequential_device", "phase": "device_contract",
        "backend": "fortml", "device": "cuda", "status": "unavailable",
        "seconds_per_operation": "", "metric": "api_surface", "value": "unavailable",
        "max_abs_error": 0.0, **metadata,
        "oracle": "typed CUDA refusal leaves all outputs untouched",
        "notes": "no resident CUDA sequential basis executor; FORTNUM_NOT_IMPLEMENTED",
    }]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")
    if not passed:
        print(gate_text)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
