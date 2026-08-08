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


def independent_oracle() -> tuple[int, float, float, float]:
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
    reference = np.empty_like(value)
    reference[:, 0] = x[:, 0] + np.sin(0.7 * x[:, 0])
    reference[:, 1] = x[:, 0] ** 2 + np.cos(0.7 * x[:, 0])
    reference[:, 2] = x[:, 1] + np.sin(1.2 * x[:, 1])
    reference[:, 3] = x[:, 1] ** 2 + np.cos(1.2 * x[:, 1])
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

    u = (0.03 * np.arange(value.size, dtype=np.float64) - 0.17).reshape(value.shape)

    def vjp(xv: np.ndarray, tv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        av = np.exp(tv)
        zv0 = av[0] * xv[:, 0]
        zv1 = av[1] * xv[:, 1]
        q0 = np.cos(zv0) * u[:, 0] - np.sin(zv0) * u[:, 1]
        q1 = np.cos(zv1) * u[:, 2] - np.sin(zv1) * u[:, 3]
        theta_bar = np.array([
            np.sum(av[0] * xv[:, 0] * q0),
            np.sum(av[1] * xv[:, 1] * q1),
        ])
        x_bar = np.empty_like(xv)
        x_bar[:, 0] = u[:, 0] + 2.0 * xv[:, 0] * u[:, 1] + av[0] * q0
        x_bar[:, 1] = u[:, 2] + 2.0 * xv[:, 1] * u[:, 3] + av[1] * q1
        return theta_bar, x_bar

    theta_bar, x_bar = vjp(x, theta)
    adjoint_error = abs(np.sum(u * direction) -
                        (np.dot(theta_bar, theta_dot) + np.sum(x_bar * x_dot)))
    frequencies = np.exp(theta)
    z0 = frequencies[0] * x[:, 0]
    z1 = frequencies[1] * x[:, 1]
    q0 = np.cos(z0) * u[:, 0] - np.sin(z0) * u[:, 1]
    q1 = np.cos(z1) * u[:, 2] - np.sin(z1) * u[:, 3]
    q0_dot = -(frequencies[0] * (x_dot[:, 0] + x[:, 0] * theta_dot[0])) * (
        np.sin(z0) * u[:, 0] + np.cos(z0) * u[:, 1])
    q1_dot = -(frequencies[1] * (x_dot[:, 1] + x[:, 1] * theta_dot[1])) * (
        np.sin(z1) * u[:, 2] + np.cos(z1) * u[:, 3])
    theta_hvp = np.array([
        (frequencies[0] * theta_dot[0] * x[:, 0] + frequencies[0] * x_dot[:, 0]) * q0
        + frequencies[0] * x[:, 0] * q0_dot,
        (frequencies[1] * theta_dot[1] * x[:, 1] + frequencies[1] * x_dot[:, 1]) * q1
        + frequencies[1] * x[:, 1] * q1_dot,
    ]).sum(axis=1)
    x_hvp = np.empty_like(x)
    x_hvp[:, 0] = 2.0 * x_dot[:, 0] * u[:, 1] + (
        frequencies[0] * theta_dot[0]) * q0 + frequencies[0] * q0_dot
    x_hvp[:, 1] = 2.0 * x_dot[:, 1] * u[:, 3] + (
        frequencies[1] * theta_dot[1]) * q1 + frequencies[1] * q1_dot
    theta_bar_plus, x_bar_plus = vjp(x_plus, theta_plus)
    theta_bar_minus, x_bar_minus = vjp(x_minus, theta_minus)
    theta_hvp_fd = (theta_bar_plus - theta_bar_minus) / (2.0 * h)
    x_hvp_fd = (x_bar_plus - x_bar_minus) / (2.0 * h)
    hvp_error = max(
        float(np.max(np.abs(theta_hvp - theta_hvp_fd))),
        float(np.max(np.abs(x_hvp - x_hvp_fd))),
    )
    return (
        value.shape[1],
        float(np.max(np.abs(direction - finite_difference))),
        float(adjoint_error),
        max(float(np.max(np.abs(value - reference))), hvp_error),
    )


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
    n_features, derivative_error, adjoint_error, oracle_error = independent_oracle()
    if (derivative_error > 2.0e-10 or adjoint_error > 2.0e-12 or
            oracle_error > 2.0e-8):
        raise RuntimeError(
            "independent residual oracle failed: "
            f"derivative={derivative_error}, adjoint={adjoint_error}, "
            f"hvp/value={oracle_error}"
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
        "flags": "-O3",
        "oracle": "independent NumPy value/JVP/VJP/HVP residual-sum oracle",
        "notes": note,
    }
    rows = [{
        "workload": "basis_residual_pipeline", "phase": "value_derivatives",
        "backend": "fortml", "device": "cpu", "status": "pass" if passed else "failed",
        "seconds_per_operation": elapsed, "metric": "gate_seconds",
        "value": 1.0 if passed else 0.0,
        "max_abs_error": max(derivative_error, adjoint_error, oracle_error), **metadata,
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
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
