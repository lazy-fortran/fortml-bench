#!/usr/bin/env python3
"""Correctness-gated benchmark for learned basis fan-in.

The NumPy fixture derives value and first-order products without importing the
Fortran implementation. The Fortran unit gate supplies the independent HVP,
transaction, metadata, and device-refusal checks.
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_branches", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version",
    "numpy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    )
    for line in status.splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.array(
        [[0.2, 0.8], [-0.4, 1.1], [0.7, -0.3], [0.1, 0.6], [-0.5, -0.9]],
        dtype=np.float64,
    )
    x_dot = np.array(
        [[-0.3, 0.6], [0.2, -0.2], [0.1, 0.3], [-0.4, 0.4], [0.5, -0.1]],
        dtype=np.float64,
    )
    theta = np.array([1.25, -0.4, np.log(0.7), np.log(1.1)])
    theta_dot = np.array([-0.05, 0.04, 0.09, -0.07])
    return x, x_dot, theta, theta_dot


def components(x: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frequencies = np.exp(theta[2:4])
    polynomial = np.column_stack(
        [x[:, 0], x[:, 0] ** 2, x[:, 1], x[:, 1] ** 2]
    )
    fourier = np.column_stack([
        np.sin(frequencies[0] * x[:, 0]),
        np.cos(frequencies[0] * x[:, 0]),
        np.sin(frequencies[1] * x[:, 1]),
        np.cos(frequencies[1] * x[:, 1]),
    ])
    return polynomial, fourier


def evaluate(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    polynomial, fourier = components(x, theta)
    return theta[0] * polynomial + theta[1] * fourier


def analytic_products() -> tuple[float, float, float]:
    x, x_dot, theta, theta_dot = fixture()
    polynomial, fourier = components(x, theta)
    frequencies = np.exp(theta[2:4])
    polynomial_dot = np.column_stack([
        x_dot[:, 0], 2.0 * x[:, 0] * x_dot[:, 0],
        x_dot[:, 1], 2.0 * x[:, 1] * x_dot[:, 1],
    ])
    argument_dot = frequencies * (x_dot + x * theta_dot[2:4])
    fourier_dot = np.column_stack([
        np.cos(frequencies[0] * x[:, 0]) * argument_dot[:, 0],
        -np.sin(frequencies[0] * x[:, 0]) * argument_dot[:, 0],
        np.cos(frequencies[1] * x[:, 1]) * argument_dot[:, 1],
        -np.sin(frequencies[1] * x[:, 1]) * argument_dot[:, 1],
    ])
    direction = (
        theta_dot[0] * polynomial + theta[0] * polynomial_dot
        + theta_dot[1] * fourier + theta[1] * fourier_dot
    )
    h = 2.0e-6
    finite_difference = (
        evaluate(x + h * x_dot, theta + h * theta_dot)
        - evaluate(x - h * x_dot, theta - h * theta_dot)
    ) / (2.0 * h)
    u = (0.037 * np.arange(20, dtype=np.float64) - 0.23).reshape(5, 4)
    z0 = frequencies[0] * x[:, 0]
    z1 = frequencies[1] * x[:, 1]
    q0 = np.cos(z0) * u[:, 0] - np.sin(z0) * u[:, 1]
    q1 = np.cos(z1) * u[:, 2] - np.sin(z1) * u[:, 3]
    theta_bar = np.array([
        np.sum(u * polynomial),
        np.sum(u * fourier),
        theta[1] * np.sum(frequencies[0] * x[:, 0] * q0),
        theta[1] * np.sum(frequencies[1] * x[:, 1] * q1),
    ])
    x_bar = np.empty_like(x)
    x_bar[:, 0] = theta[0] * (
        u[:, 0] + 2.0 * x[:, 0] * u[:, 1]
    ) + theta[1] * frequencies[0] * q0
    x_bar[:, 1] = theta[0] * (
        u[:, 2] + 2.0 * x[:, 1] * u[:, 3]
    ) + theta[1] * frequencies[1] * q1
    adjoint_error = abs(
        np.sum(u * direction)
        - np.dot(theta_bar, theta_dot)
        - np.sum(x_bar * x_dot)
    )
    value_reference = theta[0] * polynomial + theta[1] * fourier
    value_error = float(np.max(np.abs(evaluate(x, theta) - value_reference)))
    return (
        float(np.max(np.abs(direction - finite_difference))),
        float(adjoint_error),
        value_error,
    )


def parse_metrics(output: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for line in output.splitlines():
        if line.startswith("basis_blend_") and "," in line:
            key, value = line.strip().split(",", 1)
            metrics[key] = value
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/basis_blend_pipeline.csv"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()

    derivative_error, adjoint_error, value_error = analytic_products()
    oracle_error = max(derivative_error, adjoint_error, value_error)
    if derivative_error > 2.0e-9 or adjoint_error > 2.0e-12:
        raise RuntimeError(
            "independent blend oracle failed: "
            f"direction={derivative_error}, adjoint={adjoint_error}"
        )
    build = subprocess.run(
        ["fo", "build"], cwd=fortml, capture_output=True, text=True,
    )
    unit = subprocess.run(
        ["fo", "test", "test_basis_blend_pipeline"], cwd=fortml,
        capture_output=True, text=True,
    )
    app = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_basis_blend_pipeline"],
        cwd=fortml, capture_output=True, text=True,
    )
    metrics = parse_metrics(app.stdout + "\n" + app.stderr)
    passed = build.returncode == 0 and unit.returncode == 0 and app.returncode == 0
    seconds = metrics.get("basis_blend_transform_seconds", "")
    app_value_error = float(metrics.get("basis_blend_value_error", "inf"))
    app_adjoint_error = float(metrics.get("basis_blend_adjoint_error", "inf"))
    max_error = max(oracle_error, app_value_error, app_adjoint_error)
    note = (
        "NumPy value/JVP/VJP oracle plus Fortran HVP, transaction, metadata, "
        "CPU dispatch, and typed accelerator refusal gates"
    )
    if not passed:
        combined = (
            build.stdout + build.stderr + unit.stdout + unit.stderr
            + app.stdout + app.stderr
        ).strip()
        note += ": " + (combined.splitlines()[-1] if combined else "no gate output")

    ignored = (output, root / "results" / "basis_blend_pipeline.csv")
    metadata = {
        "n_samples": 4096, "n_features": 4, "n_branches": 2,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml, (
            fortml / "test_mlp_amsgrad_checkpoint.txt",
            fortml / "test_mlp_radam_checkpoint.txt",
        )),
        "benchmark_revision": revision(root, ignored), "compiler": "gfortran",
        "flags": "-O3",
        "oracle": "independent NumPy learned-blend value/JVP/VJP oracle",
        "notes": note,
    }
    rows = [{
        "workload": "basis_blend_pipeline", "phase": "value_derivatives",
        "backend": "fortml", "device": "cpu",
        "status": "pass" if passed else "failed",
        "seconds_per_operation": seconds, "metric": "transform_seconds",
        "value": 1.0 if passed else 0.0, "max_abs_error": max_error,
        **metadata,
    }]
    for device in ("cuda", "openacc"):
        device_metadata = dict(metadata)
        device_metadata.update({
            "oracle": f"typed {device.upper()} refusal preserves output buffers",
            "notes": f"no resident {device.upper()} blend executor; "
                     "FORTNUM_NOT_IMPLEMENTED",
        })
        rows.append({
            "workload": "basis_blend_pipeline", "phase": "device_contract",
            "backend": "fortml", "device": device, "status": "unavailable",
            "seconds_per_operation": "", "metric": "api_surface",
            "value": "unavailable", "max_abs_error": 0.0, **device_metadata,
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
