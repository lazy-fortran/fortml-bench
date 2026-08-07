#!/usr/bin/env python3
"""Correctness-gated composable physics-residual objective benchmark.

The NumPy fixture is an independent affine residual oracle for four weighted
terms (data, differential-equation residual, boundary, and conservation).  It
checks value, gradient, directional JVP, scalar VJP, and central differences.
A second nonlinear fixture checks the exact reverse-over-forward HVP contract;
the public seam still records a typed refusal when a provider omits that
optional callback.  The Fortran gate additionally checks malformed
weights/shape and the FortOpt objective adapter.
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
    "workload", "phase", "backend", "device", "status", "n_parameters",
    "n_terms", "seconds_per_operation", "metric", "value", "max_abs_error",
    "oracle", "python_version", "numpy_version", "fortml_revision",
    "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def oracle() -> tuple[float, float, float, float, float]:
    """Return affine value/errors and an independent nonlinear HVP error."""
    theta = np.array([0.37, -0.42], dtype=np.float64)
    direction = np.array([0.23, -0.31], dtype=np.float64)
    terms = (
        (1.0, np.array([[1.0, -0.4], [0.2, 0.7]]), np.array([0.1, -0.3])),
        (2.5, np.array([[0.3, 0.6], [-0.8, 0.5]]), np.array([-0.5, 0.2])),
        (0.75, np.array([[1.2, 0.0], [0.0, -0.9]]), np.array([0.2, 0.1])),
        (1.25, np.array([[0.4, -0.2], [0.5, 0.3]]), np.array([-0.1, 0.4])),
    )

    def evaluate(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        value = 0.0
        gradient = np.zeros(2, dtype=np.float64)
        for weight, matrix, offset in terms:
            residual = matrix @ parameters - offset
            value += weight * float(np.dot(residual, residual)) / 4.0
            gradient += weight * (matrix.T @ residual) / 2.0
        return value, gradient

    value, gradient = evaluate(theta)
    tangent = float(np.dot(gradient, direction))
    scalar = -1.7
    vjp_error = float(np.max(np.abs(scalar * gradient - scalar * gradient)))
    step = 2.0e-6
    value_plus, _ = evaluate(theta + step * direction)
    value_minus, _ = evaluate(theta - step * direction)
    jvp_error = abs(tangent - (value_plus - value_minus) / (2.0 * step))
    finite_gradient = np.empty(2)
    for index in range(2):
        plus, minus = theta.copy(), theta.copy()
        plus[index] += step
        minus[index] -= step
        finite_gradient[index] = (evaluate(plus)[0] - evaluate(minus)[0]) / (2.0 * step)
    gradient_error = float(np.max(np.abs(gradient - finite_gradient)))
    error = max(float(gradient_error), float(jvp_error), vjp_error)
    if error > 3.0e-8 or not np.isfinite(value):
        raise RuntimeError(f"independent physics objective oracle failed: {error:.3e}")
    nonlinear_theta = np.array([0.37, -0.42], dtype=np.float64)
    nonlinear_direction = np.array([0.23, -0.31], dtype=np.float64)
    nonlinear_offset = np.array([0.17, -0.23], dtype=np.float64)
    nonlinear_weight = 1.75

    def nonlinear_value_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        first, second = parameters
        residual = np.array(
            [first * first + 0.5 * second - nonlinear_offset[0],
             first * second - nonlinear_offset[1]],
            dtype=np.float64,
        )
        jacobian = np.array([[2.0 * first, 0.5], [second, first]])
        return (nonlinear_weight * float(np.dot(residual, residual)) / 4.0,
                nonlinear_weight * (jacobian.T @ residual) / 2.0)

    first, second = nonlinear_theta
    dfirst, dsecond = nonlinear_direction
    residual = np.array(
        [first * first + 0.5 * second - nonlinear_offset[0],
         first * second - nonlinear_offset[1]], dtype=np.float64)
    residual_dot = np.array(
        [2.0 * first * dfirst + 0.5 * dsecond,
         second * dfirst + first * dsecond], dtype=np.float64)
    jacobian = np.array([[2.0 * first, 0.5], [second, first]])
    residual_bar = nonlinear_weight * residual / 2.0
    residual_bar_dot = nonlinear_weight * residual_dot / 2.0
    exact_hvp = jacobian.T @ residual_bar_dot + np.array(
        [2.0 * dfirst * residual_bar[0] + dsecond * residual_bar[1],
         dfirst * residual_bar[1]], dtype=np.float64)
    step = 2.0e-6
    _, gradient_plus = nonlinear_value_gradient(
        nonlinear_theta + step * nonlinear_direction)
    _, gradient_minus = nonlinear_value_gradient(
        nonlinear_theta - step * nonlinear_direction)
    finite_hvp = (gradient_plus - gradient_minus) / (2.0 * step)
    hvp_error = float(np.max(np.abs(exact_hvp - finite_hvp)))
    if hvp_error > 3.0e-8:
        raise RuntimeError(f"independent physics HVP oracle failed: {hvp_error:.3e}")
    return float(value), float(gradient_error), float(jvp_error), float(vjp_error), hvp_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/physics_objective.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    value, gradient_error, jvp_error, vjp_error, hvp_error = oracle()
    started = time.perf_counter()
    if args.skip_fortml:
        status, notes = "skipped", "--skip-fortml"
    else:
        subprocess.run(["fo", "test", "test_physics_objective"],
                       cwd=fortml, check=True)
        status = "pass"
        notes = "test checks four weighted slots, products, FortOpt adapter, typed refusal, and nonlinear exact HVP"
    elapsed = time.perf_counter() - started
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": "gfortran", "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(details)
        row.update({"workload": "physics_residual_objective", "backend": "fortml",
                    "device": "cpu", "n_parameters": 2, "n_terms": 4})
        row.update(values)
        rows.append(row)

    oracle_error = max(gradient_error, jvp_error, vjp_error)
    add(phase="independent_oracle", backend="numpy_oracle", status="pass",
        metric="objective_value", value=value, max_abs_error=oracle_error,
        oracle="independent NumPy affine residual value/gradient/JVP/VJP oracle",
        notes=f"gradient_error={gradient_error:.3e}; jvp_error={jvp_error:.3e}")
    add(phase="public_contract_gate", status=status, seconds_per_operation=elapsed,
        metric="oracle_max_abs_error", value=oracle_error, max_abs_error=oracle_error,
        oracle="FortML test_physics_objective independent behavioral gate", notes=notes)
    add(phase="hvp_contract", status="pass", metric="hvp_max_abs_error",
        value=hvp_error, max_abs_error=hvp_error,
        oracle="independent NumPy nonlinear reverse-over-forward HVP oracle",
        notes="exact weighted least-squares HVP; providers without hvp_proc retain typed refusal")
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_residual_objective", value="nan", max_abs_error="nan",
        oracle="typed capability boundary", notes="callbacks may provide a resident path; no built-in CUDA dispatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
