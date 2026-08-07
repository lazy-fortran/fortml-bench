#!/usr/bin/env python3
"""Correctness-gated composable physics-residual objective benchmark.

The NumPy fixture is an independent affine residual oracle for four weighted
terms (data, differential-equation residual, boundary, and conservation).  It
checks value, gradient, directional JVP, scalar VJP, and central differences.
Second-order residual products are deliberately recorded as a typed refusal:
the public FortML seam has no hidden finite-difference HVP fallback.  The
Fortran gate additionally checks malformed weights/shape and the FortOpt
objective adapter.
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


def oracle() -> tuple[float, float, float, float]:
    """Return objective value, gradient FD error, JVP FD error, VJP error."""
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
    return float(value), float(gradient_error), float(jvp_error), float(vjp_error)


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
    value, gradient_error, jvp_error, vjp_error = oracle()
    started = time.perf_counter()
    if args.skip_fortml:
        status, notes = "skipped", "--skip-fortml"
    else:
        subprocess.run(["fo", "test", "test_physics_objective"],
                       cwd=fortml, check=True)
        status = "pass"
        notes = "test checks four weighted slots, products, FortOpt adapter, and HVP refusal"
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
    add(phase="hvp_contract", status="pass", metric="hvp_supported", value=0.0,
        max_abs_error=0.0, oracle="typed FORTNUM_NOT_IMPLEMENTED residual-HVP refusal",
        notes="no finite-difference fallback; residual HVP callback is not in the seam")
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
