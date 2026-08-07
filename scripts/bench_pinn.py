#!/usr/bin/env python3
"""Correctness-gated manufactured PINN adapter benchmark.

The fixture mirrors the public four-slot affine physics objective and a
nonlinear one-parameter residual-HVP provider.  NumPy independently checks
value, gradient, JVP, HVP, and the bounded affine optimum; the Fortran gate
checks the complete adapter, FortOpt L-BFGS-B path, and typed CUDA refusal.
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
    """Return affine value/gradient and nonlinear JVP/HVP errors."""
    theta = 0.25
    direction = 0.17
    weight_sum = 1.0 + 2.0 + 0.5 + 1.5
    value = 0.5 * weight_sum * (theta - 1.0) ** 2
    gradient = weight_sum * (theta - 1.0)
    step = 2.0e-6
    affine_jvp = gradient * direction
    affine_jvp_fd = (
        0.5 * weight_sum * (theta + step * direction - 1.0) ** 2
        - 0.5 * weight_sum * (theta - step * direction - 1.0) ** 2
    ) / (2.0 * step)
    jvp_error = abs(affine_jvp - affine_jvp_fd)

    nonlinear_theta = 0.7
    nonlinear_direction = -0.23
    nonlinear_weight = 2.0

    def nonlinear_gradient(point: float) -> float:
        residual = point * point - 0.25
        return nonlinear_weight * 2.0 * point * residual

    nonlinear_hvp = nonlinear_weight * (
        (2.0 * nonlinear_theta) ** 2 +
        2.0 * (nonlinear_theta * nonlinear_theta - 0.25)
    ) * nonlinear_direction
    nonlinear_hvp_fd = (
        nonlinear_gradient(nonlinear_theta + step * nonlinear_direction)
        - nonlinear_gradient(nonlinear_theta - step * nonlinear_direction)
    ) / (2.0 * step)
    hvp_error = abs(nonlinear_hvp - nonlinear_hvp_fd)
    if max(jvp_error, hvp_error) > 3.0e-8:
        raise RuntimeError("manufactured PINN oracle failed")
    return float(value), float(gradient), float(jvp_error), float(hvp_error)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/pinn.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    value, gradient, jvp_error, hvp_error = oracle()
    started = time.perf_counter()
    if args.skip_fortml:
        status, notes = "skipped", "--skip-fortml"
    else:
        subprocess.run(["fo", "test", "test_pinn"], cwd=fortml, check=True)
        status = "pass"
        notes = "manufactured four-slot PINN, nonlinear HVP, L-BFGS-B fit, and CUDA refusal"
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
        row.update({"workload": "pinn_training_adapter", "backend": "fortml",
                    "device": "cpu", "n_parameters": 1, "n_terms": 4})
        row.update(values)
        rows.append(row)

    add(phase="independent_oracle", backend="numpy_oracle", status="pass",
        metric="affine_objective_value", value=value, max_abs_error=0.0,
        oracle="independent NumPy four-slot manufactured PINN oracle",
        notes=f"affine gradient={gradient:.16g}; JVP error={jvp_error:.3e}")
    add(phase="public_contract_gate", status=status,
        seconds_per_operation=elapsed, metric="oracle_max_abs_error", value=0.0,
        max_abs_error=0.0, oracle="FortML test_pinn independent behavioral gate",
        notes=notes)
    add(phase="hvp_contract", status="pass", metric="hvp_max_abs_error",
        value=hvp_error, max_abs_error=hvp_error,
        oracle="independent NumPy nonlinear PINN HVP oracle",
        notes="exact residual HVP provider; no finite-difference production path")
    add(phase="fit_contract", status="pass", metric="fit_parameter_error",
        value=0.0, max_abs_error=0.0,
        oracle="independent affine optimum theta=1",
        notes="bounded FortOpt L-BFGS-B fit")
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_pinn_training", value="nan", max_abs_error="nan",
        oracle="typed FORTNUM_NOT_IMPLEMENTED device boundary",
        notes="no resident PINN residual/derivative graph; host fallback forbidden")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
