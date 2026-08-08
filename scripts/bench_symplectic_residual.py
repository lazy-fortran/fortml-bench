#!/usr/bin/env python3
"""Correctness-gated canonical symplectic-form residual evidence.

The NumPy rows are an independent harmonic-oscillator velocity-Verlet oracle.
The FortML row runs the Fortran residual, JVP/VJP, value, physics-constraint,
and device-boundary test.  The CUDA row is intentionally unavailable because
the current implementation has no resident symplectic derivative kernel.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_coordinates",
    "step_size", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
STEP = 0.17


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def base(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    row.update(details)
    row.update({
        "workload": "symplectic_residual", "phase": "",
        "backend": "", "device": "cpu", "status": "",
        "n_coordinates": 1, "step_size": STEP,
        "seconds_per_operation": "", "metric": "", "value": "",
        "max_abs_error": "", "oracle": "", "notes": "",
    })
    row.update(values)
    return row


def verlet_map(step: float) -> np.ndarray:
    return np.array([
        [1.0 - 0.5 * step * step, step],
        [-step + 0.25 * step**3, 1.0 - 0.5 * step * step],
    ], dtype=np.float64)


def verlet_map_dot(step: float) -> np.ndarray:
    return np.array([
        [-step, 1.0],
        [-1.0 + 0.75 * step * step, -step],
    ], dtype=np.float64)


def numpy_oracle() -> dict[str, float]:
    """Check value, JVP and VJP identities independently of FortML."""
    jacobian = verlet_map(STEP)
    jacobian_dot = verlet_map_dot(STEP)
    omega = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
    defect = jacobian.T @ omega @ jacobian - omega
    defect_dot = jacobian_dot.T @ omega @ jacobian + jacobian.T @ omega @ jacobian_dot
    epsilon = 1.0e-6
    finite_dot = (verlet_map(STEP + epsilon) - verlet_map(STEP - epsilon)) / (2.0 * epsilon)
    tangent_error = float(np.max(np.abs(jacobian_dot - finite_dot)))
    residual_bar = np.array([[0.4, 0.2], [-0.7, 0.9]], dtype=np.float64)
    jacobian_bar = omega @ jacobian @ residual_bar.T + omega.T @ jacobian @ residual_bar
    direction = jacobian_dot
    adjoint_error = abs(float(np.sum(residual_bar * defect_dot)) -
                        float(np.sum(jacobian_bar * direction)))
    value = float(np.sum(defect * defect) / 8.0)
    value_dot = float(np.sum(defect * defect_dot) / 4.0)
    if max(float(np.max(np.abs(defect))), tangent_error, adjoint_error,
           abs(value), abs(value_dot)) > 3.0e-10:
        raise RuntimeError("independent symplectic residual oracle failed")
    return {
        "defect": float(np.max(np.abs(defect))),
        "tangent_error": tangent_error,
        "adjoint_error": float(adjoint_error),
        "value": value,
        "value_dot": value_dot,
    }


def run_fortml(fortml: Path, details: dict[str, str]) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.setdefault("FO_SCAN_FALLBACK", "regex")
    started = time.perf_counter()
    result = subprocess.run(
        ["fo", "test", "test_symplectic"], cwd=fortml, env=environment,
        capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - started
    output = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode != 0:
        note = output.splitlines()[-1] if output else "FortML test failed"
        return base(details, phase="public_contract_gate", backend="fortml",
                    status="failed", seconds_per_operation=elapsed,
                    metric="behavioral_test", value=0.0,
                    oracle="test_symplectic", notes=note)
    return base(details, phase="public_contract_gate", backend="fortml",
                status="pass", seconds_per_operation=elapsed,
                metric="behavioral_test", value=1.0, max_abs_error=0.0,
                oracle="independent harmonic-oscillator Verlet residual test",
                notes="residual/value JVP-VJP, physics bridge, and CUDA refusal")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/symplectic_residual.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    fortml = args.fortml.resolve()
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": os.environ.get("FFLAGS", "-O3"),
    }
    oracle = numpy_oracle()
    rows = [
        base(details, phase="independent_oracle", backend="numpy_oracle",
             status="pass", metric="symplectic_defect", value=oracle["defect"],
             max_abs_error=max(oracle["defect"], oracle["tangent_error"],
                               oracle["adjoint_error"]),
             oracle="independent harmonic-oscillator velocity-Verlet map",
             notes=(f"tangent_error={oracle['tangent_error']:.3e}; "
                    f"adjoint_error={oracle['adjoint_error']:.3e}; "
                    f"value_dot={oracle['value_dot']:.3e}")),
    ]
    if args.skip_fortml:
        rows.append(base(details, phase="public_contract_gate", backend="fortml",
                         status="skipped", metric="behavioral_test",
                         oracle="test_symplectic", notes="--skip-fortml"))
    else:
        rows.append(run_fortml(fortml, details))
    rows.append(base(details, phase="device_contract", backend="fortml",
                     device="cuda", status="unavailable",
                     metric="resident_symplectic_derivative_graph",
                     oracle="typed FORTNUM_NOT_IMPLEMENTED boundary",
                     notes="no host fallback or resident CUDA kernel is claimed"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
