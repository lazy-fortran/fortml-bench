#!/usr/bin/env python3
"""Correctness-gated Hamiltonian vector-field VJP evidence.

The NumPy row is an independent quadratic Hamiltonian oracle.  The FortML
row executes the release app, which checks the canonical field VJP/JVP adjoint
identity for both packed parameters and the input state.  The CUDA row remains
an explicit capability boundary because no resident HNN derivative graph is
currently exposed.
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
    "parameter_count", "seconds_per_operation", "metric", "value",
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
        "workload": "hamiltonian_vector_field_vjp", "phase": "",
        "backend": "", "device": "cpu", "status": "",
        "n_coordinates": 1, "parameter_count": 20,
        "seconds_per_operation": "", "metric": "", "value": "",
        "max_abs_error": "", "oracle": "", "notes": "",
    })
    row.update(values)
    return row


def independent_oracle() -> tuple[float, float]:
    """Check the canonical VJP/JVP identity for a quadratic H(q,p)."""
    state = np.array([0.29, -0.34], dtype=np.float64)
    direction = np.array([-0.08, 0.12], dtype=np.float64)
    field_bar = np.array([0.61, -0.37], dtype=np.float64)
    hessian = np.array([[1.3, 0.2], [0.2, 0.9]], dtype=np.float64)
    canonical = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
    field = canonical @ (hessian @ state)
    field_dot = canonical @ (hessian @ direction)
    state_bar = (canonical @ hessian).T @ field_bar
    lhs = float(field_bar @ field_dot)
    rhs = float(state_bar @ direction)
    adjoint_error = abs(lhs - rhs)
    step = 1.0e-6
    def vector(point: np.ndarray) -> np.ndarray:
        return canonical @ (hessian @ point)
    finite_error = float(np.max(np.abs(
        field_dot - (vector(state + step * direction) -
                     vector(state - step * direction)) / (2.0 * step))))
    if max(adjoint_error, finite_error) > 3.0e-10:
        raise RuntimeError("independent Hamiltonian VJP oracle failed")
    return adjoint_error, finite_error


def run_fortml(fortml: Path, details: dict[str, str]) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.setdefault("FO_SCAN_FALLBACK", "regex")
    started = time.perf_counter()
    result = subprocess.run(
        ["fo", "exec", "fortml_bench_hamiltonian_vector_field_vjp"],
        cwd=fortml, env=environment, capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - started
    output = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode != 0 or "hamiltonian_vector_field_vjp" not in output:
        note = output.splitlines()[-1] if output else "FortML release app failed"
        return base(details, phase="release_app", backend="fortml", status="failed",
                    seconds_per_operation=elapsed, metric="adjoint_error",
                    value="nan", max_abs_error="nan",
                    oracle="fortml_bench_hamiltonian_vector_field_vjp", notes=note)
    csv_lines = [line for line in output.splitlines()
                 if line.startswith("hamiltonian_vector_field_vjp,")]
    if not csv_lines:
        return base(details, phase="release_app", backend="fortml", status="failed",
                    seconds_per_operation=elapsed, metric="adjoint_error",
                    value="nan", max_abs_error="nan",
                    oracle="fortml_bench_hamiltonian_vector_field_vjp",
                    notes="release app emitted no CSV data row")
    fields = csv_lines[-1].split(",")
    error = float(fields[2])
    return base(details, phase="release_app", backend="fortml", status="pass",
                seconds_per_operation=elapsed, metric="adjoint_error",
                value=error, max_abs_error=error,
                oracle="release HNN vector-field VJP/JVP adjoint identity",
                notes="separable HNN parameter/state reverse products")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/hamiltonian_vector_field_vjp.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": os.environ.get("FFLAGS", "-O3"),
    }
    oracle_adjoint, oracle_finite = independent_oracle()
    rows = [
        base(details, phase="independent_oracle", backend="numpy_oracle",
             status="pass", metric="adjoint_error", value=oracle_adjoint,
             max_abs_error=max(oracle_adjoint, oracle_finite),
             oracle="independent quadratic H(q,p) canonical VJP/JVP oracle",
             notes=f"finite_difference_error={oracle_finite:.3e}"),
    ]
    if args.skip_fortml:
        rows.append(base(details, phase="release_app", backend="fortml",
                         status="skipped", metric="adjoint_error",
                         oracle="fortml_bench_hamiltonian_vector_field_vjp",
                         notes="--skip-fortml"))
    else:
        rows.append(run_fortml(fortml, details))
    rows.append(base(details, phase="device_contract", backend="fortml",
                     device="cuda", status="unavailable",
                     metric="resident_hnn_derivative_graph", value="nan",
                     max_abs_error="nan",
                     oracle="typed FORTNUM_NOT_IMPLEMENTED boundary",
                     notes="no host fallback or resident CUDA HNN graph is claimed"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
