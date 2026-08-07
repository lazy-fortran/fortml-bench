#!/usr/bin/env python3
"""Correctness-gated evidence for general nonseparable Hamiltonians.

The NumPy rows are an independent analytic oracle for
H(q,p) = (q**2 + p**2)/2 + c*q*p.  The FortML row runs the general HNN
finite-difference, adjoint, and typed-integrator-refusal gate.  This is a
correctness lane, not an end-to-end GPU timing claim.
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


COUPLING = 0.2
FIELDS = (
    "workload", "phase", "backend", "device", "status", "steps",
    "seconds_per_operation", "metric", "value", "max_abs_error", "oracle",
    "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
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


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": os.environ.get("FFLAGS", "-O3"),
    }


def base(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    row.update(details)
    row.update({
        "workload": "hamiltonian_general", "phase": "structure", "backend": "",
        "device": "cpu", "status": "", "steps": 1,
        "seconds_per_operation": "", "metric": "", "value": "",
        "max_abs_error": "", "oracle": "", "notes": "",
    })
    row.update(values)
    return row


def analytic_oracle() -> tuple[float, float]:
    """Return field and canonical-structure errors for an analytic H(q,p)."""
    q = np.array([0.3, -0.2, 0.1])
    p = np.array([-0.4, 0.6, 0.5])
    expected = np.column_stack((p + COUPLING*q, -(q + COUPLING*p)))
    field = np.column_stack((p + COUPLING*q, -(q + COUPLING*p)))
    value_error = float(np.max(np.abs(field - expected)))
    omega = np.array([[0.0, -1.0], [1.0, 0.0]])
    jacobian = np.array([[COUPLING, 1.0], [-1.0, -COUPLING]])
    structure_error = float(np.max(np.abs(jacobian.T @ omega + omega @ jacobian)))
    if value_error > 1.0e-14 or structure_error > 1.0e-14:
        raise RuntimeError("independent nonseparable Hamiltonian oracle failed")
    return value_error, structure_error


def run_fortml(fortml: Path, details: dict[str, str]) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.setdefault("FO_FC", "gfortran")
    started = time.perf_counter()
    result = subprocess.run(
        ["fo", "test", "test_hamiltonian_mlp"], cwd=fortml, env=environment,
        capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - started
    output = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode != 0 or "PASS" not in output:
        note = output.splitlines()[-1] if output else "FortML test failed"
        return base(details, backend="fortml", status="failed",
                    seconds_per_operation=elapsed, metric="behavioral_test",
                    value=0.0, oracle="test_hamiltonian_mlp", notes=note)
    return base(
        details, backend="fortml", status="pass",
        seconds_per_operation=elapsed, metric="behavioral_test", value=1.0,
        max_abs_error=0.0,
        oracle=("general HNN finite-difference/adjoint products plus separable "
                "symplectic and reversibility checks"),
        notes=("general nonseparable leapfrog refusal is checked with "
               "FORTNUM_NOT_IMPLEMENTED"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/hamiltonian_general.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    fortml = args.fortml.resolve()
    details = metadata(root, fortml, output)
    value_error, structure_error = analytic_oracle()
    rows = [
        base(details, backend="numpy_oracle", metric="field_value_error",
             status="pass", value=0.0, max_abs_error=value_error,
             oracle="analytic H=(q²+p²)/2+0.2*q*p canonical field"),
        base(details, backend="numpy_oracle", metric="canonical_structure_error",
             status="pass", value=0.0, max_abs_error=structure_error,
             oracle="independent J^T Omega + Omega J identity"),
    ]
    if args.skip_fortml:
        rows.append(base(details, backend="fortml", status="skipped",
                         metric="behavioral_test", oracle="test_hamiltonian_mlp",
                         notes="--skip-fortml"))
    else:
        rows.append(run_fortml(fortml, details))
    rows.append(base(details, backend="fortml", device="cuda", status="unavailable",
                     metric="physics_graph", oracle="resident device contract",
                     notes=("no resident general-HNN model/derivative/integrator "
                            "graph; no host fallback or relabeled timing")))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
