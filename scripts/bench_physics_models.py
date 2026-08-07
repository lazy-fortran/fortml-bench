#!/usr/bin/env python3
"""Correctness-gated evidence for the current Hamiltonian MLP prototype.

The NumPy row is an independent velocity-Verlet harmonic-oscillator oracle.
The FortML row runs the existing finite-difference, adjoint, symplectic-form,
and reversibility test.  It is deliberately not an end-to-end PINN or GPU
benchmark: no resident physics graph is currently exposed by FortML.
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


STEP = 0.07
FIELDS = (
    "workload", "phase", "backend", "device", "status", "steps",
    "step_size", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
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
        "workload": "physics_models", "phase": "structure", "backend": "",
        "device": "cpu", "status": "", "steps": 1, "step_size": STEP,
        "seconds_per_operation": "", "metric": "", "value": "",
        "max_abs_error": "", "oracle": "", "notes": "",
    })
    row.update(values)
    return row


def harmonic_oracle() -> tuple[float, float]:
    """Check the exact 2-D harmonic velocity-Verlet map independently."""
    h = STEP
    map_forward = np.array([
        [1.0 - 0.5 * h * h, h],
        [-h * (1.0 - 0.25 * h * h), 1.0 - 0.5 * h * h],
    ])
    map_reverse = np.array([
        [1.0 - 0.5 * h * h, -h],
        [h * (1.0 - 0.25 * h * h), 1.0 - 0.5 * h * h],
    ])
    omega = np.array([[0.0, -1.0], [1.0, 0.0]])
    symplectic_defect = float(np.max(np.abs(map_forward.T @ omega @ map_forward - omega)))
    reversibility = float(np.max(np.abs(map_reverse @ map_forward - np.eye(2))))
    if symplectic_defect > 1.0e-14 or reversibility > 1.0e-14:
        raise RuntimeError("independent harmonic symplectic oracle failed")
    return symplectic_defect, reversibility


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
    if result.returncode != 0:
        note = output.splitlines()[-1] if output else "FortML test failed"
        return base(details, backend="fortml", status="failed",
                    seconds_per_operation=elapsed, metric="behavioral_test",
                    value=0.0, max_abs_error="", oracle="test_hamiltonian_mlp",
                    notes=note)
    if "PASS" not in output:
        return base(details, backend="fortml", status="failed",
                    seconds_per_operation=elapsed, metric="behavioral_test",
                    value=0.0, oracle="test_hamiltonian_mlp",
                    notes="test emitted no PASS marker")
    return base(details, backend="fortml", status="pass",
                seconds_per_operation=elapsed, metric="behavioral_test",
                value=1.0, max_abs_error=0.0,
                oracle=("finite-difference, adjoint, symplectic-form, and "
                        "reversibility checks in test_hamiltonian_mlp"),
                notes="CPU prototype only; not an end-to-end training timing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/physics_models.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    fortml = args.fortml.resolve()
    details = metadata(root, fortml, output)
    defect, reversibility = harmonic_oracle()
    rows = [
        base(details, backend="numpy_oracle", status="pass", device="cpu",
             metric="symplectic_defect", value=defect, max_abs_error=defect,
             oracle="independent harmonic-oscillator velocity-Verlet map",
             notes=f"reversibility_error={reversibility:.3e}"),
    ]
    if args.skip_fortml:
        rows.append(base(details, backend="fortml", status="skipped",
                         metric="behavioral_test", oracle="test_hamiltonian_mlp",
                         notes="--skip-fortml"))
    else:
        rows.append(run_fortml(fortml, details))
    rows.append(base(details, backend="fortml", device="cuda", status="unavailable",
                     metric="physics_graph", oracle="resident device contract",
                     notes=("no resident Hamiltonian/PINN graph is exposed; "
                            "CUDA/OpenACC end-to-end timing is not claimed")))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
