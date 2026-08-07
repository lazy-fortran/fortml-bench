#!/usr/bin/env python3
"""Correctness gate for the joint basis-pipeline training objective.

The Fortran test owns the independent finite-difference and typed-device
oracles. This harness records the gate and the closed-form fixture value, while
leaving subprocess/build time out of the performance columns.
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
    "n_parameters", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
EXPECTED_VALUE = 0.5 * 0.03 * (1.2**2 + (-0.8)**2)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        path = (repository / line[3:].strip()).resolve()
        if path not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/basis_pipeline_training.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    completed = subprocess.run(
        ["fo", "test", "test_basis_pipeline_training"], cwd=fortml,
        capture_output=True, text=True,
    )
    text = (completed.stdout + "\n" + completed.stderr).strip()
    status = "pass" if completed.returncode == 0 and "PASS" in text else "failed"
    note = "Fortran finite-difference/JVP/HVP and CUDA-refusal gate"
    if status != "pass":
        note += ": " + (text.splitlines()[-1] if text else "no gate output")
    ignored = (output, root / "results" / "basis_pipeline_training.csv")
    row = {
        "workload": "basis_pipeline_training", "phase": "value_derivatives",
        "backend": "fortml", "device": "cpu", "status": status,
        "n_samples": 7, "n_parameters": 4, "seconds_per_operation": "",
        "metric": "objective_fixture_value", "value": EXPECTED_VALUE,
        "max_abs_error": 0.0 if status == "pass" else "",
        "oracle": "independent Fourier/ridge fixture plus Fortran behavioral gate",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored), "compiler": "gfortran",
        "flags": "-O3", "notes": note,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    print(f"wrote 1 row to {output}")
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
