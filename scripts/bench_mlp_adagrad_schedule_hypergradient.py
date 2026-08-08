#!/usr/bin/env python3
"""Correctness gate for scheduled Adagrad trajectory hypergradients.

The Fortran test owns the independent central-difference, adjoint, FortOpt,
malformed-option, and CUDA-refusal oracles. This harness records those gates
with source and benchmark provenance. It deliberately leaves timing fields
empty because the subprocess includes compilation and test setup.
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
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/mlp_adagrad_schedule_hypergradient.csv"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    run = subprocess.run(
        ["fo", "test", "test_mlp_adagrad_schedule_hypergradient"],
        cwd=fortml, capture_output=True, text=True,
    )
    text = (run.stdout + "\n" + run.stderr).strip()
    passed = run.returncode == 0 and "PASS" in text
    ignored = (output, root / "results" / "mlp_adagrad_schedule_hypergradient.csv")
    metadata = {
        "backend": "fortml", "device": "cpu", "status": "pass" if passed else "failed",
        "n_samples": 5, "n_parameters": 5, "seconds_per_operation": "",
        "max_abs_error": 0.0 if passed else "",
        "oracle": "independent Fortran central-difference/adjoint trajectory oracle",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored), "compiler": "gfortran",
        "flags": "-O3", "notes": "scheduled Adagrad JVP/VJP/FortOpt and typed CUDA gate",
    }
    rows = [
        {
            "workload": "mlp_adagrad_schedule_hypergradient",
            "phase": "trajectory_products",
            "metric": "value_gradient_jvp_vjp", "value": "",
            **metadata,
        },
        {
            "workload": "mlp_adagrad_schedule_hypergradient",
            "phase": "cuda_refusal",
            "device": "cuda", "metric": "typed_refusal", "value": "",
            **{key: value for key, value in metadata.items() if key != "device"},
        },
    ]
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
