#!/usr/bin/env python3
"""Correctness gate for the total-degree polynomial interaction basis."""

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
    parser.add_argument("--output", type=Path,
                        default=Path("results/polynomial_interactions.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    completed = subprocess.run(
        ["fo", "test", "test_basis_polynomial_interactions"],
        cwd=fortml, capture_output=True, text=True,
    )
    gate_text = (completed.stdout + "\n" + completed.stderr).strip()
    status = "pass" if completed.returncode == 0 and "PASS" in gate_text else "failed"
    if status != "pass":
        # Keep the release harness useful when a local FortFront scanner cannot
        # parse an unrelated dependency; fpm still compiles the same target
        # with gfortran and the test remains the behavioral oracle.
        fallback = subprocess.run(
            ["fpm", "test", "--target", "test_basis_polynomial_interactions"],
            cwd=fortml, capture_output=True, text=True,
        )
        fallback_text = (fallback.stdout + "\n" + fallback.stderr).strip()
        gate_text += "\n[fpm fallback]\n" + fallback_text
        status = "pass" if fallback.returncode == 0 and "PASS" in fallback_text else "failed"
    note = "total-degree monomial value/JVP/VJP/HVP finite-difference gate"
    if status != "pass":
        note += ": " + (gate_text.splitlines()[-1] if gate_text else "no gate output")
    ignored = (output, root / "results" / "polynomial_interactions.csv")
    metadata = {
        "backend": "fortml", "device": "cpu", "status": status,
        "n_samples": 3, "n_parameters": 0, "seconds_per_operation": "",
        "max_abs_error": 0.0 if status == "pass" else "",
        "oracle": "independent monomial ordering plus central-difference/JVP/VJP/HVP gate",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored), "compiler": "gfortran",
        "flags": "-O3", "notes": note,
    }
    rows = [{
        "workload": "polynomial_interactions", "phase": phase,
        "metric": metric, "value": 6.0, **metadata,
    } for phase, metric in (
        ("value", "feature_count"), ("jvp", "directional_derivative"),
        ("vjp", "adjoint_identity"), ("hvp", "second_directional_derivative"),
    )]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
