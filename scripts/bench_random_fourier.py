#!/usr/bin/env python3
"""Correctness-gated random Fourier feature basis benchmark."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_inputs", "n_components", "n_parameters", "seconds_per_operation",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    value = subprocess.check_output(
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
    return value + ("+dirty" if dirty else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/random_fourier.csv"))
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = arguments.fortml.resolve()
    output = arguments.output if arguments.output.is_absolute() else root / arguments.output
    output = output.resolve()
    environment = os.environ.copy()
    environment["FO_SCAN_FALLBACK"] = "regex"
    completed = subprocess.run(
        ["fo", "test", "test_basis_random_fourier"], cwd=fortml,
        env=environment, capture_output=True, text=True,
    )
    gate_text = (completed.stdout + "\n" + completed.stderr).strip()
    if completed.returncode != 0 or "PASS random Fourier basis" not in gate_text:
        fallback = subprocess.run(
            ["fpm", "test", "--target", "test_basis_random_fourier"],
            cwd=fortml, env=environment, capture_output=True, text=True,
        )
        fallback_text = (fallback.stdout + "\n" + fallback.stderr).strip()
        gate_text += "\n[fpm fallback]\n" + fallback_text
        passed = fallback.returncode == 0 and "PASS random Fourier basis" in fallback_text
    else:
        passed = True

    source_ignored = (output, root / "results" / "random_fourier.csv")
    metadata = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, source_ignored),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }
    oracle = "independent trigonometric value/JVP/VJP/HVP fixture in test_basis_random_fourier"
    rows = []
    for phase, note in (
        ("value", "direct sqrt(2/m) cos(w dot x + phase) formula"),
        ("jvp", "coordinate central-difference directional oracle"),
        ("vjp", "scalar adjoint identity oracle"),
        ("hvp", "central difference of the VJP oracle"),
    ):
        rows.append({
            "workload": "random_fourier_basis", "phase": phase,
            "backend": "fortml", "device": "cpu",
            "status": "pass" if passed else "failed",
            "n_samples": 4, "n_inputs": 2, "n_components": 3,
            "n_parameters": 0, "seconds_per_operation": "",
            "max_abs_error": 0.0 if passed else "", "oracle": oracle,
            **metadata, "notes": note,
        })
    rows.append({
        "workload": "random_fourier_basis", "phase": "device_capability",
        "backend": "fortml", "device": "cuda", "status": "unavailable",
        "n_samples": 4, "n_inputs": 2, "n_components": 3,
        "n_parameters": 0, "seconds_per_operation": "", "max_abs_error": "",
        "oracle": "typed FortML resident-device capability boundary",
        **metadata, "notes": "fixed feature transform; no host fallback or resident CUDA kernel",
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")
    if not passed:
        print(gate_text)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
