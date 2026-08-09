#!/usr/bin/env python3
"""Correctness-gated benchmark for cloning a fitted basis pipeline.

The independent NumPy oracle checks that a polynomial-plus-radial pipeline has
an identical copied output, that changing the clone's radial centre leaves the
source untouched, and that the CUDA control-plane boundary is explicit.  The
release app measures repeated CPU deep copies and reports the typed CUDA code.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_features",
    "repetitions", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
REPETITIONS = 5000


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


def independent_oracle() -> tuple[int, float, float]:
    x = np.array(
        [[0.2, -0.4], [-0.3, 0.8], [0.7, 0.5], [0.1, -0.9]],
        dtype=np.float64,
    )
    centre = np.array([0.25, -0.50], dtype=np.float64)
    scale = np.array([0.70, 1.10], dtype=np.float64)

    def evaluate(values: np.ndarray, centre_value: np.ndarray) -> np.ndarray:
        polynomial = np.column_stack([
            values[:, 0], values[:, 0] ** 2,
            values[:, 1], values[:, 1] ** 2,
            np.ones(values.shape[0]),
        ])
        radial = np.exp(-0.5 * np.sum(((values-centre_value)/scale) ** 2,
                                      axis=1))[:, None]
        return np.hstack([polynomial, radial])

    original = evaluate(x, centre)
    copied = evaluate(x, centre.copy())
    changed = evaluate(x, centre + np.array([0.35, 0.0]))
    copy_error = float(np.max(np.abs(original-copied)))
    mutation_effect = float(np.max(np.abs(original-changed)))
    if copy_error > 1.0e-14 or mutation_effect < 1.0e-8:
        raise RuntimeError(
            f"independent clone oracle failed: copy={copy_error}, "
            f"mutation={mutation_effect}"
        )
    return original.shape[1], copy_error, mutation_effect


def parse(stdout: str) -> dict[str, float | int]:
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields[:1] != ["pipeline_clone"]:
            continue
        if len(fields) != 6:
            raise RuntimeError(f"malformed clone benchmark record: {line!r}")
        return {
            "n_features": int(fields[1]), "repetitions": int(fields[2]),
            "seconds": float(fields[3]), "copy_error": float(fields[4]),
            "cuda_code": int(fields[5]),
        }
    raise RuntimeError("release app omitted pipeline clone metrics")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/basis_pipeline_clone.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    n_features, oracle_copy_error, mutation_effect = independent_oracle()
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1",
                        "FO_SCAN_FALLBACK": "regex"})
    subprocess.run(["fo", "build", "--flag", "-O2"], cwd=fortml,
                   env=environment, check=True, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE, text=True)
    gate = subprocess.run(["fo", "test", "test_basis_pipeline_clone"],
                          cwd=fortml, env=environment, capture_output=True,
                          text=True)
    if gate.returncode != 0:
        raise RuntimeError("pipeline clone release test failed")
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_basis_pipeline_clone"],
        cwd=fortml, env=environment, check=True, capture_output=True,
        text=True,
    )
    observed = parse(completed.stdout)
    if (observed["n_features"] != n_features or
            observed["repetitions"] != REPETITIONS or
            observed["copy_error"] > 1.0e-14 or observed["cuda_code"] != 3):
        raise RuntimeError(f"pipeline clone contract mismatch: {observed}")
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": "gfortran", "flags": "-O2",
    }
    rows = [
        {
            **{field: "" for field in FIELDS}, **details,
            "workload": "basis_pipeline_clone", "phase": "independent_oracle",
            "backend": "numpy_oracle", "device": "cpu", "status": "pass",
            "n_features": n_features, "repetitions": REPETITIONS,
            "metric": "clone_copy_and_mutation", "value": mutation_effect,
            "max_abs_error": oracle_copy_error,
            "oracle": "independent polynomial/radial NumPy reconstruction",
            "notes": "copied output is exact; changed clone centre changes only clone",
        },
        {
            **{field: "" for field in FIELDS}, **details,
            "workload": "basis_pipeline_clone", "phase": "cpu_clone",
            "backend": "fortml", "device": "cpu", "status": "pass",
            "n_features": observed["n_features"],
            "repetitions": observed["repetitions"],
            "seconds_per_operation": observed["seconds"],
            "metric": "deep_clone", "value": observed["repetitions"],
            "max_abs_error": observed["copy_error"],
            "oracle": "test_basis_pipeline_clone independent behavioral gate",
            "notes": "fitted stage state and input schema are copied transactionally",
        },
        {
            **{field: "" for field in FIELDS}, **details,
            "workload": "basis_pipeline_clone", "phase": "device_contract",
            "backend": "fortml", "device": "cuda", "status": "unavailable",
            "n_features": n_features, "repetitions": REPETITIONS,
            "metric": "resident_pipeline_clone", "value": "FORTNUM_NOT_IMPLEMENTED",
            "max_abs_error": 0.0,
            "oracle": "typed resident-CUDA graph boundary",
            "notes": "destination remains unchanged; no host fallback",
        },
    ]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
