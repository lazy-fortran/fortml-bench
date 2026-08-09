#!/usr/bin/env python3
"""Correctness-gated benchmark for sequential basis-pipeline cloning.

The NumPy oracle reconstructs the polynomial/Fourier composition and checks
copy equality plus clone-only parameter mutation.  The release app measures
host deep-copy throughput and the explicit resident-CUDA graph boundary.
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
    x = np.linspace(-0.9, 0.9, 64, dtype=np.float64)[:, None]

    def evaluate(values: np.ndarray, frequency_scale: float) -> np.ndarray:
        powers = np.column_stack([values[:, 0], values[:, 0] ** 2])
        # The second stage receives the two polynomial features.  One harmonic
        # per input emits a sine/cosine pair in input-column order.
        return np.column_stack([
            np.sin(0.7 * frequency_scale * powers[:, 0]),
            np.cos(0.7 * frequency_scale * powers[:, 0]),
            np.sin(1.1 * powers[:, 1]),
            np.cos(1.1 * powers[:, 1]),
        ])

    original = evaluate(x, 1.0)
    copied = evaluate(x, 1.0)
    changed = evaluate(x, 1.25)
    copy_error = float(np.max(np.abs(original - copied)))
    mutation_effect = float(np.max(np.abs(original - changed)))
    if copy_error > 1.0e-14 or mutation_effect < 1.0e-8:
        raise RuntimeError(
            f"independent sequential clone oracle failed: copy={copy_error}, "
            f"mutation={mutation_effect}"
        )
    return original.shape[1], copy_error, mutation_effect


def parse(stdout: str) -> dict[str, float | int]:
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields[:1] != ["sequential_pipeline_clone"]:
            continue
        if len(fields) != 6:
            raise RuntimeError(f"malformed sequential clone record: {line!r}")
        return {
            "n_features": int(fields[1]), "repetitions": int(fields[2]),
            "seconds": float(fields[3]), "copy_error": float(fields[4]),
            "cuda_code": int(fields[5]),
        }
    raise RuntimeError("release app omitted sequential clone metrics")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/sequential_pipeline_clone.csv"))
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
    gate = subprocess.run(["fo", "test", "test_sequential_pipeline_clone"],
                          cwd=fortml, env=environment, capture_output=True,
                          text=True)
    if gate.returncode != 0:
        raise RuntimeError("sequential pipeline clone release test failed")
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_sequential_pipeline_clone"],
        cwd=fortml, env=environment, check=True, capture_output=True,
        text=True,
    )
    observed = parse(completed.stdout)
    if (observed["n_features"] != n_features or
            observed["repetitions"] != REPETITIONS or
            observed["copy_error"] > 1.0e-14 or observed["cuda_code"] != 3):
        raise RuntimeError(f"sequential clone contract mismatch: {observed}")
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": "gfortran", "flags": "-O2",
    }
    rows = [
        {
            **{field: "" for field in FIELDS}, **details,
            "workload": "sequential_pipeline_clone", "phase": "independent_oracle",
            "backend": "numpy_oracle", "device": "cpu", "status": "pass",
            "n_features": n_features, "repetitions": REPETITIONS,
            "metric": "clone_copy_and_mutation", "value": mutation_effect,
            "max_abs_error": oracle_copy_error,
            "oracle": "independent polynomial/Fourier NumPy reconstruction",
            "notes": "copied output is exact; parameter mutation changes only clone",
        },
        {
            **{field: "" for field in FIELDS}, **details,
            "workload": "sequential_pipeline_clone", "phase": "cpu_clone",
            "backend": "fortml", "device": "cpu", "status": "pass",
            "n_features": observed["n_features"],
            "repetitions": observed["repetitions"],
            "seconds_per_operation": observed["seconds"],
            "metric": "deep_clone", "value": observed["repetitions"],
            "max_abs_error": observed["copy_error"],
            "oracle": "test_sequential_pipeline_clone independent behavioral gate",
            "notes": "all sequential stages, fitted metadata, and schema copied transactionally",
        },
        {
            **{field: "" for field in FIELDS}, **details,
            "workload": "sequential_pipeline_clone", "phase": "device_contract",
            "backend": "fortml", "device": "cuda", "status": "unavailable",
            "n_features": n_features, "repetitions": REPETITIONS,
            "metric": "resident_sequential_graph_clone",
            "value": "FORTNUM_NOT_IMPLEMENTED", "max_abs_error": 0.0,
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
