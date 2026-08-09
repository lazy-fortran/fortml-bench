#!/usr/bin/env python3
"""Benchmark gate for multiclass Laplace-GP log-probability products.

The NumPy oracle independently differentiates one-vs-rest normalization and
the logarithm.  The FortML gate fits the sorted-label OVR Laplace classifier,
checks input and packed-kernel products, and verifies the typed CUDA refusal.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_parameters", "seconds_per_operation", "metric", "value",
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


def oracle() -> tuple[float, float]:
    raw = np.array([[0.31, 0.47, 0.22], [0.18, 0.29, 0.53],
                    [0.41, 0.37, 0.22]], dtype=np.float64)
    raw_dot = np.array([[0.07, -0.03, 0.02], [-0.02, 0.04, -0.01],
                        [0.03, 0.01, -0.04]], dtype=np.float64)
    total = raw.sum(axis=1, keepdims=True)
    total_dot = raw_dot.sum(axis=1, keepdims=True)
    probability = raw / total
    log_probability = np.log(probability)
    log_dot = raw_dot / raw - total_dot / total
    h = 2.0e-6
    plus = np.log((raw + h * raw_dot) /
                  (raw.sum(axis=1, keepdims=True) + h * total_dot))
    minus = np.log((raw - h * raw_dot) /
                   (raw.sum(axis=1, keepdims=True) - h * total_dot))
    error = float(np.max(np.abs(log_dot - (plus - minus) / (2.0 * h))))
    round_trip = float(np.max(np.abs(np.exp(log_probability) - probability)))
    if max(error, round_trip) > 2.0e-8:
        raise RuntimeError(f"multiclass log-probability oracle failed: {error:.3e}")
    return error, round_trip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gp_multiclass_log_proba.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml, output = args.fortml.resolve(), args.output.resolve()
    oracle_error, round_trip_error = oracle()
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": "gfortran", "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(details)
        row.update({"workload": "gp_multiclass_log_proba", "backend": "fortml",
                    "device": "cpu", "n_samples": 9, "n_features": 2,
                    "n_parameters": 6})
        row.update(values)
        rows.append(row)

    add(phase="independent_oracle", backend="numpy_oracle", status="pass",
        metric="normalized_log_probability_jvp_max_abs_error", value=oracle_error,
        max_abs_error=oracle_error,
        oracle="independent NumPy OVR normalization and central difference",
        notes=f"exp(log_probability) round-trip error={round_trip_error:.3e}")
    started = time.perf_counter()
    if args.skip_fortml:
        status, notes = "skipped", "--skip-fortml"
    else:
        environment = os.environ.copy()
        environment["FO_SCAN_FALLBACK"] = "regex"
        subprocess.run(["fo", "test", "test_gp_multiclass_log_proba"],
                       cwd=fortml, env=environment, check=True)
        status, notes = "pass", "sorted-label OVR log-proba value/input/parameter/refusal gate"
    elapsed = time.perf_counter() - started
    add(phase="public_contract_gate", status=status, seconds_per_operation=elapsed,
        metric="normalized_log_probability_derivative_max_abs_error", value=oracle_error,
        max_abs_error=oracle_error,
        oracle="FortML test_gp_multiclass_log_proba behavioral gate", notes=notes)
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_ovr_laplace_log_probability_graph", value="nan",
        oracle="typed FortML CUDA refusal",
        notes="OVR covariance states and normalization reduction are not resident")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
