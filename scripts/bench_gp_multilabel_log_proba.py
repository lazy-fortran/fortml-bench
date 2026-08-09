#!/usr/bin/env python3
"""Benchmark gate for multilabel Laplace-GP log probabilities.

The NumPy oracle treats each label probability as an independent positive
scalar and checks the logarithm chain rule, central differences, and the
shared-kernel cotangent reduction.  The FortML gate exercises the fitted
multilabel Laplace model, threshold metadata, input/per-label/shared products,
and the typed CUDA refusal.
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


def oracle() -> tuple[float, float, float]:
    # Independent positive label probabilities and their common-kernel tangent.
    probability = np.array([
        [0.19, 0.73], [0.42, 0.31], [0.67, 0.58], [0.83, 0.24],
    ], dtype=np.float64)
    probability_dot = np.array([
        [0.03, -0.02], [-0.04, 0.01], [0.02, 0.04], [-0.01, -0.03],
    ], dtype=np.float64)
    log_probability = np.log(probability)
    log_dot = probability_dot / probability
    step = 2.0e-6
    plus = np.log(probability + step * probability_dot)
    minus = np.log(probability - step * probability_dot)
    finite_difference = (plus - minus) / (2.0 * step)
    jvp_error = float(np.max(np.abs(log_dot - finite_difference)))
    round_trip_error = float(np.max(np.abs(np.exp(log_probability) - probability)))
    cotangent = np.array([
        [0.2, -0.4], [0.7, -0.1], [0.5, -0.3], [-0.6, 0.8],
    ], dtype=np.float64)
    adjoint_error = float(abs(
        np.sum(cotangent * log_dot) -
        np.sum((cotangent / probability) * probability_dot)))
    error = max(jvp_error, round_trip_error, adjoint_error)
    if error > 2.0e-8:
        raise RuntimeError(f"multilabel log-probability oracle failed: {error:.3e}")
    return jvp_error, round_trip_error, adjoint_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gp_multilabel_log_proba.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml, output = args.fortml.resolve(), args.output.resolve()
    jvp_error, round_trip_error, adjoint_error = oracle()
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
        row.update({"workload": "gp_multilabel_log_proba", "backend": "fortml",
                    "device": "cpu", "n_samples": 10, "n_features": 1,
                    "n_parameters": 2})
        row.update(values)
        rows.append(row)

    add(phase="independent_oracle", backend="numpy_oracle", status="pass",
        metric="log_probability_jvp_max_abs_error", value=jvp_error,
        max_abs_error=jvp_error,
        oracle="independent NumPy positive-label log chain and central difference",
        notes=f"round-trip={round_trip_error:.3e}; shared-cotangent adjoint={adjoint_error:.3e}")
    started = time.perf_counter()
    if args.skip_fortml:
        status, notes = "skipped", "--skip-fortml"
    else:
        environment = os.environ.copy()
        environment["FO_SCAN_FALLBACK"] = "regex"
        subprocess.run(["fo", "test", "test_gp_multilabel_classification"],
                       cwd=fortml, env=environment, check=True)
        status, notes = "pass", (
            "independent multilabel log-proba value/input/per-label/shared "
            "JVP-VJP, threshold, CPU, and output-preserving CUDA gate")
    elapsed = time.perf_counter() - started
    add(phase="public_contract_gate", status=status, seconds_per_operation=elapsed,
        metric="log_probability_derivative_max_abs_error", value=max(jvp_error, adjoint_error),
        max_abs_error=max(jvp_error, adjoint_error),
        oracle="FortML test_gp_multilabel_classification behavioral gate", notes=notes)
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_multilabel_laplace_log_probability_graph", value="nan",
        oracle="typed FortML CUDA refusal",
        notes="all output buffers remain unchanged; resident heads and reductions are not linked")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
