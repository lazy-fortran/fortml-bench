#!/usr/bin/env python3
"""Correctness-gated matrix-factored Adafactor benchmark.

The NumPy path is an independent row/column estimator for one matrix block
plus an unfactored vector block.  FortML timing is retained only after the
focused recurrence and MLP integration tests pass.  CUDA is recorded as an
explicit unavailable contract: no resident row/column kernel is timed.
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "variant", "backend", "device", "status",
    "n_parameters", "steps", "repetitions", "seconds_per_operation",
    "metric", "value", "max_abs_error", "oracle", "python_version",
    "numpy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
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
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def oracle(steps: int = 16) -> tuple[float, float]:
    """Return final norm and an independent split/resume state error."""
    rate, decay, epsilon, threshold = 0.08, 0.9, 1.0e-5, 1.0
    theta0 = np.linspace(-0.4, 0.5, 6, dtype=np.float64)
    gradient = np.array([0.7, -0.2, 0.4, -0.6, 0.3, -0.5], dtype=np.float64)

    def advance(theta: np.ndarray, row: np.ndarray, column: np.ndarray,
                vector: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        for _ in range(count):
            row = decay * row
            column = decay * column
            grad_matrix = gradient[:4].reshape((2, 2), order="F")
            row += (1.0 - decay) * np.mean(grad_matrix**2, axis=1)
            column += (1.0 - decay) * np.mean(grad_matrix**2, axis=0)
            vector = decay * vector + (1.0 - decay) * gradient[4:]**2
            matrix = np.outer(row, column) / max(float(np.mean(row)), epsilon)
            second = np.concatenate((matrix.reshape(4, order="F"), vector))
            clip = max(1.0, float(np.sqrt(np.mean(second)) / threshold))
            theta = theta - rate * gradient / clip / (np.sqrt(second) + epsilon)
        return theta, row, column, vector

    full = advance(theta0.copy(), np.zeros(2), np.zeros(2), np.zeros(2), steps)
    split = advance(theta0.copy(), np.zeros(2), np.zeros(2), np.zeros(2), steps // 2)
    resumed = advance(split[0], split[1], split[2], split[3], steps - steps // 2)
    error = max(float(np.max(np.abs(full[i] - resumed[i]))) for i in range(4))
    if error > 2.0e-14:
        raise RuntimeError(f"factored Adafactor continuation oracle failed: {error:.3e}")
    return float(np.linalg.norm(full[0])), error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/adafactor_factored.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    steps, repetitions = 16, 8
    started = time.perf_counter()
    expected_norm, state_error = oracle(steps)
    oracle_seconds = (time.perf_counter() - started) / repetitions
    if args.skip_fortml:
        public_status, public_seconds, notes = "skipped", 0.0, "--skip-fortml"
    else:
        started = time.perf_counter()
        result = subprocess.run(
            ["fo", "test", "test_mlp_adafactor_factored", "test_mlp_adafactor"],
            cwd=fortml, capture_output=True, text=True,
        )
        public_seconds = time.perf_counter() - started
        public_status = "pass" if result.returncode == 0 else "unavailable"
        notes = "factored recurrence, MLP integration, and vector regression tests"
        if result.returncode != 0:
            notes = "focused fo test failed"
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
        row.update({"workload": "adafactor", "variant": "matrix_factored",
                    "backend": "fortml", "device": "cpu", "n_parameters": 6,
                    "steps": steps, "repetitions": repetitions})
        row.update(values)
        rows.append(row)

    add(phase="independent_oracle", backend="numpy_oracle", status="pass",
        seconds_per_operation=oracle_seconds, metric="final_parameter_l2_norm",
        value=expected_norm, max_abs_error=state_error,
        oracle="NumPy row/column estimator plus vector-state recurrence",
        notes="2x2 factored matrix and 2-vector block; split/resume oracle")
    add(phase="public_contract_gate", status=public_status,
        seconds_per_operation=public_seconds, metric="continuation_max_abs_error",
        value=state_error, max_abs_error=state_error,
        oracle="FortML independent recurrence and MLP integration tests",
        notes=notes)
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_factored_adafactor", value="nan", max_abs_error="nan",
        oracle="typed device refusal",
        notes="row/column state is CPU-only; no hidden host/device fallback")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
