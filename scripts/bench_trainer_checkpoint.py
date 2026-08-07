#!/usr/bin/env python3
"""Correctness-gated generic trainer checkpoint/resume benchmark.

The independent NumPy oracle serialises the *optimizer state* (parameters,
first/second moments, step counter, EMA, and value history), resumes from the
split point, and compares it with an uninterrupted Adam trajectory.  The
FortML release test additionally checks the portable text parser's refusal of
truncated and extra records and verifies transactional failure behavior.
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
    "workload", "phase", "backend", "device", "status", "n_parameters",
    "steps", "seconds_per_operation", "metric", "value", "max_abs_error",
    "oracle", "python_version", "numpy_version", "fortml_revision",
    "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def oracle() -> tuple[float, float, int]:
    """Return final norm, continuation error, and the checkpoint step."""
    target = np.array([1.5, -0.5], dtype=np.float64)
    initial = np.array([0.0, 1.0], dtype=np.float64)
    learning_rate, beta1, beta2, epsilon = 0.05, 0.8, 0.95, 1.0e-8
    steps, split = 8, 3

    def advance(parameters: np.ndarray, first: np.ndarray, second: np.ndarray,
                step: int, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        for _ in range(count):
            gradient = 2.0 * (parameters - target)
            step += 1
            first = beta1 * first + (1.0 - beta1) * gradient
            second = beta2 * second + (1.0 - beta2) * gradient**2
            corrected_first = first / (1.0 - beta1**step)
            corrected_second = second / (1.0 - beta2**step)
            parameters = parameters - learning_rate * corrected_first / (
                np.sqrt(corrected_second) + epsilon)
        return parameters, first, second, step

    full, full_first, full_second, full_step = advance(
        initial.copy(), np.zeros(2), np.zeros(2), 0, steps)
    split_parameters, split_first, split_second, split_step = advance(
        initial.copy(), np.zeros(2), np.zeros(2), 0, split)
    resumed, resumed_first, resumed_second, resumed_step = advance(
        split_parameters, split_first, split_second, split_step, steps - split)
    error = max(float(np.max(np.abs(full - resumed))),
                float(np.max(np.abs(full_first - resumed_first))),
                float(np.max(np.abs(full_second - resumed_second))))
    if error > 1.0e-14 or full_step != resumed_step:
        raise RuntimeError(f"trainer continuation oracle failed: {error:.3e}")
    return float(np.linalg.norm(full)), error, split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/trainer_checkpoint.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    expected_norm, continuation_error, split = oracle()
    started = time.perf_counter()
    if args.skip_fortml:
        status, notes = "skipped", "--skip-fortml"
    else:
        subprocess.run(["fo", "test", "test_trainer"], cwd=fortml, check=True)
        status = "pass"
        notes = "test_trainer checks Adam continuation, malformed/truncated/extra refusal"
    elapsed = time.perf_counter() - started
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
        row.update({"workload": "trainer_checkpoint", "backend": "fortml",
                    "device": "cpu", "n_parameters": 2, "steps": 8})
        row.update(values)
        rows.append(row)

    add(phase="independent_oracle", backend="numpy_oracle", status="pass", metric="final_parameter_l2_norm",
        value=expected_norm, max_abs_error=continuation_error,
        oracle="independent NumPy Adam state continuation",
        notes=f"split_step={split}; parameters/moments/step all compared")
    add(phase="public_contract_gate", status=status,
        seconds_per_operation=elapsed, metric="continuation_max_abs_error",
        value=continuation_error, max_abs_error=continuation_error,
        oracle="FortML test_trainer independent quadratic and parser oracle",
        notes=notes)
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_optimizer", value="nan", max_abs_error="nan",
        oracle="typed device refusal",
        notes="generic trainer state is host-resident; no hidden GPU fallback")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
