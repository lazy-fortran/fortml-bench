#!/usr/bin/env python3
"""Correctness-gated unfactored Adafactor trainer benchmark.

The NumPy recurrence is independent of FortML.  It checks update-RMS clipping,
checkpoint-style state continuation, and the explicit CPU/CUDA boundary before
timing the public FortML trainer/MLP tests.  The flat trainer API has no matrix
layout metadata, so this lane is deliberately named ``unfactored_vector``;
matrix row/column factorization is not inferred from a packed vector.
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


def oracle(n: int = 4096, steps: int = 32, split: int = 13) -> tuple[float, float, float]:
    rate, decay, epsilon, threshold = 0.08, 0.93, 1.0e-6, 1.0
    initial = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    target = np.linspace(0.5, -0.5, n, dtype=np.float64)

    def advance(x: np.ndarray, moment: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(count):
            gradient = x - target
            moment = decay * moment + (1.0 - decay) * gradient**2
            update_rms = np.sqrt(np.mean(moment))
            clip_scale = max(1.0, float(update_rms / threshold))
            x = x - rate * gradient / clip_scale / (np.sqrt(moment) + epsilon)
        return x, moment

    full, full_moment = advance(initial.copy(), np.zeros(n), steps)
    split_x, split_moment = advance(initial.copy(), np.zeros(n), split)
    resumed, resumed_moment = advance(split_x, split_moment, steps - split)
    continuation_error = max(float(np.max(np.abs(full - resumed))),
                             float(np.max(np.abs(full_moment - resumed_moment))))
    if continuation_error > 2.0e-14:
        raise RuntimeError(f"Adafactor continuation oracle failed: {continuation_error:.3e}")
    return float(np.linalg.norm(full)), continuation_error, float(np.mean(full_moment))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/adafactor.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    n, steps, repetitions = 4096, 32, 8
    started = time.perf_counter()
    norm, continuation_error, moment_mean = oracle(n, steps)
    oracle_seconds = (time.perf_counter() - started) / repetitions
    if args.skip_fortml:
        public_status, public_seconds, notes = "skipped", 0.0, "--skip-fortml"
    else:
        started = time.perf_counter()
        result = subprocess.run(
            ["fo", "test", "test_trainer", "test_mlp_adafactor"],
            cwd=fortml, capture_output=True, text=True,
        )
        public_seconds = time.perf_counter() - started
        public_status = "pass" if result.returncode == 0 else "unavailable"
        notes = "test_trainer and test_mlp_adafactor" if result.returncode == 0 else "fo test failed"
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
        row.update({"workload": "adafactor", "variant": "unfactored_vector",
                    "backend": "fortml", "device": "cpu", "n_parameters": n,
                    "steps": steps, "repetitions": repetitions})
        row.update(values)
        rows.append(row)

    add(phase="independent_oracle", backend="numpy_oracle", status="pass",
        seconds_per_operation=oracle_seconds, metric="final_parameter_l2_norm",
        value=norm, max_abs_error=continuation_error,
        oracle="NumPy squared-gradient state continuation",
        notes=f"split={13}; mean_second_moment={moment_mean:.17g}")
    add(phase="public_contract_gate", status=public_status,
        seconds_per_operation=public_seconds, metric="continuation_max_abs_error",
        value=continuation_error, max_abs_error=continuation_error,
        oracle="FortML independent trainer and MLP recurrence/checkpoint tests",
        notes=notes)
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_adafactor", value="nan", max_abs_error="nan",
        oracle="typed device refusal",
        notes="CPU unfactored state only; no hidden host/GPU fallback")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
