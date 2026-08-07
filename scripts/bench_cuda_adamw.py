#!/usr/bin/env python3
"""Correctness gate for FortML's resident native-CUDA AdamW state kernel.

The CUDA test owns device allocation and launch.  This harness independently
reconstructs the exact AdamW recurrence in NumPy, then accepts the native gate
only when its reported maximum error and step count pass the same contract.
The gate intentionally exports no kernel timing: its subprocess duration
includes compilation and is not a valid resident-step measurement.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


N_PARAMETERS = 5
STEPS = 7
LEARNING_RATE = 0.035
BETA1 = 0.81
BETA2 = 0.93
EPSILON = 1.0e-6
WEIGHT_DECAY = 0.17
ORACLE_TOLERANCE = 3.0e-13

FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_parameters",
    "steps", "repetitions", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("NVCC", "nvcc"),
        "flags": os.environ.get("NVCCFLAGS", "-O3 -arch=native"),
    }


def base(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = {field: "" for field in FIELDS}
    row.update(details)
    row.update({
        "workload": "cuda_adamw_state", "phase": "optimizer_step",
        "n_parameters": N_PARAMETERS, "steps": STEPS,
        "oracle": "independent NumPy AdamW recurrence",
    })
    row.update(values)
    return row


def adamw_oracle() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return final state, checksum, and an independently timed norm."""
    initial_parameters = np.array([0.2, -0.1, 0.3, -0.25, 0.05], dtype=np.float64)

    def advance() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        parameters = initial_parameters.copy()
        first = np.zeros(N_PARAMETERS, dtype=np.float64)
        second = np.zeros(N_PARAMETERS, dtype=np.float64)
        for step in range(STEPS):
            gradient = parameters - 0.07 * np.arange(1, N_PARAMETERS + 1)
            gradient += 0.01 * step
            first = BETA1 * first + (1.0 - BETA1) * gradient
            second = BETA2 * second + (1.0 - BETA2) * gradient**2
            bias1 = 1.0 - BETA1 ** (step + 1)
            bias2 = 1.0 - BETA2 ** (step + 1)
            parameters = ((1.0 - LEARNING_RATE * WEIGHT_DECAY) * parameters -
                          LEARNING_RATE * (first / bias1) /
                          (np.sqrt(second / bias2) + EPSILON))
        return parameters, first, second

    started = time.perf_counter()
    states = [advance() for _ in range(32)]
    seconds = (time.perf_counter() - started) / len(states)
    parameters, first, second = states[-1]
    checksum = float(np.sum(parameters) + np.sum(first) + np.sum(second))
    if not all(np.all(np.isfinite(state)) for state in states[-1]):
        raise RuntimeError("AdamW NumPy oracle produced nonfinite state")
    return parameters, first, second, seconds, checksum


def run_gate(fortml: Path) -> tuple[str, str, float | None, float | None]:
    script = fortml / "test" / "run_cuda_adamw_state.sh"
    if not script.is_file():
        return "unavailable", f"test script is absent: {script.name}", None, None
    started = time.perf_counter()
    process = subprocess.run([str(script)], cwd=fortml, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    output = (process.stdout + "\n" + process.stderr).strip()
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    lowered = output.lower()
    if process.returncode != 0:
        return "failed", lines[-1] if lines else "CUDA AdamW gate returned nonzero status", None, elapsed
    if "skipped" in lowered or "unavailable" in lowered:
        notes = next((line for line in lines if "skip" in line.lower() or
                      "unavailable" in line.lower()), "CUDA AdamW gate unavailable")
        return "unavailable", notes, None, elapsed
    match = re.search(r"max error ([0-9.+\-eE]+)", output)
    error = float(match.group(1)) if match else None
    if "pass" not in lowered or error is None:
        return "failed", lines[-1] if lines else "CUDA AdamW gate emitted no PASS/error marker", error, elapsed
    if error > ORACLE_TOLERANCE:
        return "failed", f"native error {error:.3e} exceeds {ORACLE_TOLERANCE:.3e}", error, elapsed
    note = next((line for line in lines if "pass" in line.lower()), "CUDA AdamW gate passed")
    return "pass", note, error, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/cuda_adamw.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    details = metadata(root, fortml, output)
    parameters, first, second, seconds, checksum = adamw_oracle()
    norm = float(np.linalg.norm(parameters))
    rows = [base(
        details, backend="numpy_oracle", device="cpu", status="pass",
        repetitions=32, seconds_per_operation=seconds, metric="parameter_l2_norm",
        value=norm, max_abs_error=0.0,
        notes=(f"state_checksum={checksum:.16e}; beta1={BETA1}; beta2={BETA2}; "
               f"epsilon={EPSILON}; decoupled_weight_decay={WEIGHT_DECAY}"),
    )]
    status, note, observed_error, elapsed = run_gate(fortml)
    rows.append(base(
        details, backend="fortml", device="cuda", status=status,
        repetitions="", seconds_per_operation="", metric="parameter_l2_norm",
        value=norm, max_abs_error=(observed_error if observed_error is not None else ""),
        notes=(f"expected_state_checksum={checksum:.16e}; "
               "gate subprocess duration is not a kernel timing; " + note),
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
