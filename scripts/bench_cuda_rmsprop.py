#!/usr/bin/env python3
"""Correctness gate for FortML's resident native-CUDA RMSprop state kernel.

The CUDA gate owns device allocation and launch.  This harness independently
reconstructs the centered, momentum-enabled recurrence in NumPy and accepts
the native result only when the reported state error is within the declared
tolerance.  The gate also checks typed create/step/download refusals; its
subprocess duration includes compilation and is not a kernel timing.
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


N_PARAMETERS = 4
STEPS = 5
LEARNING_RATE = 0.08
DECAY = 0.8
EPSILON = 1.0e-5
MOMENTUM = 0.2
ORACLE_TOLERANCE = 2.0e-12

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
        "workload": "cuda_rmsprop_state", "phase": "optimizer_step",
        "n_parameters": N_PARAMETERS, "steps": STEPS,
        "oracle": "independent NumPy centered RMSprop recurrence",
    })
    row.update(values)
    return row


def rmsprop_oracle() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                               float, float]:
    initial = np.array([0.2, -0.1, 0.3, -0.25], dtype=np.float64)

    def advance() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        parameters = initial.copy()
        square = np.zeros(N_PARAMETERS, dtype=np.float64)
        mean = np.zeros(N_PARAMETERS, dtype=np.float64)
        buffer = np.zeros(N_PARAMETERS, dtype=np.float64)
        for _ in range(STEPS):
            gradient = parameters - 0.1 * np.arange(1, N_PARAMETERS + 1)
            square = DECAY * square + (1.0 - DECAY) * gradient**2
            mean = DECAY * mean + (1.0 - DECAY) * gradient
            variance = np.maximum(square - mean**2, 0.0)
            direction = gradient / (np.sqrt(variance) + EPSILON)
            buffer = MOMENTUM * buffer + direction
            parameters = parameters - LEARNING_RATE * buffer
        return parameters, square, mean, buffer

    started = time.perf_counter()
    states = [advance() for _ in range(32)]
    seconds = (time.perf_counter() - started) / len(states)
    parameters, square, mean, buffer = states[-1]
    checksum = float(np.sum(parameters) + np.sum(square) +
                     np.sum(mean) + np.sum(buffer))
    if not all(np.all(np.isfinite(state)) for state in states[-1]):
        raise RuntimeError("RMSprop NumPy oracle produced nonfinite state")
    return parameters, square, mean, buffer, seconds, checksum


def run_gate(fortml: Path) -> tuple[str, str, float | None]:
    script = fortml / "test" / "run_cuda_rmsprop_state.sh"
    if not script.is_file():
        return "unavailable", f"test script is absent: {script.name}", None
    process = subprocess.run([str(script)], cwd=fortml, capture_output=True,
                             text=True)
    output = (process.stdout + "\n" + process.stderr).strip()
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    lowered = output.lower()
    if process.returncode != 0:
        return "failed", lines[-1] if lines else "CUDA RMSprop gate returned nonzero status", None
    if "skipped" in lowered or "unavailable" in lowered:
        note = next((line for line in lines if "skip" in line.lower() or
                     "unavailable" in line.lower()),
                    "CUDA RMSprop gate unavailable")
        return "unavailable", note, None
    match = re.search(r"max error ([0-9.+\-eE]+)", output)
    error = float(match.group(1)) if match else None
    if "pass" not in lowered or error is None:
        return "failed", lines[-1] if lines else "CUDA RMSprop gate emitted no PASS/error marker", error
    if error > ORACLE_TOLERANCE:
        return "failed", f"native error {error:.3e} exceeds {ORACLE_TOLERANCE:.3e}", error
    note = next((line for line in lines if "pass" in line.lower()),
                "CUDA RMSprop gate passed")
    return "pass", note, error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/cuda_rmsprop.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    details = metadata(root, fortml, output)
    parameters, square, mean, buffer, seconds, checksum = rmsprop_oracle()
    norm = float(np.linalg.norm(parameters))
    rows = [base(
        details, backend="numpy_oracle", device="cpu", status="pass",
        repetitions=32, seconds_per_operation=seconds, metric="parameter_l2_norm",
        value=norm, max_abs_error=0.0,
        notes=(f"state_checksum={checksum:.16e}; decay={DECAY}; "
               f"epsilon={EPSILON}; momentum={MOMENTUM}; centered=1"),
    )]
    status, note, observed_error = run_gate(fortml)
    rows.append(base(
        details, backend="fortml", device="cuda", status=status,
        repetitions="", seconds_per_operation="", metric="parameter_l2_norm",
        value=norm,
        max_abs_error=(observed_error if observed_error is not None else ""),
        notes=(f"expected_state_checksum={checksum:.16e}; "
               "create/step/download refusal checks included; "
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
