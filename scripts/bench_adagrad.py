#!/usr/bin/env python3
"""Independent Adagrad recurrence and checkpoint/resume benchmark.

The NumPy lane is a behavioral oracle for FortOpt's canonical update
``G <- G + g**2`` and epsilon-stabilized diagonal step.  The FortML release
app exports its final parameter norm and timing; the timing is retained only
after that norm matches the independent recurrence.
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


N_PARAMETERS = 4096
STEPS = 128
SPLIT_STEP = 64
LEARNING_RATE = 1.0e-2
EPSILON = 1.0e-8
REPETITIONS = 16

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
        "benchmark_revision": revision(
            root, (output, root / "results" / "pca.csv", root / "results" / "adagrad.csv")
        ),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def base_row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = dict(details)
    row.update({
        "workload": "adagrad", "phase": "", "backend": "", "device": "cpu",
        "status": "", "n_parameters": N_PARAMETERS, "steps": STEPS,
        "repetitions": "", "seconds_per_operation": "", "metric": "",
        "value": "", "max_abs_error": "", "oracle": "", "notes": "",
    })
    row.update(values)
    return row


def gradients(theta: np.ndarray, target: np.ndarray) -> np.ndarray:
    # The quadratic objective is independent of the optimizer recurrence.
    return theta - target


def run_oracle() -> list[dict[str, Any]]:
    index = np.arange(1, N_PARAMETERS + 1, dtype=np.float64)
    initial = 0.1 * np.cos(0.003 * index)
    target = 0.25 * np.sin(0.0017 * index)

    def advance(theta: np.ndarray, accumulator: np.ndarray, count: int,
                steps: int) -> tuple[np.ndarray, np.ndarray]:
        for step in range(count + 1, count + steps + 1):
            gradient = gradients(theta, target)
            accumulator = accumulator + gradient**2
            theta = theta - LEARNING_RATE * gradient / (np.sqrt(accumulator) + EPSILON)
        return theta, accumulator

    started = time.perf_counter()
    for _ in range(REPETITIONS):
        full_theta, full_accumulator = advance(initial.copy(), np.zeros_like(initial), 0, STEPS)
    full_seconds = (time.perf_counter() - started) / REPETITIONS
    first_theta, first_accumulator = advance(initial.copy(), np.zeros_like(initial), 0, SPLIT_STEP)
    resumed_theta, resumed_accumulator = advance(first_theta, first_accumulator, SPLIT_STEP,
                                                  STEPS - SPLIT_STEP)
    state_error = max(float(np.max(np.abs(resumed_theta - full_theta))),
                      float(np.max(np.abs(resumed_accumulator - full_accumulator))))
    if state_error > 1.0e-14:
        raise RuntimeError(f"Adagrad checkpoint/resume mismatch: {state_error:.3e}")
    expected_norm = float(np.linalg.norm(full_theta))
    return [base_row(
        {}, phase="train", backend="numpy_oracle", status="pass",
        repetitions=REPETITIONS, seconds_per_operation=full_seconds,
        metric="parameter_l2_norm", value=expected_norm, max_abs_error=0.0,
        oracle="independent quadratic gradient plus Adagrad recurrence",
        notes="G <- G + gradient**2; epsilon-stabilized diagonal update"),
        base_row(
            {}, phase="resume", backend="numpy_oracle", status="pass",
            repetitions=1, seconds_per_operation="", metric="state_max_abs_error",
            value=state_error, max_abs_error=state_error,
            oracle="independent uninterrupted versus split/resumed trajectory",
            notes=f"split_step={SPLIT_STEP}; state includes parameters and accumulator")]


def run_fortml(fortml: Path, target: str, details: dict[str, str],
               expected_norm: float) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    source = fortml / "app" / f"{target}.f90"
    if not source.is_file():
        return base_row(details, phase="train", backend="fortml", status="unavailable",
                        oracle="FortML release-app protocol",
                        notes=f"release target source is absent: {source.name}")
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                           env=environment, capture_output=True, text=True)
    if build.returncode != 0:
        note = build.stderr.strip().splitlines()[-1] if build.stderr.strip() else "fo build failed"
        return base_row(details, phase="train", backend="fortml", status="unavailable",
                        oracle="FortML release-app protocol", notes=note)
    run = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                         env=environment, capture_output=True, text=True)
    if run.returncode == 0:
        match = re.search(
            r"^adagrad_training,\s*\d+,\s*\d+,\s*([0-9Ee+.-]+),\s*([0-9Ee+.-]+)$",
            run.stdout, re.MULTILINE)
        if match is None:
            return base_row(details, phase="train", backend="fortml", status="unavailable",
                            oracle="FortML release-app protocol",
                            notes="release app emitted no parseable recurrence record")
        actual_norm = float(match.group(1))
        seconds = float(match.group(2))
        error = abs(actual_norm - expected_norm)
        if error > 1.0e-12:
            raise RuntimeError(f"FortML Adagrad parameter norm mismatch: {error:.3e}")
        return base_row(details, phase="train", backend="fortml", status="pass",
                        repetitions=REPETITIONS, seconds_per_operation=seconds,
                        metric="parameter_l2_norm", value=actual_norm,
                        max_abs_error=error,
                        oracle="FortOpt Adagrad release app plus independent NumPy recurrence",
                        notes=target)
    stderr = run.stderr.strip().splitlines()
    note = f"target {target!r} unavailable"
    for line in stderr:
        if "target" in line.lower() or "unknown" in line.lower():
            note = line.strip()
            break
    return base_row(details, phase="train", backend="fortml", status="unavailable",
                    oracle="FortML release-app protocol", notes=note)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/adagrad.csv"))
    parser.add_argument("--target", default="fortml_bench_adagrad_training")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    fortml = args.fortml.resolve()
    details = metadata(root, fortml, output)
    rows = run_oracle()
    expected_norm = float(rows[0]["value"])
    if args.skip_fortml:
        rows.append(base_row(details, phase="train", backend="fortml", status="skipped",
                             oracle="FortML release-app protocol", notes="--skip-fortml"))
    else:
        row = run_fortml(fortml, args.target, details, expected_norm)
        rows.append(row)
    for row in rows:
        row.update(details)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
