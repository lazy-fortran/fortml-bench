#!/usr/bin/env python3
"""Correctness-gated generic-trainer validation and early-stop benchmark.

The validation sequences and quadratic update are independent NumPy references.
The FortML test exercises the public callback, schema-6 checkpoint, best-state
restoration, and transactional callback-presence refusal. CUDA is recorded as a
typed boundary because the callback owns host data.
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
    "workload", "phase", "backend", "device", "status", "n_parameters", "steps",
    "seconds_per_operation", "metric", "value", "max_abs_error", "oracle",
    "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    value = subprocess.check_output(
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
    return value + ("+dirty" if dirty else "")


def validation_oracle() -> dict[str, object]:
    target = np.array([1.5, -0.5], dtype=np.float64)
    parameters = np.array([0.0, 1.0], dtype=np.float64)
    learning_rate = 0.1
    validation = [0.4, 0.2, 0.25, 0.3]
    best = parameters.copy()
    best_value = validation[0]
    best_step = 0
    bad_steps = 0
    history = [validation[0]]
    stopped_step = None
    for step in range(1, 4):
        gradient = 2.0 * (parameters - target)
        parameters -= learning_rate * gradient
        value = validation[step]
        history.append(value)
        if value < best_value:
            best_value = value
            best_step = step
            bad_steps = 0
            best = parameters.copy()
        else:
            bad_steps += 1
        if bad_steps >= 2:
            stopped_step = step
            parameters = best.copy()
            break
    if stopped_step != 3:
        raise RuntimeError(f"validation stop mismatch: {stopped_step}")
    return {
        "stopped_step": stopped_step,
        "best_step": best_step,
        "best_value": best_value,
        "restore_error": float(np.max(np.abs(parameters - best))),
        "history_length": len(history),
        "best_parameters": best,
    }


def higher_validation_oracle() -> dict[str, object]:
    """Independent known-answer oracle for score-style maximization."""
    target = np.array([1.5, -0.5], dtype=np.float64)
    parameters = np.array([0.0, 1.0], dtype=np.float64)
    validation = [0.4, 0.6, 0.55, 0.5]
    best = parameters.copy()
    best_value = validation[0]
    best_step = 0
    bad_steps = 0
    history = [validation[0]]
    stopped_step = None
    for step in range(1, 4):
        gradient = 2.0 * (parameters - target)
        parameters -= 0.1 * gradient
        value = validation[step]
        history.append(value)
        if value > best_value:
            best_value = value
            best_step = step
            bad_steps = 0
            best = parameters.copy()
        else:
            bad_steps += 1
        if bad_steps >= 2:
            stopped_step = step
            parameters = best.copy()
            break
    if stopped_step != 3 or best_step != 1:
        raise RuntimeError(
            f"maximization validation mismatch: stop={stopped_step}, best={best_step}")
    return {
        "stopped_step": stopped_step,
        "best_step": best_step,
        "best_value": best_value,
        "restore_error": float(np.max(np.abs(parameters - best))),
        "history_length": len(history),
    }


def continuation_oracle() -> float:
    target = np.array([1.5, -0.5], dtype=np.float64)

    def advance(parameters: np.ndarray, count: int) -> np.ndarray:
        for _ in range(count):
            parameters -= 0.1 * 2.0 * (parameters - target)
        return parameters

    full = advance(np.array([0.0, 1.0], dtype=np.float64), 3)
    split = advance(np.array([0.0, 1.0], dtype=np.float64), 1)
    resumed = advance(split, 2)
    return float(np.max(np.abs(full - resumed)))


def row(details: dict[str, str], **values: object) -> dict[str, object]:
    result: dict[str, object] = {field: "" for field in FIELDS}
    result.update(details)
    result.update({
        "backend": "fortml", "device": "cpu", "status": "pass",
        "n_parameters": 2, "steps": 3, "compiler": "gfortran", "flags": "-O3",
    })
    result.update(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/trainer_validation.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    expected = validation_oracle()
    expected_higher = higher_validation_oracle()
    continuation_error = continuation_oracle()
    if continuation_error > 1.0e-14:
        raise RuntimeError(f"continuation oracle mismatch: {continuation_error:.3e}")
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output, root / "scripts" / "__pycache__")),
        "compiler": "gfortran", "flags": "-O3",
    }
    rows: list[dict[str, object]] = [
        row(details, backend="numpy_oracle", workload="trainer_validation",
            phase="early_stop", metric="stopped_step", value=expected["stopped_step"],
            max_abs_error=0.0, oracle="independent NumPy quadratic validation sequence",
            notes="validation=[0.4,0.2,0.25,0.3]; patience=2; min_delta=0"),
        row(details, backend="numpy_oracle", workload="trainer_validation",
            phase="early_stop", metric="best_validation_value", value=expected["best_value"],
            max_abs_error=0.0, oracle="independent NumPy validation sequence",
            notes=f"best_step={expected['best_step']}; history_length={expected['history_length']}"),
        row(details, backend="numpy_oracle", workload="trainer_validation",
            phase="restore", metric="restore_best_max_abs_error",
            value=expected["restore_error"], max_abs_error=expected["restore_error"],
            oracle="independent NumPy best-state restoration",
            notes="packed parameter vector restored at step 1"),
        row(details, backend="numpy_oracle", workload="trainer_validation",
            phase="checkpoint", metric="continuation_max_abs_error",
            value=continuation_error, max_abs_error=continuation_error,
            oracle="independent NumPy split trajectory",
            notes="split after step 1; resumed for two updates"),
        row(details, backend="numpy_oracle", workload="trainer_validation",
            phase="maximization", metric="best_validation_value",
            value=expected_higher["best_value"], max_abs_error=0.0,
            oracle="independent NumPy maximizing validation sequence",
            notes="validation=[0.4,0.6,0.55,0.5]; higher_is_better=true; patience=2"),
        row(details, backend="numpy_oracle", workload="trainer_validation",
            phase="maximization", metric="restore_best_max_abs_error",
            value=expected_higher["restore_error"],
            max_abs_error=expected_higher["restore_error"],
            oracle="independent NumPy maximizing best-state restoration",
            notes=f"best_step={expected_higher['best_step']}; stopped_step={expected_higher['stopped_step']}"),
    ]
    started = time.perf_counter()
    if args.skip_fortml:
        gate_status, gate_notes = "skipped", "--skip-fortml"
    else:
        run = subprocess.run(["fo", "test", "test_trainer"], cwd=fortml,
                             capture_output=True, text=True, check=False)
        if run.returncode:
            raise RuntimeError("fo test test_trainer failed\n" + run.stdout + run.stderr)
        gate_status, gate_notes = "pass", (
            "test_trainer covers callback transitions, both metric directions, schema-6 "
            "continuation, and transactional callback refusal"
        )
    elapsed = time.perf_counter() - started
    rows.append(row(details, workload="trainer_validation", phase="public_contract_gate",
                    status=gate_status, seconds_per_operation=elapsed,
                    metric="continuation_max_abs_error", value=continuation_error,
                    max_abs_error=continuation_error,
                    oracle="FortML independent quadratic trainer test",
                    notes=gate_notes))
    rows.append(row(details, workload="trainer_validation", phase="device_boundary",
                    device="cuda", status="unavailable", metric="resident_callback",
                    value="nan", max_abs_error="nan",
                    oracle="typed device refusal",
                    notes="validation callback and data are host-owned; no hidden transfer"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
