#!/usr/bin/env python3
"""Correctness-gated built-in MLP learning-rate schedule benchmark.

The NumPy formulas are intentionally independent from FortML.  A complete
FortML release app must emit every schedule value and analytic derivative for
the same grid before its timing is retained.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


BASE_RATE = 0.2
UPDATES = (1, 2, 5, 10, 12)
REPETITIONS = 4096
FINITE_DIFFERENCE_STEP = 1.0e-6
ORACLE_TOLERANCE = 2.0e-12
QUANTITIES = ("rate", "d_base_rate", "d_min_rate_fraction", "d_decay_factor")
SCHEDULES = (
    ("constant", {}),
    ("linear_warmup", {"warmup_updates": 4}),
    ("cosine_decay", {"total_updates": 10, "min_rate_fraction": 0.1}),
    ("warmup_cosine", {"warmup_updates": 2, "total_updates": 10,
                        "min_rate_fraction": 0.2}),
    ("exponential_decay", {"warmup_updates": 2, "decay_factor": 0.8}),
)
FIELDS = (
    "workload", "schedule", "phase", "backend", "device", "status",
    "n_updates", "repetitions", "seconds_per_operation", "metric", "value",
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


def schedule_value(name: str, update: int, base_rate: float = BASE_RATE,
                   warmup_updates: int | None = None,
                   total_updates: int | None = None,
                   min_rate_fraction: float | None = None,
                   decay_factor: float | None = None) -> tuple[float, float, float, float]:
    """Return rate and products with respect to base/min-fraction/decay."""
    factor = 1.0
    d_min_factor = 0.0
    d_decay_factor = 0.0
    if name == "linear_warmup":
        warmup = 4 if warmup_updates is None else warmup_updates
        factor = min(1.0, update / float(warmup))
    elif name == "cosine_decay":
        total = 10 if total_updates is None else total_updates
        progress = min(1.0, update / float(total))
        cosine = math.cos(math.pi * progress)
        minimum = 0.1 if min_rate_fraction is None else min_rate_fraction
        factor = minimum + (1.0 - minimum) * 0.5 * (1.0 + cosine)
        d_min_factor = 0.5 * (1.0 - cosine)
    elif name == "warmup_cosine":
        warmup = 2 if warmup_updates is None else warmup_updates
        total = 10 if total_updates is None else total_updates
        if update <= warmup:
            factor = update / warmup
        else:
            progress = min(1.0, (update - warmup) / (total - warmup))
            cosine = math.cos(math.pi * progress)
            minimum = 0.2 if min_rate_fraction is None else min_rate_fraction
            factor = minimum + (1.0 - minimum) * 0.5 * (1.0 + cosine)
            d_min_factor = 0.5 * (1.0 - cosine)
    elif name == "exponential_decay":
        warmup = 2 if warmup_updates is None else warmup_updates
        decay = 0.8 if decay_factor is None else decay_factor
        elapsed = max(0, update - warmup)
        factor = decay ** elapsed
        if elapsed > 0:
            d_decay_factor = elapsed * decay ** (elapsed - 1)
    elif name != "constant":
        raise ValueError(f"unknown schedule {name}")
    rate = base_rate * factor
    return rate, factor, base_rate * d_min_factor, base_rate * d_decay_factor


def independent_oracle() -> dict[tuple[str, int, str], float]:
    expected: dict[tuple[str, int, str], float] = {}
    for schedule_index, (name, parameters) in enumerate(SCHEDULES, 1):
        for update in UPDATES:
            index = 100 * (schedule_index - 1) + update
            values = schedule_value(name, update, **parameters)
            for quantity, value in zip(QUANTITIES, values):
                expected[(name, index, quantity)] = value
            # Check the continuous products independently with central
            # differences.  This is an oracle check, not an implementation
            # self-consistency check against FortML.
            value, d_base, d_min, d_decay = values
            plus = schedule_value(name, update, BASE_RATE + FINITE_DIFFERENCE_STEP,
                                  **parameters)[0]
            minus = schedule_value(name, update, BASE_RATE - FINITE_DIFFERENCE_STEP,
                                   **parameters)[0]
            if abs(d_base - (plus - minus) / (2.0 * FINITE_DIFFERENCE_STEP)) > 2.0e-10:
                raise RuntimeError(f"base-rate oracle failed for {name} update {update}")
            if "min_rate_fraction" in parameters:
                plus = schedule_value(name, update,
                                      min_rate_fraction=parameters["min_rate_fraction"] +
                                      FINITE_DIFFERENCE_STEP)[0]
                minus = schedule_value(name, update,
                                       min_rate_fraction=parameters["min_rate_fraction"] -
                                       FINITE_DIFFERENCE_STEP)[0]
                if abs(d_min - (plus - minus) / (2.0 * FINITE_DIFFERENCE_STEP)) > 2.0e-10:
                    raise RuntimeError(f"minimum-rate oracle failed for {name} update {update}")
            if "decay_factor" in parameters:
                plus = schedule_value(name, update,
                                      decay_factor=parameters["decay_factor"] +
                                      FINITE_DIFFERENCE_STEP)[0]
                minus = schedule_value(name, update,
                                       decay_factor=parameters["decay_factor"] -
                                       FINITE_DIFFERENCE_STEP)[0]
                if abs(d_decay - (plus - minus) / (2.0 * FINITE_DIFFERENCE_STEP)) > 2.0e-9:
                    raise RuntimeError(f"decay-factor oracle failed for {name} update {update}")
    return expected


def timed(operation: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    result = None
    for _ in range(REPETITIONS):
        result = operation()
    return result, (time.perf_counter() - started) / (REPETITIONS * len(UPDATES))


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def row(details: dict[str, str], schedule: str, phase: str, backend: str,
        status: str, value: Any, seconds: Any, error: Any, notes: str,
        device: str = "cpu") -> dict[str, Any]:
    result: dict[str, Any] = {field: "" for field in FIELDS}
    result.update(details)
    result.update({
        "workload": "mlp_schedules", "schedule": schedule, "phase": phase,
        "backend": backend, "device": device, "status": status,
        "n_updates": len(UPDATES), "repetitions": REPETITIONS,
        "seconds_per_operation": seconds, "metric": phase, "value": value,
        "max_abs_error": error, "oracle": "independent NumPy schedule formulas",
        "notes": notes,
    })
    return result


def device_refusal_rows(details: dict[str, str]) -> list[dict[str, Any]]:
    """Record the CUDA schedule boundary without retaining a fake timing."""
    return [row(
        details, name, "device_capability", "fortml", "unavailable", "", "", "",
        "schedule device_supported(CUDA)=false; no resident optimizer lowering",
        device="cuda",
    ) for name, _ in SCHEDULES]


def run_numpy(details: dict[str, str], expected: dict[tuple[str, int, str], float]) -> list[dict[str, Any]]:
    rows = []
    for schedule_index, (name, parameters) in enumerate(SCHEDULES, 1):
        index_values = [100 * (schedule_index - 1) + update for update in UPDATES]
        for quantity_index, quantity in enumerate(QUANTITIES):
            def operation() -> np.ndarray:
                return np.asarray([
                    schedule_value(name, update, **parameters)[quantity_index]
                    for update in UPDATES
                ], dtype=np.float64)

            actual, seconds = timed(operation)
            expected_values = np.asarray([
                expected[(name, index, quantity)] for index in index_values
            ])
            error = float(np.max(np.abs(actual - expected_values)))
            if error > 1.0e-15:
                raise RuntimeError(f"NumPy {name} {quantity} self-check failed: {error:.3e}")
            rows.append(row(details, name, quantity, "numpy_oracle", "pass",
                            float(np.sum(actual)), seconds, error,
                            "central-difference products checked before timing"))
    return rows


def read_oracle(path: Path) -> dict[tuple[str, int], float]:
    values: dict[tuple[str, int], float] = {}
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            key = (record["quantity"], int(record["index"]))
            if key in values:
                raise RuntimeError(f"duplicate FortML schedule oracle key: {key}")
            values[key] = float(record["value"])
    return values


def run_fortml(root: Path, details: dict[str, str],
               expected: dict[tuple[str, int, str], float]) -> list[dict[str, Any]]:
    source = root / "app" / "fortml_bench_mlp_schedules.f90"
    if not source.is_file():
        return [row(details, "all", "release_app", "fortml", "unavailable", "", "", "",
                    f"release target source is absent: {source.name}")]
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"),
                         "OMP_NUM_THREADS": "1"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=root,
                           env=environment, capture_output=True, text=True)
    if build.returncode != 0:
        return [row(details, "all", "release_app", "fortml", "unavailable", "", "", "",
                    "fo build failed; no FortML timing retained")]
    with tempfile.TemporaryDirectory(dir="/mnt/storage") as directory:
        oracle_path = Path(directory) / "mlp_schedule_oracle.csv"
        run_environment = dict(environment)
        run_environment.update({"FORTML_BENCH_MLP_SCHEDULE_ORACLE": str(oracle_path),
                                "FORTML_BENCH_ORACLE_ONLY": "1"})
        check = subprocess.run(["fo", "exec", "--no-build", "fortml_bench_mlp_schedules"],
                               cwd=root, env=run_environment, capture_output=True, text=True)
        if check.returncode != 0 or not oracle_path.is_file():
            return [row(details, "all", "release_app", "fortml", "unavailable", "", "", "",
                        "release target did not emit its complete oracle")]
        actual = read_oracle(oracle_path)
        required = {(quantity, index) for (name, index, quantity) in expected}
        if set(actual) != required:
            raise RuntimeError("FortML schedule app omitted a complete value/derivative array")
        errors = [abs(actual[(quantity, index)] - value)
                  for (name, index, quantity), value in expected.items()]
        error = float(max(errors))
        if error > ORACLE_TOLERANCE:
            raise RuntimeError(f"FortML schedule oracle mismatch: {error:.3e}")
        timed_environment = dict(environment)
        timed_environment.pop("FORTML_BENCH_MLP_SCHEDULE_ORACLE", None)
        timed_environment.pop("FORTML_BENCH_ORACLE_ONLY", None)
        timed_run = subprocess.run(
            ["fo", "exec", "--no-build", "fortml_bench_mlp_schedules"],
            cwd=root, env=timed_environment, capture_output=True, text=True,
        )
        if timed_run.returncode != 0:
            return [row(details, "all", "release_app", "fortml", "unavailable", "", "", "",
                        "release target timing invocation failed")]
        match = re.search(r"mlp_schedule_rate_with_derivatives,\s*([0-9.eE+-]+)",
                          timed_run.stdout)
        if match is None:
            raise RuntimeError("FortML schedule app omitted timing output")
        seconds = float(match.group(1))
        return [row(details, name, quantity, "fortml", "pass", float(actual[(quantity, index)]),
                    seconds, abs(actual[(quantity, index)] - expected[(name, index, quantity)]),
                    "complete-array app checked before retaining timing")
                    for name, index, quantity in expected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/mlp_schedules.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    details = metadata(root, args.fortml.resolve(), output)
    expected = independent_oracle()
    rows = run_numpy(details, expected)
    rows.extend(run_fortml(args.fortml.resolve(), details, expected))
    rows.extend(device_refusal_rows(details))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
