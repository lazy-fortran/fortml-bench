#!/usr/bin/env python3
"""Correctness-gated chronological validation benchmark.

The NumPy implementation is an independent blocked-window oracle.  It checks
every one-based train/test index emitted by the FortML release app before
retaining the measured timing.  A CUDA row is a typed capability record:
splitters own CPU index metadata and never hide a host fallback behind a CUDA
claim.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


N_SAMPLES = 29
N_SPLITS = 4
TEST_SIZE = 3
GAP = 2
MAX_TRAIN_SIZE = 7
REPETITIONS = 256
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_splits", "test_size", "gap", "max_train_size", "repetitions",
    "seconds_per_operation", "max_abs_error", "oracle", "python_version",
    "numpy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    """Return a revision plus a dirty marker outside explicitly ignored files."""
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names: set[str] = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            pass
    dirty = any(
        line[3:].split(" -> ")[-1].strip() not in ignored_names
        for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True,
        ).splitlines()
    )
    return head + ("+dirty" if dirty else "")


def windows() -> list[tuple[np.ndarray, np.ndarray]]:
    """Build the independent one-based expanding/rolling window oracle."""
    initial_train = N_SAMPLES - N_SPLITS * TEST_SIZE - GAP
    if initial_train < 1:
        raise ValueError("fixture has no initial training rows")
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(N_SPLITS):
        test_start = initial_train + GAP + fold * TEST_SIZE + 1
        test_end = test_start + TEST_SIZE - 1
        train_end = test_start - GAP - 1
        train_start = max(1, train_end - MAX_TRAIN_SIZE + 1)
        train = np.arange(train_start, train_end + 1, dtype=np.int64)
        test = np.arange(test_start, test_end + 1, dtype=np.int64)
        if train.size == 0 or test.size != TEST_SIZE or test_end > N_SAMPLES:
            raise ValueError("oracle generated an invalid window")
        result.append((train, test))
    return result


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "workload": "time_series_split", "phase": "split", "backend": "",
        "device": "cpu", "status": "", "n_samples": N_SAMPLES,
        "n_splits": N_SPLITS, "test_size": TEST_SIZE, "gap": GAP,
        "max_train_size": MAX_TRAIN_SIZE, "repetitions": REPETITIONS,
        "seconds_per_operation": "", "max_abs_error": "", "oracle": "",
        "notes": "",
    }
    result.update(details)
    result.update(values)
    return result


def run_numpy(
    details: dict[str, str], expected: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    for _ in range(REPETITIONS):
        actual = windows()
        if any(not np.array_equal(a_train, e_train) or not np.array_equal(a_test, e_test)
               for (a_train, a_test), (e_train, e_test) in zip(actual, expected)):
            raise RuntimeError("NumPy time-series oracle is not self-consistent")
    seconds = (time.perf_counter() - started) / REPETITIONS
    return [row(
        details, backend="numpy_oracle", status="pass",
        seconds_per_operation=seconds, max_abs_error=0.0,
        oracle="independent NumPy chronological rolling-window formula",
        notes="gap and every one-based train/test row checked",
    )]


def read_fortran(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    train: list[list[int]] = [[] for _ in range(N_SPLITS)]
    test: list[list[int]] = [[] for _ in range(N_SPLITS)]
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            fold = int(record["fold"]) - 1
            if fold < 0 or fold >= N_SPLITS:
                raise RuntimeError("FortML emitted an invalid fold")
            value = int(record["value"])
            if record["quantity"] == "train":
                train[fold].append(value)
            elif record["quantity"] == "test":
                test[fold].append(value)
            else:
                raise RuntimeError("FortML emitted an unknown oracle quantity")
    return [(
        np.asarray(train[fold], dtype=np.int64),
        np.asarray(test[fold], dtype=np.int64),
    ) for fold in range(N_SPLITS)]


def run_fortml(
    fortml: Path,
    details: dict[str, str],
    expected: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"),
                        "OMP_NUM_THREADS": "1"})
    build = subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
        capture_output=True, text=True,
    )
    if build.returncode:
        raise RuntimeError(build.stderr.strip() or build.stdout.strip() or "fo build failed")
    with tempfile.TemporaryDirectory(dir="/mnt/storage", prefix="fortml-time-series-") as directory:
        oracle_path = Path(directory) / "time_series.csv"
        check_environment = dict(environment)
        check_environment["FORTML_BENCH_TIME_SERIES_ORACLE"] = str(oracle_path)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", "fortml_bench_time_series_split"],
            cwd=fortml, env=check_environment, capture_output=True, text=True,
        )
        if completed.returncode or not oracle_path.is_file():
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or
                               "release target did not emit its complete oracle")
        actual = read_fortran(oracle_path)
        errors = [
            np.max(np.abs(a - e))
            for (a_train, a_test), (e_train, e_test) in zip(actual, expected)
            for a, e in ((a_train, e_train), (a_test, e_test))
            if a.size == e.size
        ]
        if any(a.size != e.size for (a_train, a_test), (e_train, e_test) in zip(actual, expected)
               for a, e in ((a_train, e_train), (a_test, e_test))):
            raise RuntimeError("FortML emitted a window with the wrong size")
        error = float(max(errors, default=0.0))
        if error != 0.0:
            raise RuntimeError(f"FortML time-series oracle mismatch: {error:.3e}")
        orientation = next((float(match.group(1)) for line in completed.stdout.splitlines()
                            for match in [re.match(r"^score_orientation,([^,]+)$", line.strip())]
                            if match), None)
        if orientation is None or abs(orientation + 0.2) > 1.0e-15:
            raise RuntimeError("FortML scorer orientation metadata mismatch")
        timing = next((float(match.group(1)) for line in completed.stdout.splitlines()
                       for match in [re.match(r"^time_series_split,([^,]+)$", line.strip())]
                       if match), None)
    if timing is None:
        raise RuntimeError("FortML release app emitted no timing")
    return [
        row(details, backend="fortml", status="pass", seconds_per_operation=timing,
            max_abs_error=error,
            oracle="independent NumPy chronological rolling-window formula",
            notes="all train/test assignments and scorer orientation checked before timing"),
        row(details, phase="metadata", backend="fortml", status="pass",
            seconds_per_operation="", max_abs_error=0.0,
            oracle="independent maximize-orientation scorer oracle",
            notes="log-loss 0.2 is oriented to -0.2"),
    ]


def device_refusal(details: dict[str, str]) -> list[dict[str, Any]]:
    return [row(
        {**details, "device": "cuda"}, backend="fortml", status="unavailable",
        oracle="FortML device capability boundary; no CUDA index iterator",
        notes="time-series split and clone/scorer metadata own CPU control-plane state only",
    )]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/time_series_split.csv"))
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = arguments.fortml.resolve()
    output = arguments.output.resolve()
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }
    expected = windows()
    records = run_numpy(details, expected) + run_fortml(fortml, details, expected) + \
        device_refusal(details)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: item.get(field, "") for field in FIELDS} for item in records)
    print(f"wrote {len(records)} rows to {output}")


if __name__ == "__main__":
    main()
