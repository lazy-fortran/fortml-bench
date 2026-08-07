#!/usr/bin/env python3
"""Correctness-gated grouped K-fold index benchmark.

The NumPy implementation is an independent largest-first greedy packing
oracle.  FortML timings are retained only after every test assignment and
train/test group isolation record agrees with that oracle.  CUDA is a typed
capability row because this splitter only owns host index metadata today.
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


GROUPS = np.array([1, 1, 1, 2, 2, 3, 4, 4, 5, 6], dtype=np.int64)
N_SPLITS = 3
REPETITIONS = 512
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_groups", "n_splits", "repetitions", "seconds_per_operation",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    value = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return value + ("+dirty" if dirty else "")


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def group_assignment(groups: np.ndarray, n_splits: int) -> np.ndarray:
    """Return a one-based fold assignment from a separate Python oracle."""
    unique, counts = np.unique(groups, return_counts=True)
    if n_splits < 2 or n_splits > unique.size:
        raise ValueError("invalid split count")
    # Stable descending count order; NumPy's stable sort retains first
    # occurrence order for equal-size groups.
    order = np.argsort(-counts, kind="stable")
    fold_sizes = np.zeros(n_splits, dtype=np.int64)
    assignment = np.zeros(groups.size, dtype=np.int64)
    for group_index in order:
        fold = int(np.argmin(fold_sizes))
        assignment[groups == unique[group_index]] = fold + 1
        fold_sizes[fold] += counts[group_index]
    return assignment


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "workload": "group_kfold", "phase": "split", "backend": "",
        "device": "cpu", "status": "", "n_samples": GROUPS.size,
        "n_groups": np.unique(GROUPS).size, "n_splits": N_SPLITS,
        "repetitions": REPETITIONS, "seconds_per_operation": "",
        "max_abs_error": "", "oracle": "", "notes": "",
    }
    result.update(details)
    result.update(values)
    return result


def run_numpy(details: dict[str, str], expected: np.ndarray) -> list[dict[str, Any]]:
    started = time.perf_counter()
    for _ in range(REPETITIONS):
        actual = group_assignment(GROUPS, N_SPLITS)
        if not np.array_equal(actual, expected):
            raise RuntimeError("NumPy group K-fold oracle is not self-consistent")
    seconds = (time.perf_counter() - started) / REPETITIONS
    return [row(
        details, backend="numpy_oracle", status="pass",
        seconds_per_operation=seconds, max_abs_error=0.0,
        oracle="independent NumPy stable largest-first greedy group packing",
        notes="group IDs remain intact; one-based fold assignments",
    )]


def read_fortran(path: Path) -> np.ndarray:
    assignment = np.zeros(GROUPS.size, dtype=np.int64)
    seen = np.zeros(GROUPS.size, dtype=bool)
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            if record["quantity"] != "test":
                continue
            index = int(record["value"]) - 1
            fold = int(record["fold"])
            if index < 0 or index >= GROUPS.size or seen[index]:
                raise RuntimeError("FortML emitted an invalid or duplicate group test index")
            assignment[index] = fold
            seen[index] = True
    if not np.all(seen):
        raise RuntimeError("FortML omitted a group test index")
    return assignment


def run_fortml(fortml: Path, details: dict[str, str], expected: np.ndarray) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    build = subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
        capture_output=True, text=True,
    )
    if build.returncode != 0:
        return [row(details, backend="fortml", status="unavailable",
                    oracle="FortML release-app protocol", notes="fo build failed")]
    with tempfile.TemporaryDirectory(dir="/mnt/storage", prefix="fortml-group-kfold-") as directory:
        oracle_path = Path(directory) / "group_kfold.csv"
        check_environment = dict(environment)
        check_environment["FORTML_BENCH_GROUP_KFOLD_ORACLE"] = str(oracle_path)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", "fortml_bench_group_kfold"], cwd=fortml,
            env=check_environment, capture_output=True, text=True,
        )
        if completed.returncode != 0 or not oracle_path.is_file():
            return [row(details, backend="fortml", status="unavailable",
                        oracle="FortML release-app protocol",
                        notes="release target did not emit its complete oracle")]
        actual = read_fortran(oracle_path)
        error = float(np.max(np.abs(actual - expected)))
        if error != 0.0:
            raise RuntimeError(f"FortML group K-fold oracle mismatch: {error:.3e}")
        for group in np.unique(GROUPS):
            if np.unique(actual[GROUPS == group]).size != 1:
                raise RuntimeError("FortML split leaks a group across test folds")
        timing = next((float(match.group(1)) for line in completed.stdout.splitlines()
                       for match in [re.match(r"^group_kfold_split,([^,]+)$", line.strip())]
                       if match), None)
    if timing is None:
        raise RuntimeError("FortML group K-fold app emitted no timing")
    return [row(
        details, backend="fortml", status="pass", seconds_per_operation=timing,
        max_abs_error=error,
        oracle="independent NumPy stable largest-first greedy group packing",
        notes="all group test assignments checked before timing",
    )]


def device_refusal(details: dict[str, str]) -> list[dict[str, Any]]:
    return [row(
        {**details, "device": "cuda"}, backend="fortml", status="unavailable",
        oracle="FortML device capability boundary; no CUDA index iterator",
        notes="group splitting owns CPU index metadata only",
    )]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/group_kfold.csv"))
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = arguments.fortml.resolve()
    details = metadata(root, fortml, arguments.output)
    expected = group_assignment(GROUPS, N_SPLITS)
    rows = run_numpy(details, expected) + run_fortml(fortml, details, expected) + device_refusal(details)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: item.get(field, "") for field in FIELDS} for item in rows)


if __name__ == "__main__":
    main()
