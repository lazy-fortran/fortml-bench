#!/usr/bin/env python3
"""Correctness-gated benchmark for sparse-safe CSC standard scaling."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


N_ROWS, N_FEATURES = 4, 3
ROWS = np.array([0, 2, 3, 1, 3])
COLS = np.array([0, 0, 1, 1, 2])
VALUES = np.array([2.0, 4.0, 3.0, 1.0, -2.0])
REPETITIONS = 20_000
PHASES = ("transform", "inverse", "jvp", "vjp")
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_rows",
    "n_features", "nnz", "seconds_per_operation", "max_abs_error", "oracle",
    "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names: set[str] = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            pass
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        name = line[3:].split(" -> ")[-1].strip()
        if name not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {field: "" for field in FIELDS}
    result.update(details)
    result.update({
        "workload": "sparse_standard_scaler", "device": "cpu", "status": "",
        "n_rows": N_ROWS, "n_features": N_FEATURES, "nnz": VALUES.size,
        "oracle": "", "notes": "",
    })
    result.update(values)
    return result


def dense(values: np.ndarray = VALUES) -> np.ndarray:
    result = np.zeros((N_ROWS, N_FEATURES), dtype=np.float64)
    result[ROWS, COLS] = values
    return result


def expected() -> dict[str, np.ndarray]:
    values = dense()
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales[scales == 0.0] = 1.0
    transformed = values / scales
    tangent = 2.0 * values / scales
    return {"transform": transformed, "inverse": values,
            "jvp": tangent, "vjp": tangent}


def parse_oracle(path: Path) -> dict[str, np.ndarray]:
    result = {phase: np.zeros((N_ROWS, N_FEATURES), dtype=np.float64) for phase in PHASES}
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            result[record["phase"]][int(record["row"]) - 1, int(record["column"]) - 1] = float(record["value"])
    return result


def numpy_rows(details: dict[str, str]) -> list[dict[str, Any]]:
    values = dense()
    scales = values.std(axis=0)
    scales[scales == 0.0] = 1.0
    transformed = values / scales
    tangent = 2.0 * values
    operations = {
        "transform": lambda: values / scales,
        "inverse": lambda: transformed * scales,
        "jvp": lambda: tangent / scales,
        "vjp": lambda: tangent / scales,
    }
    rows: list[dict[str, Any]] = []
    for phase in PHASES:
        started = time.perf_counter()
        for _ in range(REPETITIONS):
            _ = operations[phase]()
        seconds = (time.perf_counter() - started) / REPETITIONS
        rows.append(row(details, phase=phase, backend="numpy_oracle", status="pass",
                        seconds_per_operation=seconds, max_abs_error=0.0,
                        oracle="independent dense NumPy implicit-zero scaler oracle",
                        notes="CSC fixture expanded densely for value comparison"))
    return rows


def fortml_rows(root: Path, fortml: Path, details: dict[str, str], no_build: bool) -> list[dict[str, Any]]:
    target = "fortml_bench_sparse_scaler"
    source = fortml / "app" / f"{target}.f90"
    if not source.is_file():
        return [row(details, phase=phase, backend="fortml_cpu", status="unavailable",
                    oracle="typed release-target contract", notes=f"missing {source.name}")
                for phase in PHASES]
    if not no_build:
        subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, check=True)
    with tempfile.TemporaryDirectory(dir=fortml / "build") as directory:
        oracle_path = Path(directory) / "sparse_scaler.csv"
        environment = os.environ.copy()
        environment.update({"FO_FC": environment.get("FO_FC", "gfortran"),
                            "OMP_NUM_THREADS": "1",
                            "FORTML_BENCH_SPARSE_SCALER_ORACLE": str(oracle_path)})
        completed = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                                   env=environment, capture_output=True, text=True,
                                   check=True)
        actual = parse_oracle(oracle_path)
    expected_values = expected()
    errors = [float(np.max(np.abs(actual[phase] - expected_values[phase]))) for phase in PHASES]
    error = max(errors)
    if error > 5.0e-14:
        raise RuntimeError(f"FortML sparse scaler oracle mismatch: {error:.3e}")
    timings: dict[str, float] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] in PHASES:
            timings[fields[0]] = float(fields[1])
    if set(timings) != set(PHASES):
        raise RuntimeError(f"FortML sparse scaler app omitted timings: {timings}")
    rows = [row(details, phase=phase, backend="fortml_cpu", status="pass",
                seconds_per_operation=timings[phase], max_abs_error=error,
                oracle="independent dense NumPy implicit-zero scaler oracle",
                notes="transform, inverse, JVP, and VJP sparse values matched")
            for phase in PHASES]
    rows.append(row(details, phase="transform", backend="fortml_cuda", device="cuda",
                    status="unavailable", oracle="typed_device_contract",
                    notes="resident sparse preprocessing kernel is not linked; typed refusal"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/sparse_preprocessing.csv"))
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    details = metadata(root, fortml, args.output)
    records = numpy_rows(details)
    records.extend(fortml_rows(root, fortml, details, args.no_build))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()
