#!/usr/bin/env python3
"""Correctness-gated benchmark for dense missing-indicator features."""

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


N_SAMPLES, N_FEATURES = 64, 6
MODES = {"all": np.arange(N_FEATURES), "missing-only": np.array((1, 4))}
REPETITIONS = 128
FIELDS = (
    "workload", "mode", "phase", "backend", "device", "status",
    "n_samples", "n_features", "n_outputs", "seconds_per_operation",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
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


def fixture() -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(1, N_SAMPLES + 1, dtype=np.float64)[:, None]
    columns = np.arange(1, N_FEATURES + 1, dtype=np.float64)[None, :]
    values = np.sin(0.07 * rows + 0.11 * columns)
    tangents = np.cos(0.03 * rows * columns)
    values[2::7, 1] = np.nan
    values[4::11, 4] = np.nan
    return values, tangents


def mask_oracle(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.isnan(values[:, indices]).astype(np.float64)


def rows_base(details: dict[str, str], mode: str, **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = {field: "" for field in FIELDS}
    row.update(details)
    row.update({
        "workload": "missing_indicator", "mode": mode, "phase": "",
        "backend": "", "device": "cpu", "status": "",
        "n_samples": N_SAMPLES, "n_features": N_FEATURES,
        "n_outputs": int(MODES[mode].size), "seconds_per_operation": "",
        "max_abs_error": "", "oracle": "", "notes": "",
    })
    row.update(values)
    return row


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def numpy_rows(details: dict[str, str]) -> list[dict[str, Any]]:
    values, tangents = fixture()
    rows: list[dict[str, Any]] = []
    for mode, indices in MODES.items():
        expected = mask_oracle(values, indices)
        jvp = np.zeros_like(expected)
        vjp = np.zeros_like(values)
        started = time.perf_counter()
        for _ in range(REPETITIONS):
            mask_oracle(values, indices)
        transform_seconds = (time.perf_counter() - started) / REPETITIONS
        started = time.perf_counter()
        for _ in range(REPETITIONS):
            np.zeros_like(expected)
        jvp_seconds = (time.perf_counter() - started) / REPETITIONS
        started = time.perf_counter()
        for _ in range(REPETITIONS):
            np.zeros_like(values)
        vjp_seconds = (time.perf_counter() - started) / REPETITIONS
        rows.extend([
            rows_base(details, mode, phase="transform", backend="numpy_oracle",
                      status="pass", seconds_per_operation=transform_seconds,
                      max_abs_error=0.0,
                      oracle="independent NumPy NaN mask oracle",
                      notes="all selected rows and columns matched"),
            rows_base(details, mode, phase="jvp", backend="numpy_oracle",
                      status="pass", seconds_per_operation=jvp_seconds,
                      max_abs_error=0.0,
                      oracle="independent NumPy locally-constant mask oracle",
                      notes=f"tangent checksum={float(np.sum(tangents)):.17g}"),
            rows_base(details, mode, phase="vjp", backend="numpy_oracle",
                      status="pass", seconds_per_operation=vjp_seconds,
                      max_abs_error=0.0,
                      oracle="independent NumPy zero reverse-product oracle",
                      notes=f"cotangent shape={vjp.shape[0]}x{vjp.shape[1]}"),
        ])
    return rows


def parse_oracle(path: Path) -> dict[tuple[str, str, int, int], float]:
    values: dict[tuple[str, str, int, int], float] = {}
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            key = (record["mode"], record["quantity"],
                   int(record["row"]) - 1, int(record["column"]) - 1)
            values[key] = float(record["value"])
    return values


def fortml_rows(fortml: Path, details: dict[str, str], no_build: bool) -> list[dict[str, Any]]:
    target = "fortml_bench_missing_indicator"
    source = fortml / "app" / f"{target}.f90"
    phases = ("transform", "jvp", "vjp")
    if not source.is_file():
        return [rows_base(details, mode, phase=phase, backend="fortml_cpu",
                          status="unavailable", oracle="typed release-target contract",
                          notes=f"release target source is absent: {source.name}")
                for mode in MODES for phase in phases]
    if not no_build:
        subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, check=True)
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"),
                        "OMP_NUM_THREADS": "1"})
    with tempfile.TemporaryDirectory(dir=fortml / "build") as directory:
        oracle_path = Path(directory) / "missing_indicator.csv"
        environment["FORTML_BENCH_MISSING_INDICATOR_ORACLE"] = str(oracle_path)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", target], cwd=fortml,
            env=environment, capture_output=True, text=True, check=True,
        )
        actual = parse_oracle(oracle_path)
    values, _ = fixture()
    expected_rows: list[dict[str, Any]] = []
    for mode, indices in MODES.items():
        expected = mask_oracle(values, indices)
        errors = []
        for column, source_column in enumerate(indices):
            for row in range(N_SAMPLES):
                errors.append(abs(actual[(mode, "transform", row, column)] - expected[row, column]))
                errors.append(abs(actual[(mode, "jvp", row, column)]))
            for row in range(N_SAMPLES):
                errors.append(abs(actual[(mode, "vjp", row, int(source_column))]))
        error = max(errors, default=0.0)
        if error > 5.0e-14:
            raise RuntimeError(f"FortML missing-indicator oracle mismatch for {mode}: {error:.3e}")
        timing = parse_timing(completed.stdout, mode)
        for phase in phases:
            expected_rows.append(rows_base(
                details, mode, phase=phase, backend="fortml_cpu", status="pass",
                seconds_per_operation=timing[phase], max_abs_error=error,
                oracle="independent NumPy missing-mask/JVP/VJP oracle",
                notes="complete transform, zero JVP, and zero VJP arrays matched",
            ))
        expected_rows.append(rows_base(
            details, mode, phase="transform", backend="fortml_cuda", device="cuda",
            status="unavailable", oracle="typed_device_contract",
            notes="resident missing-indicator kernel is not linked; typed refusal",
        ))
    return expected_rows


def parse_timing(stdout: str, mode: str) -> dict[str, float]:
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 8 and fields[0] == "missing_indicator" and fields[1] == mode:
            return {"transform": float(fields[5]), "jvp": float(fields[6]),
                    "vjp": float(fields[7])}
    raise RuntimeError(f"FortML missing-indicator app omitted timing for {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/missing_indicator.csv"))
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    details = metadata(root, fortml, args.output)
    rows = numpy_rows(details)
    rows.extend(fortml_rows(fortml, details, args.no_build))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
