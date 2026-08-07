#!/usr/bin/env python3
"""Correctness-gated monotonic XGBoost-style tree benchmark.

The Fortran executable reports complete query vectors.  This harness computes
the monotonicity violation independently with NumPy, so a scalar emitted by
the application cannot make a malformed tree pass.  CUDA rows are explicit
typed refusals: no resident CUDA tree kernel is linked, and host predictions
are never relabeled as GPU timings.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


N_SAMPLES = 192
N_FEATURES = 2
N_ESTIMATORS = 8
N_QUERY = 257
TOLERANCE = 2.0e-12

FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_estimators", "constraint", "seconds_per_operation",
    "metric", "value", "max_abs_error", "oracle", "python_version",
    "numpy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
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


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def parse_output(stdout: str) -> dict[str, list[float]]:
    parsed: dict[str, list[float]] = {}
    for line in stdout.splitlines():
        if line.startswith("xgb_monotone_") and "_values " in line:
            name, values = line.split(None, 1)
            parsed[name] = [float(value) for value in values.split()]
    expected = {"xgb_monotone_exact_values", "xgb_monotone_hist_values"}
    missing = expected - parsed.keys()
    if missing:
        raise RuntimeError(f"FortML monotonic app omitted {sorted(missing)}")
    for name, values in parsed.items():
        if len(values) != N_QUERY or not np.all(np.isfinite(values)):
            raise RuntimeError(f"{name} has malformed query values")
    return parsed


def run_fortran(fortml: Path, details: dict[str, str]) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_xgboost"], cwd=fortml,
        env=environment, check=True, capture_output=True, text=True,
    )
    vectors = parse_output(completed.stdout)
    records: list[dict[str, Any]] = []
    cases = (
        ("exact", 1, "xgb_monotone_exact_values"),
        ("hist", -1, "xgb_monotone_hist_values"),
    )
    lines = completed.stdout.splitlines()
    for method, constraint, key in cases:
        vector = np.asarray(vectors[key], dtype=np.float64)
        if constraint > 0:
            violation = float(np.max(np.maximum(vector[:-1] - vector[1:], 0.0)))
        else:
            violation = float(np.max(np.maximum(vector[1:] - vector[:-1], 0.0)))
        if violation > TOLERANCE:
            raise RuntimeError(
                f"FortML {method} monotone oracle violation {violation:.3e}"
            )
        for phase, prefix in (("fit", f"xgb_monotone_{method}_fit"),
                              ("predict", f"xgb_monotone_{method}_predict")):
            fields = next(
                line.split(",") for line in lines if line.startswith(prefix + ",")
            )
            records.append(row(
                details,
                workload=f"xgboost_monotone_{method}", phase=phase,
                backend="fortml", device="cpu", status="pass",
                n_samples=N_SAMPLES, n_features=N_FEATURES,
                n_estimators=N_ESTIMATORS, constraint=constraint,
                seconds_per_operation=float(fields[4]),
                metric="monotonicity_violation", value=violation,
                max_abs_error=violation,
                oracle="independent NumPy query-vector monotonicity oracle",
                notes=("exact recursive split bounds" if method == "exact" else
                       "weighted-quantile histogram split bounds"),
            ))
    return records


def unsupported_rows(details: dict[str, str]) -> list[dict[str, Any]]:
    return [
        row(
            details,
            workload="xgboost_monotone_exact", phase="capability_check",
            backend="fortml", device="cuda", status="unavailable",
            n_samples=N_SAMPLES, n_features=N_FEATURES,
            n_estimators=N_ESTIMATORS, constraint=1,
            oracle="declared device contract",
            notes="no resident CUDA XGBoost tree kernel is linked; typed refusal",
        ),
        row(
            details,
            workload="xgboost_monotone_hist", phase="capability_check",
            backend="fortml", device="cuda", status="unavailable",
            n_samples=N_SAMPLES, n_features=N_FEATURES,
            n_estimators=N_ESTIMATORS, constraint=-1,
            oracle="declared device contract",
            notes="no resident CUDA histogram kernel is linked; typed refusal",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/xgboost_monotonic_constraints.csv"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    details = metadata(root, fortml, args.output.resolve())
    records = run_fortran(fortml, details)
    records.extend(unsupported_rows(details))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()
