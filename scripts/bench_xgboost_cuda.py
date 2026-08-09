#!/usr/bin/env python3
"""Correctness-gated XGBoost resident CUDA policy benchmark.

The Fortran release app fits one finite numeric gbtree model and four models
whose CUDA routes are deliberately unsupported. Numeric native execution is
accepted only when it matches the CPU prediction. Categorical, missing,
ranking, and DART requests must return FORTNUM_NOT_IMPLEMENTED and preserve
their sentinel outputs. A typed refusal is never relabelled as a GPU timing.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_estimators", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/xgboost_cuda.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/XGBOOST_CUDA.md"))
    parser.add_argument("--target", default="fortml_bench_xgboost_cuda")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    report = args.report if args.report.is_absolute() else root / args.report
    ignored = (output.resolve(), report.resolve())
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
        "oracle": "independent CPU XGBoost prediction and typed policy contract",
    }
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, check=True)
    completed = subprocess.run(
        ["fo", "exec", "--no-build", args.target], cwd=fortml,
        check=True, capture_output=True, text=True,
    )
    observed: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) == 2:
            observed[fields[0]] = fields[1]
    required = (
        "xgb_cuda_numeric_status", "xgb_cuda_numeric_error",
        "xgb_cuda_categorical_status", "xgb_cuda_categorical_sentinel_error",
        "xgb_cuda_missing_status", "xgb_cuda_dart_status",
        "xgb_cuda_ranking_status",
    )
    missing = [key for key in required if key not in observed]
    if missing:
        raise RuntimeError(f"release app omitted rows: {missing}")

    numeric_status = int(observed["xgb_cuda_numeric_status"])
    numeric_error = float(observed["xgb_cuda_numeric_error"])
    sentinel_error = float(observed["xgb_cuda_categorical_sentinel_error"])
    unsupported = {
        "categorical": int(observed["xgb_cuda_categorical_status"]),
        "missing": int(observed["xgb_cuda_missing_status"]),
        "dart": int(observed["xgb_cuda_dart_status"]),
        "ranking": int(observed["xgb_cuda_ranking_status"]),
    }
    if numeric_status not in (0, 3):
        raise RuntimeError(f"unexpected numeric CUDA status {numeric_status}")
    if numeric_status == 0 and numeric_error > 2.0e-11:
        raise RuntimeError(f"numeric resident prediction mismatch {numeric_error}")
    if sentinel_error != 0.0:
        raise RuntimeError("categorical CUDA refusal changed sentinel output")
    if any(status != 3 for status in unsupported.values()):
        raise RuntimeError(f"unsupported policy status changed: {unsupported}")

    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        record: dict[str, object] = {field: "" for field in FIELDS}
        record.update(details)
        record.update({"workload": "xgboost_cuda", "backend": "fortml",
                       "device": "cuda", "n_samples": 8,
                       "n_features": 2, "n_estimators": 3})
        record.update(values)
        rows.append(record)

    add(phase="numeric_cpu_oracle", device="cpu", status="pass",
        metric="prediction_max_abs_error", value=numeric_error,
        max_abs_error=numeric_error,
        notes="same fitted numeric gbtree model used by device dispatch")
    if numeric_status == 0:
        add(phase="numeric_resident_prediction", status="pass",
            metric="prediction_max_abs_error", value=numeric_error,
            max_abs_error=numeric_error,
            notes="resident additive-tree CUDA path with no host fallback")
    else:
        add(phase="numeric_resident_prediction", status="unavailable",
            metric="status_code", value=numeric_status,
            max_abs_error="nan", oracle="typed capability contract",
            notes="FORTNUM_NOT_IMPLEMENTED; native CUDA plan unavailable")

    for policy, status in unsupported.items():
        add(phase=f"{policy}_refusal", status="pass", metric="status_code",
            value=status, max_abs_error=0.0,
            oracle="typed capability contract",
            notes="FORTNUM_NOT_IMPLEMENTED; no host fallback and output preserved")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# XGBoost resident CUDA policy\n\n"
        "This release lane exercises the finite numeric XGBoost `gbtree` "
        "path and its explicit device boundaries. Numeric native CUDA "
        "prediction is admitted only when it matches the independent CPU "
        "prediction. Categorical partitions, missing-default routing, "
        "ranking, and DART return `FORTNUM_NOT_IMPLEMENTED` before changing "
        "caller output. Those rows are capability evidence, not GPU timings. "
        "The resident additive-tree ABI keeps model arrays on the device and "
        "exposes transfer counters for query-only steady-state accounting.\n\n"
        "Run:\n\n```sh\n"
        "python3 scripts/bench_xgboost_cuda.py --fortml ../fortml\n```\n"
    )
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
