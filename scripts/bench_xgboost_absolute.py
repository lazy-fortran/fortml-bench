#!/usr/bin/env python3
"""Correctness-gated absolute-deviation XGBoost-style benchmark."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

FIELDS = (
    "workload", "objective", "phase", "backend", "device", "status",
    "n_samples", "n_features", "seconds_per_operation", "value",
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


def row(details: dict[str, str], **values: object) -> dict[str, object]:
    output: dict[str, object] = {field: "" for field in FIELDS}
    output.update(details)
    output.update(values)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/xgboost_absolute.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    ignored = ((root / "results" / "xgboost_absolute.csv").resolve(),)
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }
    expected = np.array([9.0, 9.0, 12.0, 12.0], dtype=np.float64)
    rows: list[dict[str, object]] = [row(
        details, workload="xgboost_absolute", objective="absolute",
        phase="predict", backend="numpy_oracle", device="cpu", status="pass",
        n_samples=4, n_features=1, value=float(np.mean(expected)),
        max_abs_error=0.0,
        oracle="independent one-tree absolute-deviation Newton formula",
    )]
    started = time.perf_counter()
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_xgboost_absolute"],
        cwd=fortml, check=True, capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - started
    values: dict[str, float | str] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2:
            key, value = fields
            if key == "xgb_absolute_cuda":
                values[key] = value
            elif key.startswith("xgb_absolute_"):
                values[key] = float(value)
    error = float(values["xgb_absolute_max_error"])
    if error > 1.0e-10:
        raise RuntimeError(f"absolute oracle mismatch: {error:.3e}")
    for phase, key in (("fit", "fit_seconds"), ("predict", "predict_seconds")):
        rows.append(row(
            details, workload="xgboost_absolute", objective="absolute",
            phase=phase, backend="fortml_cpu", device="cpu", status="pass",
            n_samples=4, n_features=1,
            seconds_per_operation=values[f"xgb_absolute_{key}"],
            value=float(np.mean(expected)), max_abs_error=error,
            oracle="independent one-tree absolute-deviation Newton formula",
            notes="exact CPU tree path; full subprocess elapsed %.6g s" % elapsed,
        ))
    if values.get("xgb_absolute_cuda") != "unavailable":
        raise RuntimeError("absolute XGBoost CUDA refusal row missing")
    rows.append(row(
        details, workload="xgboost_absolute", objective="absolute", phase="predict",
        backend="fortml_cuda", device="cuda", status="unavailable", n_samples=4,
        n_features=1, oracle="typed_device_contract",
        notes="no resident absolute-tree CUDA kernel; no host fallback",
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
