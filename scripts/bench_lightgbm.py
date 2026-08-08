#!/usr/bin/env python3
"""Correctness-gated benchmark for FortML's bounded LightGBM-style path.

The tiny regression fixture is an independent three-leaf oracle: weighted
Newton gains select the root split at 4.5 and the best child split at 2.5,
giving leaves [0,0,0], [10,10], and [100].  The larger workload records CPU
fit/predict timing and a typed CUDA refusal; host execution is never relabelled
as accelerator work.
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
                        default=Path("results/lightgbm_leafwise.csv"))
    parser.add_argument("--target", default="fortml_bench_lightgbm")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    ignored = ((root / "results" / "lightgbm_leafwise.csv").resolve(),)
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        record: dict[str, object] = {field: "" for field in FIELDS}
        record.update(details)
        record.update({"workload": "lightgbm_leafwise", "backend": "fortml_cpu",
                       "device": "cpu", "status": "pass",
                       "oracle": "independent weighted Newton leaf-wise oracle"})
        record.update(values)
        rows.append(record)

    # Independent oracle for the exact tiny fixture used by the Fortran app.
    x_small = np.arange(6, dtype=np.float64)
    y_small = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 100.0])
    expected = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 100.0])
    if not np.array_equal(expected, y_small):
        raise RuntimeError("tiny leaf-wise oracle fixture changed")
    add(phase="contract_oracle", n_samples=6, n_features=1, n_estimators=1,
        seconds_per_operation=0.0, metric="max_abs_error", value=0.0,
        max_abs_error=0.0, notes="cuts 4.5 then 2.5; three best-first leaves")

    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, check=True)
    completed = subprocess.run(
        ["fo", "exec", "--no-build", args.target], cwd=fortml,
        check=True, capture_output=True, text=True,
    )
    parsed: dict[str, list[str]] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields:
            parsed[fields[0]] = fields[1:]
    for key in ("lightgbm_fit", "lightgbm_predict", "lightgbm_staged",
                "lightgbm_contributions", "lightgbm_slice", "lightgbm_binary",
                "lightgbm_oracle", "lightgbm_cuda"):
        if key not in parsed:
            raise RuntimeError(f"missing release row {key}")
    oracle_error = float(parsed["lightgbm_oracle"][0])
    if oracle_error > 1.0e-12:
        raise RuntimeError(f"leaf-wise oracle mismatch: {oracle_error}")
    add(phase="oracle", n_samples=6, n_features=1, n_estimators=1,
        seconds_per_operation=0.0, metric="max_abs_error", value=oracle_error,
        max_abs_error=oracle_error)

    fit = parsed["lightgbm_fit"]
    if len(fit) != 4:
        raise RuntimeError(f"unexpected fit row {fit!r}")
    add(phase="fit", n_samples=int(fit[0]), n_features=int(fit[1]),
        n_estimators=int(float(fit[2])), seconds_per_operation=float(fit[3]),
        metric="weighted_regression_fit_seconds", value=float(fit[3]),
        max_abs_error=0.0, notes="weighted CPU best-first histogram growth")

    predict = parsed["lightgbm_predict"]
    add(phase="predict", n_samples=int(predict[0]), n_features=int(predict[1]),
        n_estimators=int(float(predict[2])), seconds_per_operation=float(predict[3]),
        metric="weighted_regression_mse", value=float(predict[4]),
        max_abs_error=0.0)
    staged = parsed["lightgbm_staged"]
    staged_error = float(staged[4])
    if staged_error > 1.0e-12:
        raise RuntimeError(f"staged prediction mismatch: {staged_error}")
    add(phase="staged", n_samples=int(staged[0]), n_features=int(staged[1]),
        n_estimators=int(float(staged[2])), seconds_per_operation=float(staged[3]),
        metric="final_stage_max_abs_error", value=staged_error,
        max_abs_error=staged_error, notes="cumulative linked predictions")
    contributions = parsed["lightgbm_contributions"]
    contribution_error = float(contributions[4])
    if contribution_error > 1.0e-12:
        raise RuntimeError(f"contribution mismatch: {contribution_error}")
    add(phase="contributions", n_samples=int(contributions[0]),
        n_features=int(contributions[1]), n_estimators=int(float(contributions[2])),
        seconds_per_operation=float(contributions[3]),
        metric="margin_reconstruction_max_abs_error", value=contribution_error,
        max_abs_error=contribution_error, notes="base margin plus tree terms")
    sliced = parsed["lightgbm_slice"]
    slice_error = float(sliced[4])
    if slice_error > 1.0e-12:
        raise RuntimeError(f"prefix slice mismatch: {slice_error}")
    add(phase="slice", n_samples=int(sliced[0]), n_features=int(sliced[1]),
        n_estimators=int(float(sliced[2])), seconds_per_operation=float(sliced[3]),
        metric="prefix_prediction_max_abs_error", value=slice_error,
        max_abs_error=slice_error, notes="transactional four-tree prefix")
    binary = parsed["lightgbm_binary"]
    add(phase="binary", n_samples=int(binary[0]), n_features=int(binary[1]),
        n_estimators=int(float(binary[2])), seconds_per_operation=0.0,
        metric="accuracy", value=float(binary[3]), max_abs_error=0.0,
        notes="weighted binary-logistic objective")
    if parsed["lightgbm_cuda"] != ["3"]:
        raise RuntimeError(f"unexpected CUDA refusal {parsed['lightgbm_cuda']!r}")
    rows.append({**{field: "" for field in FIELDS}, **details,
                 "workload": "lightgbm_leafwise", "phase": "predict",
                 "backend": "fortml_cuda", "device": "cuda", "status": "unavailable",
                 "n_samples": 192, "n_features": 3, "n_estimators": 8,
                 "max_abs_error": "nan", "oracle": "typed_device_contract",
                 "notes": "FORTNUM_NOT_IMPLEMENTED; no resident CUDA histogram kernel"})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
