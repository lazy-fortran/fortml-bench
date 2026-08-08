#!/usr/bin/env python3
"""Correctness-gated bounded SHAP-like tree attribution benchmark.

The fixture is a one-split regression stump.  NumPy computes the expected
path baseline and leaf corrections independently, then checks complete
FortML output vectors, additivity, and the typed CUDA refusal for both the
XGBoost-style and LightGBM-style estimators.
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


N_SAMPLES = 4
N_FEATURES = 2
N_ESTIMATORS = 1
TOLERANCE = 2.0e-12
FORTNUM_NOT_IMPLEMENTED = 3
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
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
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


def oracle() -> np.ndarray:
    """Return baseline plus feature values for the one-stump fixture."""
    expected = np.zeros((N_SAMPLES, N_FEATURES + 1), dtype=np.float64)
    expected[:, 0] = 5.0
    expected[:2, 1] = -10.0 / 3.0
    expected[2:, 1] = 10.0 / 3.0
    return expected


def run_app(fortml: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_tree_shap"],
        cwd=fortml, env=environment, check=True, capture_output=True, text=True,
    )
    records: dict[str, list[str]] = {}
    vectors: dict[str, np.ndarray] = {}
    for line in completed.stdout.splitlines():
        if line.startswith("xgb_shap_values ") or line.startswith("lgbm_shap_values "):
            name, values = line.split(None, 1)
            vectors[name] = np.fromstring(values, sep=" ")
        elif line.startswith(("xgb_shap_fit,", "lgbm_shap_fit,", "xgb_shap_cuda,",
                              "lgbm_shap_cuda,")):
            fields = [field.strip() for field in line.split(",")]
            records[fields[0]] = fields[1:]
        elif line.startswith(("xgb_shap_predict_seconds ", "lgbm_shap_predict_seconds ")):
            name, value = line.split(None, 1)
            records[name] = [value.strip()]
    expected = oracle()
    output: dict[str, Any] = {"records": records, "vectors": vectors, "expected": expected}
    for prefix in ("xgb", "lgbm"):
        observed = vectors[f"{prefix}_shap_values"]
        if observed.size != expected.size:
            raise RuntimeError(f"{prefix} SHAP vector has wrong size")
        observed = observed.reshape(expected.shape, order="F")
        error = float(np.max(np.abs(observed - expected)))
        if error > TOLERANCE:
            raise RuntimeError(f"{prefix} SHAP oracle error {error:.3e}")
        if not np.allclose(observed.sum(axis=1), np.asarray((5.0 - 10.0/3.0,
                                                               5.0 - 10.0/3.0,
                                                               5.0 + 10.0/3.0,
                                                               5.0 + 10.0/3.0)),
                           atol=TOLERANCE, rtol=0.0):
            raise RuntimeError(f"{prefix} SHAP additivity oracle failed")
        output[f"{prefix}_error"] = error
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/TREE_SHAP.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    observed = run_app(fortml)
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (args.output.resolve(),)),
        "compiler": "gfortran",
        "flags": "-O3",
    }
    rows: list[dict[str, Any]] = []
    for prefix, name in (("xgb", "xgboost"), ("lgbm", "lightgbm")):
        fit = observed["records"][f"{prefix}_shap_fit"]
        rows.append(row(details, workload=f"{name}_tree_shap", phase="fit",
                        backend="fortml", device="cpu", status="pass",
                        n_samples=N_SAMPLES, n_features=N_FEATURES,
                        n_estimators=N_ESTIMATORS,
                        seconds_per_operation=float(fit[3]), metric="max_abs_error",
                        value=observed[f"{prefix}_error"],
                        max_abs_error=observed[f"{prefix}_error"],
                        oracle="independent NumPy one-stump attribution",
                        notes="baseline plus exact subset Shapley path values"))
        rows.append(row(details, workload=f"{name}_tree_shap", phase="predict",
                        backend="fortml", device="cpu", status="pass",
                        n_samples=N_SAMPLES, n_features=N_FEATURES,
                        n_estimators=N_ESTIMATORS,
                        seconds_per_operation=float(observed["records"]
                                                    [f"{prefix}_shap_predict_seconds"][0]),
                        metric="max_abs_error", value=observed[f"{prefix}_error"],
                        max_abs_error=observed[f"{prefix}_error"],
                        oracle="independent NumPy one-stump attribution",
                        notes="repeated prediction timing"))
        rows.append(row(details, workload=f"{name}_tree_shap", phase="capability_check",
                        backend="fortml", device="cuda", status="unavailable",
                        n_samples=N_SAMPLES, n_features=N_FEATURES,
                        n_estimators=N_ESTIMATORS, oracle="declared device contract",
                        notes="no resident CUDA tree explanation kernel; typed refusal"))
        if int(observed["records"][f"{prefix}_shap_cuda"][0]) != FORTNUM_NOT_IMPLEMENTED:
            raise RuntimeError(f"{prefix} CUDA refusal status changed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} ({len(rows)} rows; max error "
          f"{max(observed['xgb_error'], observed['lgbm_error']):.3e})")


if __name__ == "__main__":
    main()
