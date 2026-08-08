#!/usr/bin/env python3
"""Independent NumPy-gated multi-output XGBoost/LightGBM release lane.

The fixture is a one-tree, two-output Newton stump.  The oracle computes the
base score and each regularized leaf update directly from the target matrix;
it does not import FortML or inspect private tree state.  The release app
also gates staged margins, input/fixed-leaf products, transactionality,
metadata, and typed CUDA refusals.
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
    "n_features", "n_outputs", "n_estimators", "seconds_per_operation",
    "metric", "value", "max_abs_error", "oracle", "python_version",
    "numpy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
)
FORTNUM_DOMAIN_ERROR = 1
FORTNUM_NOT_IMPLEMENTED = 3


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            continue
    dirty = any(
        line[3:].split(" -> ")[-1].strip() not in ignored_names
        for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True,
        ).splitlines()
    )
    return head + ("+dirty" if dirty else "")


def run_app(fortml: Path) -> dict[str, list[float] | float | int]:
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1",
                        "FO_SCAN_FALLBACK": "regex"})
    subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_xgboost_multioutput"],
        cwd=fortml, env=environment, check=True, capture_output=True, text=True,
    )
    parsed: dict[str, list[float] | float | int] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        key = fields[0]
        try:
            numbers = [float(token) for token in fields[1:]]
        except ValueError:
            continue
        if key.endswith("_status") or key.endswith("_count"):
            parsed[key] = int(numbers[0])
        elif len(numbers) == 1:
            parsed[key] = numbers[0]
        else:
            parsed[key] = numbers
    required = {
        "xgb_predict", "xgb_staged", "xgb_leaf_jvp", "xgb_leaf_vjp",
        "xgb_predict_status", "xgb_stage_status", "xgb_jvp_status",
        "xgb_vjp_status", "xgb_cuda_status", "xgb_cuda_margin_status",
        "xgb_malformed_status", "xgb_transaction_error", "xgb_output_count",
        "xgb_parameter_count", "lgb_predict", "lgb_staged", "lgb_leaf_jvp",
        "lgb_leaf_vjp", "lgb_predict_status", "lgb_stage_status",
        "lgb_jvp_status", "lgb_vjp_status", "lgb_cuda_status", "lgb_output_count",
        "lgb_parameter_count",
    }
    missing = required - parsed.keys()
    if missing:
        raise RuntimeError(f"release app omitted {sorted(missing)}")
    return parsed


def oracle() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    targets = np.asarray(((0.0, 1.0), (0.0, 1.0), (10.0, 5.0), (10.0, 5.0)))
    base = np.mean(targets, axis=0)
    gradient = base[None, :] - targets
    left = -np.sum(gradient[:2, :], axis=0) / 3.0
    right = -np.sum(gradient[2:, :], axis=0) / 3.0
    expected = np.vstack((base + left, base + left, base + right, base + right))
    expected_jvp = np.asarray(((3.0, 9.0), (3.0, 9.0), (4.0, 10.0), (4.0, 10.0)))
    expected_vjp = np.asarray((10.0, 3.0, 7.0, 14.0, 5.0, 9.0))
    return expected, expected.copy(), expected_jvp, expected_vjp


def as_matrix(values: list[float]) -> np.ndarray:
    # Fortran writes rank-2/rank-3 arrays in column-major order.
    return np.asarray(values, dtype=np.float64).reshape((2, 4)).T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/xgboost_multioutput.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    observed = run_app(fortml)
    expected, expected_staged, expected_jvp, expected_vjp = oracle()
    oracle_name = "independent NumPy two-output one-tree Newton stump"
    errors: dict[str, float] = {}
    for backend in ("xgb", "lgb"):
        prediction = as_matrix(observed[f"{backend}_predict"])  # type: ignore[arg-type]
        staged = as_matrix(observed[f"{backend}_staged"])  # type: ignore[arg-type]
        jvp = as_matrix(observed[f"{backend}_leaf_jvp"])  # type: ignore[arg-type]
        vjp = np.asarray(observed[f"{backend}_leaf_vjp"], dtype=np.float64)
        errors[f"{backend}_predict"] = float(np.max(np.abs(prediction - expected)))
        errors[f"{backend}_staged"] = float(np.max(np.abs(staged - expected_staged)))
        errors[f"{backend}_jvp"] = float(np.max(np.abs(jvp - expected_jvp)))
        errors[f"{backend}_vjp"] = float(np.max(np.abs(vjp - expected_vjp)))
        if any(errors[key] > 3.0e-13 for key in errors if key.startswith(backend)):
            raise RuntimeError(f"{backend} independent oracle mismatch: {errors}")
    for backend in ("xgb", "lgb"):
        for key in ("predict_status", "stage_status", "jvp_status", "vjp_status"):
            if int(observed[f"{backend}_{key}"]) != 0:
                raise RuntimeError(f"{backend} {key} failed")
        if int(observed[f"{backend}_output_count"]) != 2 or \
                int(observed[f"{backend}_parameter_count"]) != 6:
            raise RuntimeError(f"{backend} metadata changed")
    if int(observed["xgb_cuda_status"]) != FORTNUM_NOT_IMPLEMENTED or \
            int(observed["xgb_cuda_margin_status"]) != FORTNUM_NOT_IMPLEMENTED or \
            int(observed["lgb_cuda_status"]) != FORTNUM_NOT_IMPLEMENTED:
        raise RuntimeError("typed CUDA refusal contract changed")
    if int(observed["xgb_malformed_status"]) != FORTNUM_DOMAIN_ERROR or \
            float(observed["xgb_transaction_error"]) > 0.0:
        raise RuntimeError("transactional malformed-fit contract changed")

    fortml_rev = revision(fortml)
    bench_rev = revision(root, (args.output.resolve(),))
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": fortml_rev, "benchmark_revision": bench_rev,
        "compiler": "gfortran", "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(workload: str, phase: str, backend: str, device: str, status: str,
            metric: str, value: float, error: float | str, seconds: float = 0.0,
            notes: str = "") -> None:
        row = {field: "" for field in FIELDS}
        row.update({
            "workload": workload, "phase": phase, "backend": backend,
            "device": device, "status": status, "n_samples": 4,
            "n_features": 1, "n_outputs": 2, "n_estimators": 1,
            "seconds_per_operation": seconds, "metric": metric, "value": value,
            "max_abs_error": error, "oracle": oracle_name, **details,
            "notes": notes,
        })
        rows.append(row)

    for backend in ("xgb", "lgb"):
        prefix = "xgb" if backend == "xgb" else "lgb"
        add(f"{backend}_multioutput", "predict", "fortml", "cpu", "pass",
            "max_abs_error", 0.0, errors[f"{prefix}_predict"],
            float(observed[f"{prefix}_fit_seconds"]) if f"{prefix}_fit_seconds" in observed else 0.0,
            "closed-form two-output stump prediction")
        add(f"{backend}_multioutput", "staged_margin", "fortml", "cpu", "pass",
            "max_abs_error", 0.0, errors[f"{prefix}_staged"],
            notes="(sample,stage,output) stage tensor")
        add(f"{backend}_multioutput", "leaf_jvp", "fortml", "cpu", "pass",
            "max_abs_error", 0.0, errors[f"{prefix}_jvp"],
            notes="fixed topology, concatenated output leaf coordinates")
        add(f"{backend}_multioutput", "leaf_vjp", "fortml", "cpu", "pass",
            "max_abs_error", 0.0, errors[f"{prefix}_vjp"],
            notes="independent adjoint contraction")
        add(f"{backend}_multioutput", "predict", "fortml", "cuda", "unavailable",
            "prediction", float("nan"), "nan", notes="FORTNUM_NOT_IMPLEMENTED; no host fallback")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"maximum independent-oracle error: {max(errors.values()):.3e}")


if __name__ == "__main__":
    main()
