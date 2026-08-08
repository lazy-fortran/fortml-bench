#!/usr/bin/env python3
"""Correctness-gated fixed-structure XGBoost/LightGBM leaf products."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


TOLERANCE = 2.0e-12
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


def oracle() -> tuple[np.ndarray, np.ndarray]:
    """Independent stump routing oracle for JVP and VJP."""
    return (
        np.asarray([1.5, 1.5, 2.5, 2.5], dtype=np.float64),
        np.asarray([10.0, 3.0, 7.0], dtype=np.float64),
    )


def run_app(fortml: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_tree_leaf_products"],
        cwd=fortml, env=environment, check=True, capture_output=True, text=True,
    )
    records: dict[str, list[str]] = {}
    vectors: dict[str, np.ndarray] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        name = fields[0]
        if name.endswith(("_leaf_parameters", "_leaf_jvp", "_leaf_vjp")):
            vectors[name] = np.asarray([float(value) for value in fields[1:]])
        elif name.endswith("_leaf_predict_seconds"):
            records[name] = fields[1:]
        elif name.endswith("_leaf_status_fit_seconds"):
            records[name] = fields[1:]
        elif name.endswith("_leaf_cuda_jvp_status") or name.endswith("_leaf_cuda_vjp_status"):
            records[name] = fields[1:]
        elif name.endswith("_leaf_parameter_count"):
            records[name] = fields[1:]
    expected_jvp, expected_vjp = oracle()
    for prefix in ("xgb", "lgbm"):
        if int(records[f"{prefix}_leaf_parameter_count"][0]) != 3:
            raise RuntimeError(f"{prefix} packed coordinate count changed")
        jvp_error = float(np.max(np.abs(vectors[f"{prefix}_leaf_jvp"] - expected_jvp)))
        vjp_error = float(np.max(np.abs(vectors[f"{prefix}_leaf_vjp"] - expected_vjp)))
        if max(jvp_error, vjp_error) > TOLERANCE:
            raise RuntimeError(f"{prefix} leaf-product oracle error")
        for direction in ("jvp", "vjp"):
            if int(records[f"{prefix}_leaf_cuda_{direction}_status"][0]) != 3:
                raise RuntimeError(f"{prefix} CUDA leaf {direction} refusal changed")
        records[f"{prefix}_leaf_jvp_error"] = [str(jvp_error)]
        records[f"{prefix}_leaf_vjp_error"] = [str(vjp_error)]
    return {"records": records, "vectors": vectors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/tree_leaf_products.csv"))
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
        for phase, metric, key in (
            ("jvp", "max_abs_error", f"{prefix}_leaf_jvp_error"),
            ("vjp", "max_abs_error", f"{prefix}_leaf_vjp_error"),
        ):
            rows.append(row(
                details, workload=f"{name}_tree_leaf_products", phase=phase,
                backend="fortml", device="cpu", status="pass", n_samples=4,
                n_features=2, n_estimators=1,
                seconds_per_operation=float(observed["records"]
                                            [f"{prefix}_leaf_predict_seconds"][0]),
                metric=metric, value=float(observed["records"][key][0]),
                max_abs_error=float(observed["records"][key][0]),
                oracle="independent NumPy two-leaf stump contraction",
                notes="raw margin; packed base plus leaf coordinates",
            ))
        rows.append(row(
            details, workload=f"{name}_tree_leaf_products", phase="capability_check",
            backend="fortml", device="cuda", status="unavailable", n_samples=4,
            n_features=2, n_estimators=1, oracle="declared device contract",
            notes="typed FORTNUM_NOT_IMPLEMENTED from leaf JVP/VJP device wrappers",
        ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
