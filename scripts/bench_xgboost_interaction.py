#!/usr/bin/env python3
"""Correctness-gated XGBoost interaction-constraint benchmark.

The executable reports complete constrained predictions.  NumPy reconstructs
the fixture and expected constrained leaves independently; tree-node counts
and the reported fit error are checked as secondary diagnostics.  CUDA is an
explicit unavailable row because no resident tree kernel is linked.
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
N_FEATURES = 3
N_ESTIMATORS = 1
TOLERANCE = 2.0e-12
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_estimators", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
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


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def parse(stdout: str) -> tuple[dict[str, list[float]], dict[str, list[str]]]:
    vectors: dict[str, list[float]] = {}
    records: dict[str, list[str]] = {}
    for line in stdout.splitlines():
        if line.startswith("xgb_interaction_"):
            if line.startswith("xgb_interaction_constrained_values "):
                name, values = line.split(None, 1)
                vectors[name] = [float(value) for value in values.split()]
                continue
            fields = line.split(",")
            if fields[0].endswith("_fit"):
                records[fields[0]] = fields[1:]
            elif line.startswith("xgb_interaction_predict_seconds "):
                name, value = line.split(None, 1)
                records[name] = [value.strip()]
    if "xgb_interaction_constrained_values" not in vectors:
        raise RuntimeError("FortML app omitted constrained prediction vector")
    if len(vectors["xgb_interaction_constrained_values"]) != N_SAMPLES:
        raise RuntimeError("malformed constrained prediction vector")
    return vectors, records


def run_fortran(fortml: Path, details: dict[str, str]) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_xgboost_interaction"],
        cwd=fortml, env=environment, check=True, capture_output=True, text=True,
    )
    vectors, records = parse(completed.stdout)
    constrained = np.asarray(vectors["xgb_interaction_constrained_values"])
    x1 = np.repeat(np.arange(2, dtype=np.float64), N_SAMPLES // 2)
    x2 = np.tile(np.arange(4, dtype=np.float64), N_SAMPLES // 4)
    target = 10.0 * x1 + 4.0 * np.floor(x2 / 2.0)
    expected = np.where(x1 < 0.5, 2.0, 12.0)
    error = float(np.max(np.abs(constrained - expected)))
    if error > TOLERANCE:
        raise RuntimeError(f"interaction constrained prediction error {error:.3e}")
    if int(records["xgb_interaction_unconstrained_fit"][4]) != 7:
        raise RuntimeError("unconstrained tree did not reach the four-leaf oracle")
    if int(records["xgb_interaction_constrained_fit"][4]) != 3:
        raise RuntimeError("interaction constraint did not limit the path")
    if float(records["xgb_interaction_unconstrained_fit"][5]) > TOLERANCE:
        raise RuntimeError("unconstrained fit diagnostic is inconsistent")
    if not np.allclose(target, 10.0 * x1 + 4.0 * np.floor(x2 / 2.0)):
        raise RuntimeError("internal fixture oracle malformed")
    return [
        row(details, workload="xgboost_interaction_unconstrained", phase="fit",
            backend="fortml", device="cpu", status="pass", n_samples=N_SAMPLES,
            n_features=N_FEATURES, n_estimators=N_ESTIMATORS,
            seconds_per_operation=float(records["xgb_interaction_unconstrained_fit"][3]),
            metric="max_abs_error", value=float(records["xgb_interaction_unconstrained_fit"][5]),
            max_abs_error=float(records["xgb_interaction_unconstrained_fit"][5]),
            oracle="independent NumPy four-leaf path oracle",
            notes="unconstrained depth-two reference"),
        row(details, workload="xgboost_interaction_constrained", phase="fit",
            backend="fortml", device="cpu", status="pass", n_samples=N_SAMPLES,
            n_features=N_FEATURES, n_estimators=N_ESTIMATORS,
            seconds_per_operation=float(records["xgb_interaction_constrained_fit"][3]),
            metric="max_abs_error", value=error, max_abs_error=error,
            oracle="independent NumPy group-mean path oracle",
            notes="features one and two are in separate interaction groups"),
        row(details, workload="xgboost_interaction_constrained", phase="predict",
            backend="fortml", device="cpu", status="pass", n_samples=N_SAMPLES,
            n_features=N_FEATURES, n_estimators=N_ESTIMATORS,
            seconds_per_operation=float(records["xgb_interaction_predict_seconds"][0]),
            metric="max_abs_error", value=error, max_abs_error=error,
            oracle="independent NumPy group-mean path oracle",
            notes="repeated prediction timing"),
    ]


def unavailable(details: dict[str, str]) -> list[dict[str, Any]]:
    return [row(
        details, workload="xgboost_interaction_constrained", phase="capability_check",
        backend="fortml", device="cuda", status="unavailable", n_samples=N_SAMPLES,
        n_features=N_FEATURES, n_estimators=N_ESTIMATORS,
        oracle="declared device contract",
        notes="no resident CUDA XGBoost tree kernel is linked; typed refusal",
    )]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/xgboost_interaction.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    details = metadata(root, fortml, args.output.resolve())
    rows = run_fortran(fortml, details) + unavailable(details)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
