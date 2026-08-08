#!/usr/bin/env python3
"""Correctness-gated benchmark for fitted XGBoost prefix slicing.

The NumPy oracle independently replays the one-dimensional squared-loss
stumps in the release fixture.  A slice is accepted only when its prediction
equals the oracle's second staged prefix and differs from the complete
ensemble; the invalid-prefix status is checked separately.
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


N_SAMPLES = 8
N_ESTIMATORS = 3
SLICE_TREES = 2
LEARNING_RATE = 0.5
FORTNUM_DOMAIN_ERROR = 1

FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_estimators", "slice_trees", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


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


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def parse(stdout: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    vectors = {
        "xgb_slice_staged_1", "xgb_slice_staged_2", "xgb_slice_staged_3",
        "xgb_slice_full_prediction", "xgb_slice_prefix_prediction",
    }
    for line in stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        name = fields[0]
        if name in vectors:
            parsed[name] = np.asarray([float(value) for value in fields[1:]])
        elif name == "xgb_slice_invalid_status":
            parsed[name] = int(fields[1])
        elif name in {"xgb_slice_seconds", "xgb_slice_predict_seconds"}:
            parsed[name] = float(fields[1])
    required = vectors | {
        "xgb_slice_invalid_status", "xgb_slice_seconds", "xgb_slice_predict_seconds",
    }
    missing = required - parsed.keys()
    if missing:
        raise RuntimeError(f"release app omitted {sorted(missing)}")
    if any(parsed[name].size != N_SAMPLES for name in vectors):
        raise RuntimeError("release app emitted malformed slice vectors")
    return parsed


def oracle() -> np.ndarray:
    x = np.arange(N_SAMPLES, dtype=np.float64)
    target = np.asarray((0, 0, 0, 0, 10, 10, 10, 10), dtype=np.float64)
    margin = np.full(N_SAMPLES, np.mean(target))
    stages: list[np.ndarray] = []
    for _ in range(N_ESTIMATORS):
        gradient = margin - target
        total_score = float(np.sum(gradient))**2/N_SAMPLES
        best_gain = 0.0
        best_prediction = np.full(N_SAMPLES, -float(np.mean(gradient)))
        for threshold in 0.5*(x[:-1] + x[1:]):
            left = x < threshold
            right = ~left
            left_weight = -float(np.sum(gradient[left]))/float(np.count_nonzero(left))
            right_weight = -float(np.sum(gradient[right]))/float(np.count_nonzero(right))
            gain = 0.5*(
                float(np.sum(gradient[left]))**2/float(np.count_nonzero(left))
                + float(np.sum(gradient[right]))**2/float(np.count_nonzero(right))
                - total_score
            )
            if gain > best_gain:
                best_gain = gain
                best_prediction = np.where(left, left_weight, right_weight)
        margin = margin + LEARNING_RATE*best_prediction
        stages.append(margin.copy())
    return np.stack(stages, axis=1)


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def run(fortml: Path, target: str) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    completed = subprocess.run(
        ["fo", "exec", "--no-build", target], cwd=fortml, env=environment,
        check=True, capture_output=True, text=True,
    )
    return parse(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/xgboost_slice.csv"))
    parser.add_argument("--target", default="fortml_bench_xgboost_slice")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    details = metadata(root, fortml, args.output.resolve())
    observed = run(fortml, args.target)
    expected = oracle()
    prefix_error = float(np.max(np.abs(observed["xgb_slice_prefix_prediction"] - expected[:, 1])))
    staged_error = float(np.max(np.abs(observed["xgb_slice_staged_2"] - expected[:, 1])))
    full_error = float(np.max(np.abs(observed["xgb_slice_full_prediction"] - expected[:, -1])))
    distinct = float(np.max(np.abs(
        observed["xgb_slice_full_prediction"] - observed["xgb_slice_prefix_prediction"])))
    if max(prefix_error, staged_error, full_error) > 2.0e-12:
        raise RuntimeError(
            f"slice oracle mismatch: prefix={prefix_error:.3e}, "
            f"staged={staged_error:.3e}, full={full_error:.3e}"
        )
    if distinct < 1.0e-8:
        raise RuntimeError("slice fixture did not retain a distinct prefix")
    if observed["xgb_slice_invalid_status"] != FORTNUM_DOMAIN_ERROR:
        raise RuntimeError("invalid-prefix refusal changed")
    records = [
        row(details, workload="xgboost_slice", phase="slice", backend="fortml",
            device="cpu", status="pass", n_samples=N_SAMPLES,
            n_estimators=N_ESTIMATORS, slice_trees=SLICE_TREES,
            seconds_per_operation=observed["xgb_slice_seconds"],
            metric="prefix_prediction_max_abs_error", value=prefix_error,
            max_abs_error=prefix_error, oracle="independent NumPy stump replay",
            notes="slice prediction equals second staged prefix"),
        row(details, workload="xgboost_slice", phase="predict", backend="fortml",
            device="cpu", status="pass", n_samples=N_SAMPLES,
            n_estimators=N_ESTIMATORS, slice_trees=SLICE_TREES,
            seconds_per_operation=observed["xgb_slice_predict_seconds"],
            metric="full_prefix_max_abs_difference", value=distinct,
            max_abs_error=full_error, oracle="independent NumPy stump replay",
            notes="prefix and complete ensemble remain distinct"),
        row(details, workload="xgboost_slice", phase="invalid_prefix", backend="fortml",
            device="cpu", status="pass", n_samples=N_SAMPLES,
            n_estimators=N_ESTIMATORS, slice_trees=0, metric="status_code",
            value=observed["xgb_slice_invalid_status"], max_abs_error=0.0,
            oracle="typed domain contract", notes="zero-length prefix is refused"),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()
