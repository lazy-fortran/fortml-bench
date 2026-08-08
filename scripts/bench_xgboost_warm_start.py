#!/usr/bin/env python3
"""Correctness-gated benchmark for deterministic XGBoost warm starts.

The NumPy oracle independently replays the four depth-one Newton stumps.  A
warm continuation is accepted only when its fourth staged margin matches a
fresh four-tree fit and the typed transactional refusals leave that fitted
prefix intact.
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
N_ESTIMATORS = 4
LEARNING_RATE = 0.2
L2 = 1.0
FORTNUM_DOMAIN_ERROR = 1

FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_estimators", "seconds_per_operation", "metric", "value",
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


def oracle() -> np.ndarray:
    x = np.arange(N_SAMPLES, dtype=np.float64)
    target = np.asarray((0, 0, 0, 0, 10, 10, 10, 10), dtype=np.float64)
    margin = np.full(N_SAMPLES, np.mean(target))
    stages: list[np.ndarray] = []
    for _ in range(N_ESTIMATORS):
        gradient = margin - target
        total_score = float(np.sum(gradient)) ** 2 / (N_SAMPLES + L2)
        best_gain = 0.0
        best_prediction = np.full(N_SAMPLES, -float(np.sum(gradient)) / (N_SAMPLES + L2))
        for threshold in 0.5 * (x[:-1] + x[1:]):
            left = x < threshold
            right = ~left
            left_sum = float(np.sum(gradient[left]))
            right_sum = float(np.sum(gradient[right]))
            gain = 0.5 * (
                left_sum**2 / (np.count_nonzero(left) + L2)
                + right_sum**2 / (np.count_nonzero(right) + L2)
                - total_score
            )
            if gain > best_gain:
                best_gain = gain
                best_prediction = np.where(
                    left,
                    -left_sum / (np.count_nonzero(left) + L2),
                    -right_sum / (np.count_nonzero(right) + L2),
                )
        margin = margin + LEARNING_RATE * best_prediction
        stages.append(margin.copy())
    return np.stack(stages, axis=1)


def parse(stdout: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    vectors = {"xgb_warm_staged_4", "xgb_full_staged_4"}
    for line in stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] in vectors:
            values[fields[0]] = np.asarray([float(v) for v in fields[1:]])
        elif fields[0] in {
            "xgb_warm_invalid_target_status", "xgb_warm_changed_control_status",
            "xgb_warm_unfitted_status", "xgb_warm_estimator_count",
            "xgb_warm_requested_count",
        }:
            values[fields[0]] = int(fields[1])
        elif fields[0] == "xgb_warm_seconds":
            values[fields[0]] = float(fields[1])
    required = vectors | {
        "xgb_warm_invalid_target_status", "xgb_warm_changed_control_status",
        "xgb_warm_unfitted_status", "xgb_warm_estimator_count",
        "xgb_warm_requested_count", "xgb_warm_seconds",
    }
    missing = required - values.keys()
    if missing:
        raise RuntimeError(f"release app omitted {sorted(missing)}")
    if any(values[name].size != N_SAMPLES for name in vectors):
        raise RuntimeError("release app emitted malformed staged margins")
    return values


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
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/xgboost_warm_start.csv"),
    )
    parser.add_argument("--target", default="fortml_bench_xgboost_warm_start")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (args.output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }
    observed = run(fortml, args.target)
    expected = oracle()[:, -1]
    warm_error = float(np.max(np.abs(observed["xgb_warm_staged_4"] - expected)))
    full_error = float(np.max(np.abs(observed["xgb_full_staged_4"] - expected)))
    match_error = float(np.max(np.abs(
        observed["xgb_warm_staged_4"] - observed["xgb_full_staged_4"])))
    if max(warm_error, full_error, match_error) > 2.0e-12:
        raise RuntimeError(
            f"warm-start oracle mismatch: warm={warm_error:.3e}, "
            f"full={full_error:.3e}, match={match_error:.3e}"
        )
    for name in (
        "xgb_warm_invalid_target_status", "xgb_warm_changed_control_status",
        "xgb_warm_unfitted_status",
    ):
        if observed[name] != FORTNUM_DOMAIN_ERROR:
            raise RuntimeError(f"{name} changed: {observed[name]}")
    if observed["xgb_warm_estimator_count"] != N_ESTIMATORS or \
            observed["xgb_warm_requested_count"] != N_ESTIMATORS:
        raise RuntimeError("warm-start estimator metadata changed")
    records = [
        row(details, workload="xgboost_warm_start", phase="continuation",
            backend="fortml", device="cpu", status="pass", n_samples=N_SAMPLES,
            n_estimators=N_ESTIMATORS,
            seconds_per_operation=observed["xgb_warm_seconds"],
            metric="fourth_stage_max_abs_error", value=warm_error,
            max_abs_error=warm_error, oracle="independent NumPy stump replay",
            notes="warm suffix matches fresh fit and independent Newton oracle"),
        row(details, workload="xgboost_warm_start", phase="metadata",
            backend="fortml", device="cpu", status="pass", n_samples=N_SAMPLES,
            n_estimators=N_ESTIMATORS, metric="warm_full_max_abs_difference",
            value=match_error, max_abs_error=match_error,
            oracle="independent NumPy stump replay", notes="staged margins agree"),
    ]
    for phase, name in (
        ("invalid_target", "xgb_warm_invalid_target_status"),
        ("changed_control", "xgb_warm_changed_control_status"),
        ("unfitted_source", "xgb_warm_unfitted_status"),
    ):
        records.append(row(
            details, workload="xgboost_warm_start", phase=phase,
            backend="fortml", device="cpu", status="pass", n_samples=N_SAMPLES,
            n_estimators=N_ESTIMATORS, metric="status_code", value=observed[name],
            max_abs_error=0.0, oracle="typed transactional domain contract",
            notes="refusal leaves the fitted prefix unchanged",
        ))
    records.append(row(
        details, workload="xgboost_warm_start", phase="device_contract",
        backend="fortml", device="cuda", status="unavailable", n_samples=N_SAMPLES,
        n_estimators=N_ESTIMATORS, metric="api_surface", value="unavailable",
        max_abs_error=0.0, oracle="device capability contract",
        notes="warm-start continuation has no resident CUDA entry point",
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()
