#!/usr/bin/env python3
"""Correctness-gated multiclass SAMME.R probability-update benchmark."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


N_SAMPLES = 9
N_FEATURES = 1
N_CLASSES = 3
N_QUERY = 5
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_classes", "n_query", "n_estimators",
    "seconds_per_operation", "metric", "value", "max_abs_error", "oracle",
    "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
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


def oracle() -> tuple[np.ndarray, np.ndarray]:
    """Replay the one-stump SAMME.R geometric probability policy."""
    epsilon = 1.0e-12
    base = np.empty((N_QUERY, N_CLASSES), dtype=np.float64)
    base[0, :] = [1.0, epsilon, epsilon]
    base[1:, :] = [epsilon, 0.5, 0.5]
    probabilities = base / np.sum(base, axis=1, keepdims=True)
    predicted = np.array([-7, 4, 4, 4, 4], dtype=np.int64)
    return probabilities, predicted


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def parse(stdout: str) -> tuple[dict[str, float], np.ndarray, np.ndarray, float, int, int]:
    timing: dict[str, float] = {}
    probability_values: np.ndarray | None = None
    prediction_values: np.ndarray | None = None
    stage_weight: float | None = None
    stage_count: int | None = None
    cuda_code: int | None = None
    for line in stdout.splitlines():
        if line.startswith("sammer_fit,") or line.startswith("sammer_predict,"):
            fields = line.split(",")
            timing[fields[0]] = float(fields[-1])
        elif line.startswith("sammer_probability_values "):
            probability_values = np.fromstring(line.split(None, 1)[1], sep=" ")
        elif line.startswith("sammer_prediction_values "):
            prediction_values = np.fromstring(
                line.split(None, 1)[1], sep=" ").astype(np.int64)
        elif line.startswith("sammer_stage_weight,"):
            stage_weight = float(line.split(",", 1)[1])
        elif line.startswith("sammer_stage_count,"):
            stage_count = int(line.split(",", 1)[1])
        elif line.startswith("sammer_cuda_code,"):
            cuda_code = int(line.split(",", 1)[1])
    if set(timing) != {"sammer_fit", "sammer_predict"}:
        raise RuntimeError("release app omitted SAMME.R timings")
    if probability_values is None or prediction_values is None:
        raise RuntimeError("release app omitted SAMME.R outputs")
    if stage_weight is None or stage_count is None or cuda_code is None:
        raise RuntimeError("release app omitted SAMME.R metadata")
    if probability_values.size != N_QUERY * N_CLASSES:
        raise RuntimeError(f"unexpected SAMME.R probability size: {probability_values.size}")
    probabilities = probability_values.reshape((N_CLASSES, N_QUERY), order="C").T
    return timing, probabilities, prediction_values, stage_weight, stage_count, cuda_code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/adaboost_samme_r.csv"))
    parser.add_argument("--target", default="fortml_bench_adaboost_samme_r")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (args.output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    completed = subprocess.run(["fo", "exec", "--no-build", args.target], cwd=fortml,
                               env=environment, check=True, capture_output=True, text=True)
    timing, observed_probability, observed_prediction, observed_weight, stage_count, cuda_code = parse(completed.stdout)
    expected_probability, expected_prediction = oracle()
    probability_error = float(np.max(np.abs(observed_probability - expected_probability)))
    prediction_error = float(np.max(np.abs(observed_prediction - expected_prediction)))
    weight_error = abs(observed_weight - 1.0)
    if (probability_error > 5.0e-14 or prediction_error != 0.0 or
            weight_error > 5.0e-14 or stage_count != 1 or cuda_code == 0):
        raise RuntimeError(
            f"SAMME.R oracle mismatch: probability={probability_error:.3e}, "
            f"prediction={prediction_error:.3e}, weight={weight_error:.3e}, "
            f"stages={stage_count}, cuda_code={cuda_code}"
        )
    rows = [
        row(details, workload="adaboost_samme_r", phase="fit", backend="fortml",
            device="cpu", status="pass", n_samples=N_SAMPLES, n_features=N_FEATURES,
            n_classes=N_CLASSES, n_query=N_QUERY, n_estimators=1,
            seconds_per_operation=timing["sammer_fit"], metric="stage_weight_max_abs_error",
            value=weight_error, max_abs_error=weight_error,
            oracle="independent NumPy SAMME.R stump replay",
            notes="sorted labels [-7,4,99], clipped log-probability update"),
        row(details, workload="adaboost_samme_r", phase="predict", backend="fortml",
            device="cpu", status="pass", n_samples=N_SAMPLES, n_features=N_FEATURES,
            n_classes=N_CLASSES, n_query=N_QUERY, n_estimators=1,
            seconds_per_operation=timing["sammer_predict"],
            metric="probability_max_abs_error", value=probability_error,
            max_abs_error=probability_error,
            oracle="independent NumPy clipped geometric probability ensemble",
            notes="five query probabilities and labels match"),
        row(details, workload="adaboost_samme_r", phase="device_contract",
            backend="fortml", device="cuda", status="unavailable", n_samples=N_SAMPLES,
            n_features=N_FEATURES, n_classes=N_CLASSES, n_query=N_QUERY, n_estimators=1,
            metric="typed_refusal_status", value=cuda_code, max_abs_error=0.0,
            oracle="device capability contract",
            notes="no resident SAMME.R AdaBoost tree ensemble kernel; typed refusal"),
    ]
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
