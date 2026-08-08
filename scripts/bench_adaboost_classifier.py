#!/usr/bin/env python3
"""Correctness-gated binary AdaBoost weighted-stump benchmark."""

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
N_FEATURES = 1
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


def fixture() -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(N_SAMPLES, dtype=np.float64) % 6.0).reshape(-1, 1)
    labels = np.where(np.isin(x[:, 0], [0.0, 1.0, 3.0]), -3, 8).astype(np.int64)
    return x, labels


def oracle(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    error = 1.0 / 6.0
    alpha = 0.5 * np.log((1.0 - error) / error)
    score = np.where(x[:, 0] < 1.5, -alpha, alpha)
    probabilities = 1.0 / (1.0 + np.exp(-2.0 * score))
    predicted = np.where(score >= 0.0, 8, -3).astype(np.int64)
    return probabilities, predicted


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def parse(stdout: str) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    timing: dict[str, float] = {}
    probability_values: np.ndarray | None = None
    prediction_values: np.ndarray | None = None
    for line in stdout.splitlines():
        if line.startswith("adaboost_fit,") or line.startswith("adaboost_predict,"):
            fields = line.split(",")
            timing[fields[0]] = float(fields[-1])
        elif line.startswith("adaboost_probability_values "):
            probability_values = np.fromstring(line.split(None, 1)[1], sep=" ")
        elif line.startswith("adaboost_prediction_values "):
            prediction_values = np.fromstring(line.split(None, 1)[1], sep=" ").astype(np.int64)
    if set(timing) != {"adaboost_fit", "adaboost_predict"}:
        raise RuntimeError("release app omitted AdaBoost timings")
    if probability_values is None or prediction_values is None:
        raise RuntimeError("release app omitted AdaBoost outputs")
    return timing, probability_values, prediction_values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/adaboost_classifier.csv"))
    parser.add_argument("--target", default="fortml_bench_adaboost_classifier")
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
    timing, observed_probability, observed_prediction = parse(completed.stdout)
    x, _ = fixture()
    expected_probability, expected_prediction = oracle(x)
    probability_error = float(np.max(np.abs(observed_probability - expected_probability)))
    prediction_error = float(np.max(np.abs(observed_prediction - expected_prediction)))
    if probability_error > 5.0e-14 or prediction_error != 0.0:
        raise RuntimeError(
            f"AdaBoost oracle mismatch: probability={probability_error:.3e}, "
            f"prediction={prediction_error:.3e}"
        )
    rows = [
        row(details, workload="adaboost_classifier", phase="fit", backend="fortml",
            device="cpu", status="pass", n_samples=N_SAMPLES, n_features=N_FEATURES,
            n_estimators=1, seconds_per_operation=timing["adaboost_fit"],
            metric="probability_max_abs_error", value=probability_error,
            max_abs_error=probability_error,
            oracle="independent NumPy weighted-stump AdaBoost replay",
            notes="binary sorted-label one-stump fixture"),
        row(details, workload="adaboost_classifier", phase="predict", backend="fortml",
            device="cpu", status="pass", n_samples=N_SAMPLES, n_features=N_FEATURES,
            n_estimators=1, seconds_per_operation=timing["adaboost_predict"],
            metric="hard_label_max_abs_error", value=prediction_error,
            max_abs_error=prediction_error,
            oracle="independent NumPy signed-margin threshold replay",
            notes="probabilities and labels match"),
        row(details, workload="adaboost_classifier", phase="device_contract",
            backend="fortml", device="cuda", status="unavailable", n_samples=N_SAMPLES,
            n_features=N_FEATURES, n_estimators=1, metric="api_surface",
            value="unavailable", max_abs_error=0.0, oracle="device capability contract",
            notes="no resident AdaBoost tree ensemble kernel is linked; typed refusal"),
    ]
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
