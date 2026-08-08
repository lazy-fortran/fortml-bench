#!/usr/bin/env python3
"""Correctness-gated leakage-safe logistic calibration benchmark.

The release application fits each stratified fold independently, calibrates
held-out margins, and refits the deployment model. The NumPy oracle replays
the packed deployment parameters and checks labels, probabilities, and fold
diagnostics. CUDA is recorded as a typed refusal until the complete resident
graph exists.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


N_SAMPLES = 96
N_FEATURES = 2
CLASSES = np.array([-3, 42], dtype=np.int64)
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "cv_folds", "method", "metric", "value", "max_abs_error",
    "oracle", "seconds_per_operation", "python_version", "numpy_version",
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
            pass
    dirty = any(
        line[3:].split(" -> ")[-1].strip() not in ignored_names
        for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True,
        ).splitlines()
    )
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray]:
    first = np.array(
        [-2.0, -1.7, -1.4, -1.1, -0.7, -0.3, 0.2, 0.5, 0.8, 1.1, 1.5, 1.9],
        dtype=np.float64,
    )
    second = np.array(
        [-1.4, -0.8, -1.1, -0.5, -0.2, 0.1, -0.1, 0.3, 0.7, 0.9, 1.2, 1.6],
        dtype=np.float64,
    )
    x = np.empty((N_SAMPLES, N_FEATURES), dtype=np.float64)
    labels = np.empty(N_SAMPLES, dtype=np.int64)
    for i in range(N_SAMPLES):
        position = i % first.size
        x[i, 0] = first[position]
        x[i, 1] = second[position]
        labels[i] = 42 if position >= 6 else -3
    return x, labels


def sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def parse_oracle(path: Path) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, float]
]:
    parameters: list[float] = []
    labels = np.zeros(N_SAMPLES, dtype=np.int64)
    predictions = np.zeros(N_SAMPLES, dtype=np.int64)
    probabilities = np.zeros((N_SAMPLES, 2), dtype=np.float64)
    diagnostics: dict[int, float] = {}
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            quantity = record["quantity"]
            row = int(record["row"])
            column = int(record["column"])
            value = float(record["value"])
            if quantity == "parameter":
                parameters.append(value)
            elif quantity == "label":
                labels[row - 1] = round(value)
            elif quantity == "prediction":
                predictions[row - 1] = round(value)
            elif quantity == "probability":
                probabilities[row - 1, column - 1] = value
            elif quantity == "diagnostic":
                diagnostics[row] = value
    return np.asarray(parameters), labels, predictions, probabilities, diagnostics


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/calibrated_logistic_cv.csv"))
    parser.add_argument("--target", default="fortml_bench_calibrated_logistic_cv")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
        "oracle": "independent NumPy packed-logistic/temperature replay",
    }
    x, expected_labels = fixture()
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"),
                        "OMP_NUM_THREADS": "1"})
    with tempfile.TemporaryDirectory(dir="/mnt/storage", prefix="fortml-calibrated-logistic-") as temporary:
        oracle_path = Path(temporary) / "oracle.csv"
        environment["FORTML_BENCH_CALIBRATED_LOGISTIC_ORACLE"] = str(oracle_path)
        build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                               env=environment, capture_output=True, text=True)
        if build.returncode:
            raise RuntimeError(build.stderr.strip() or build.stdout.strip())
        run = subprocess.run(["fo", "exec", "--no-build", args.target], cwd=fortml,
                             env=environment, capture_output=True, text=True)
        if run.returncode or not oracle_path.is_file():
            raise RuntimeError(run.stderr.strip() or run.stdout.strip() or
                               "calibrated logistic app emitted no oracle")
        parameters, labels, predictions, observed, diagnostics = parse_oracle(oracle_path)
        if parameters.size != 4:
            raise RuntimeError(f"unexpected packed parameter count {parameters.size}")
        margin = x @ parameters[:N_FEATURES] + parameters[N_FEATURES]
        positive = sigmoid(margin / parameters[N_FEATURES + 1])
        expected_probabilities = np.column_stack((1.0 - positive, positive))
        probability_error = float(np.max(np.abs(observed - expected_probabilities)))
        label_error = float(np.max(np.abs(labels - expected_labels)))
        prediction_error = float(np.max(np.abs(predictions -
                                               np.where(positive > 0.5, 42, -3))))
        simplex_error = float(np.max(np.abs(observed.sum(axis=1) - 1.0)))
        if probability_error > 5.0e-13 or label_error != 0.0 or prediction_error != 0.0:
            raise RuntimeError(
                f"calibrated logistic oracle mismatch: probability={probability_error:.3e}, "
                f"labels={label_error:.3e}, predictions={prediction_error:.3e}"
            )
        if diagnostics.get(1) != 3.0 or not all(np.isfinite(diagnostics.get(i, np.nan)) for i in (2, 3)):
            raise RuntimeError("OOF diagnostics are missing or nonfinite")
        fit_seconds = next(float(line.rsplit(",", 1)[1])
                           for line in run.stdout.splitlines()
                           if line.startswith("calibrated_logistic_cv_fit,"))
        predict_seconds = next(float(line.rsplit(",", 1)[1])
                               for line in run.stdout.splitlines()
                               if line.startswith("calibrated_logistic_cv_predict,"))
    records = [
        row(details, workload="calibrated_logistic_cv", phase="fit", backend="fortml",
            device="cpu", status="pass", n_samples=N_SAMPLES, n_features=N_FEATURES,
            cv_folds=3, method="temperature", metric="packed_replay_max_abs_error",
            value=probability_error, max_abs_error=probability_error,
            seconds_per_operation=fit_seconds,
            notes="stratified held-out margins and deployment refit"),
        row(details, workload="calibrated_logistic_cv", phase="predict", backend="fortml",
            device="cpu", status="pass", n_samples=N_SAMPLES, n_features=N_FEATURES,
            cv_folds=3, method="temperature", metric="probability_simplex_error",
            value=simplex_error, max_abs_error=max(probability_error, simplex_error),
            seconds_per_operation=predict_seconds,
            notes="sorted labels and final calibrated temperature replay"),
        row(details, workload="calibrated_logistic_cv", phase="diagnostics", backend="fortml",
            device="cpu", status="pass", n_samples=N_SAMPLES, n_features=N_FEATURES,
            cv_folds=3, method="temperature", metric="oof_log_loss",
            value=diagnostics[2], max_abs_error=0.0,
            seconds_per_operation="", notes="uncalibrated out-of-fold log loss"),
        row(details, workload="calibrated_logistic_cv", phase="diagnostics", backend="fortml",
            device="cpu", status="pass", n_samples=N_SAMPLES, n_features=N_FEATURES,
            cv_folds=3, method="temperature", metric="calibrated_oof_log_loss",
            value=diagnostics[3], max_abs_error=0.0,
            seconds_per_operation="", notes="calibrated out-of-fold log loss"),
        row(details, workload="calibrated_logistic_cv", phase="device_contract",
            backend="fortml", device="cuda", status="unavailable", n_samples=N_SAMPLES,
            n_features=N_FEATURES, cv_folds=3, method="temperature",
            metric="predict_proba", value="unavailable", max_abs_error=0.0,
            oracle="typed device contract",
            notes="FORTNUM_NOT_IMPLEMENTED; resident logistic/calibration graph is absent"),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} rows to {output}")


if __name__ == "__main__":
    main()
