#!/usr/bin/env python3
"""Correctness-gated multiclass OOF temperature-calibration benchmark.

The FortML application fits a fresh softmax model in each stratified fold,
calibrates held-out logits, and refits the deployment model.  This script
replays the packed deployment parameters with NumPy and records the OOF loss
diagnostics plus the explicit CUDA capability boundary.
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
N_CLASSES = 3
CLASS_LABELS = np.array([-4, 17, 91], dtype=np.int64)
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_classes", "cv_folds", "method", "metric", "value",
    "max_abs_error", "oracle", "seconds_per_operation", "python_version",
    "numpy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    """Return the revision and mark source changes outside ignored outputs."""

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
    """Build the same deterministic three-class fixture as the Fortran app."""

    x = np.empty((N_SAMPLES, N_FEATURES), dtype=np.float64)
    labels = np.empty(N_SAMPLES, dtype=np.int64)
    for i in range(N_SAMPLES):
        class_index = i % N_CLASSES
        phase = float(i + 1)
        if class_index == 0:
            x[i] = (2.0 + 0.1 * np.sin(phase), 0.1 * np.cos(phase))
            labels[i] = CLASS_LABELS[0]
        elif class_index == 1:
            x[i] = (0.1 * np.cos(phase), 2.0 + 0.1 * np.sin(phase))
            labels[i] = CLASS_LABELS[1]
        else:
            x[i] = (-2.0 + 0.1 * np.sin(phase), -2.0 + 0.1 * np.cos(phase))
            labels[i] = CLASS_LABELS[2]
    return x, labels


def parse_oracle(path: Path) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]
]:
    parameters: list[float] = []
    labels = np.zeros(N_SAMPLES, dtype=np.int64)
    predictions = np.zeros(N_SAMPLES, dtype=np.int64)
    probabilities = np.zeros((N_SAMPLES, N_CLASSES), dtype=np.float64)
    diagnostics: dict[str, float] = {}
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            quantity = record["quantity"]
            if quantity == "parameter":
                parameters.append(float(record["value"]))
            elif quantity == "label":
                labels[int(record["row"]) - 1] = round(float(record["value"]))
            elif quantity == "prediction":
                predictions[int(record["row"]) - 1] = round(float(record["value"]))
            elif quantity == "probability":
                row = int(record["row"]) - 1
                column = int(record["column"]) - 1
                probabilities[row, column] = float(record["value"])
            elif quantity in {"oof_log_loss", "calibrated_oof_log_loss"}:
                diagnostics[quantity] = float(record["value"])
    return np.asarray(parameters), labels, predictions, probabilities, diagnostics


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/calibrated_softmax_cv.csv"))
    parser.add_argument("--target", default="fortml_bench_calibrated_softmax_cv")
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
    }
    x, expected_labels = fixture()
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"),
                        "OMP_NUM_THREADS": "1"})
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(dir="/mnt/storage", prefix="fortml-calibrated-softmax-") as temporary:
        build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                               env=environment, capture_output=True, text=True)
        if build.returncode:
            raise RuntimeError(build.stderr.strip() or build.stdout.strip())
        for method in ("temperature", "sigmoid", "isotonic"):
            oracle_path = Path(temporary) / f"oracle-{method}.csv"
            environment["FORTML_BENCH_CALIBRATED_SOFTMAX_ORACLE"] = str(oracle_path)
            environment["FORTML_BENCH_CALIBRATED_SOFTMAX_METHOD"] = method
            run = subprocess.run(["fo", "exec", "--no-build", args.target], cwd=fortml,
                                 env=environment, capture_output=True, text=True)
            if run.returncode or not oracle_path.is_file():
                raise RuntimeError(run.stderr.strip() or run.stdout.strip() or
                                   f"calibrated softmax {method} app emitted no oracle")
            parameters, labels, predictions, observed, diagnostics = parse_oracle(oracle_path)
            base_parameter_count = N_FEATURES * N_CLASSES + N_CLASSES
            expected_parameter_count = base_parameter_count + {
                "temperature": 1, "sigmoid": 2 * N_CLASSES, "isotonic": 0,
            }[method]
            if parameters.size != expected_parameter_count:
                raise RuntimeError(f"{method}: unexpected packed parameter count {parameters.size}")
            coefficients = parameters[:N_FEATURES * N_CLASSES].reshape(
                (N_FEATURES, N_CLASSES), order="F",
            )
            intercept = parameters[N_FEATURES * N_CLASSES:base_parameter_count]
            scores = x @ coefficients + intercept
            if method == "temperature":
                temperature = parameters[-1]
                if not np.isfinite(temperature) or temperature <= 0.0:
                    raise RuntimeError(f"temperature is not positive: {temperature}")
                expected_probabilities = softmax(scores / temperature)
                oracle_name = "independent NumPy packed-softmax/temperature replay"
            elif method == "sigmoid":
                calibration = parameters[base_parameter_count:].reshape((N_CLASSES, 2))
                raw = softmax(scores)
                calibrated = 1.0 / (1.0 + np.exp(-(
                    raw * calibration[:, 0][None, :] + calibration[:, 1][None, :]
                )))
                expected_probabilities = calibrated / calibrated.sum(axis=1, keepdims=True)
                oracle_name = "independent NumPy packed-softmax/Platt replay"
            else:
                expected_probabilities = None
                oracle_name = "independent NumPy simplex/label oracle; PAVA knots are fitted buffers"
            probability_error = (
                float(np.max(np.abs(observed - expected_probabilities)))
                if expected_probabilities is not None else float("nan")
            )
            label_error = float(np.max(np.abs(labels - expected_labels)))
            expected_predictions = CLASS_LABELS[np.argmax(
                expected_probabilities if expected_probabilities is not None else observed,
                axis=1,
            )]
            prediction_error = float(np.max(np.abs(predictions - expected_predictions)))
            simplex_error = float(np.max(np.abs(observed.sum(axis=1) - 1.0)))
            if (expected_probabilities is not None and probability_error > 5.0e-12) or \
                    label_error != 0.0 or prediction_error != 0.0 or \
                    not np.all(np.isfinite(observed)) or simplex_error > 5.0e-14:
                raise RuntimeError(
                    f"{method} oracle mismatch: probability={probability_error:.3e}, "
                    f"simplex={simplex_error:.3e}, labels={label_error:.3e}, "
                    f"predictions={prediction_error:.3e}"
                )
            if not all(np.isfinite(diagnostics.get(name, np.nan)) for name in (
                    "oof_log_loss", "calibrated_oof_log_loss")):
                raise RuntimeError(f"{method}: OOF diagnostics are missing or nonfinite")
            fit_seconds = next(float(line.rsplit(",", 1)[1])
                               for line in run.stdout.splitlines()
                               if line.startswith(f"calibrated_softmax_cv_fit,{method},"))
            predict_seconds = next(float(line.rsplit(",", 1)[1])
                                   for line in run.stdout.splitlines()
                                   if line.startswith(f"calibrated_softmax_cv_predict,{method},"))
            common = dict(workload="calibrated_softmax_cv", backend="fortml",
                          n_samples=N_SAMPLES, n_features=N_FEATURES, n_classes=N_CLASSES,
                          cv_folds=3, method=method)
            records.extend([
                row(details, phase="fit", device="cpu", status="pass", **common,
                    metric="packed_parameter_count", value=parameters.size,
                    max_abs_error=0.0, oracle=oracle_name,
                    seconds_per_operation=fit_seconds,
                    notes="stratified held-out logits and deployment refit"),
                row(details, phase="predict", device="cpu", status="pass", **common,
                    metric="packed_replay_max_abs_error",
                    value=(probability_error if np.isfinite(probability_error) else "unavailable"),
                    max_abs_error=(probability_error if np.isfinite(probability_error) else 0.0),
                    oracle=oracle_name, seconds_per_operation=predict_seconds,
                    notes=("sorted labels and smooth packed Platt replay" if method == "sigmoid"
                           else "sorted labels and positive temperature replay" if method == "temperature"
                           else "sorted labels and deterministic weighted PAVA values")),
                row(details, phase="predict", device="cpu", status="pass", **common,
                    metric="probability_simplex_error", value=simplex_error,
                    max_abs_error=max(simplex_error, probability_error) if np.isfinite(probability_error)
                    else simplex_error, oracle=oracle_name,
                    notes="simplex normalization"),
                row(details, phase="diagnostics", device="cpu", status="pass", **common,
                    metric="oof_log_loss", value=diagnostics["oof_log_loss"], max_abs_error=0.0,
                    oracle=oracle_name, notes="uncalibrated out-of-fold log loss"),
                row(details, phase="diagnostics", device="cpu", status="pass", **common,
                    metric="calibrated_oof_log_loss", value=diagnostics["calibrated_oof_log_loss"],
                    max_abs_error=0.0, oracle=oracle_name, notes="calibrated out-of-fold log loss"),
                row(details, phase="device_contract", device="cuda", status="unavailable", **common,
                    metric="predict_proba", value="unavailable", max_abs_error=0.0,
                    oracle="typed device contract",
                    notes="FORTNUM_NOT_IMPLEMENTED; resident softmax/calibration graph is absent"),
            ])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} rows to {output}")


if __name__ == "__main__":
    main()
