#!/usr/bin/env python3
"""Correctness-gated benchmark for weighted linear SVM classification.

The NumPy/SciPy primal squared-hinge solve is an independent behavioral
oracle.  A FortML row is retained only after all labels, classes, predictions,
and signed margins agree.  CUDA is represented as an explicit unavailable
capability row until a resident affine/SVM kernel is linked.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import scipy
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover - release environment normally includes it
    scipy = None
    minimize = None


N_SAMPLES, N_FEATURES = 192, 4
L2 = 5.0e-2
PREDICTION_REPETITIONS = 128
CLASSES = np.array([-12, 37], dtype=np.int64)
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "seconds_per_operation", "accuracy", "max_abs_error",
    "oracle", "python_version", "numpy_version", "scipy_version",
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
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        name = line[3:].split(" -> ")[-1].strip()
        if name not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = np.arange(1, N_SAMPLES + 1, dtype=np.float64)
    columns = np.arange(1, N_FEATURES + 1, dtype=np.float64)[None, :]
    x = np.sin(0.017 * phase[:, None] + 0.071 * columns)
    x += 0.12 * np.cos(0.013 * phase[:, None] * columns)
    truth = (0.8 * x[:, 0] - 0.45 * x[:, 1] + 0.25 * x[:, 2] -
             0.15 * x[:, 3] + 0.10 * np.sin(0.11 * phase))
    labels = np.where(truth >= 0.0, CLASSES[1], CLASSES[0]).astype(np.int64)
    weights = 0.75 + 0.5 * (np.mod(np.arange(1, N_SAMPLES + 1), 9) / 8.0)
    return x, labels, weights


def objective_and_gradient(theta: np.ndarray, x: np.ndarray, labels: np.ndarray,
                           weights: np.ndarray) -> tuple[float, np.ndarray]:
    encoded = np.where(labels == CLASSES[1], 1.0, -1.0)
    beta = theta[:N_FEATURES]
    intercept = theta[N_FEATURES]
    margin = encoded * (x @ beta + intercept)
    violation = np.maximum(0.0, 1.0 - margin)
    mass = float(weights.sum())
    value = float(np.dot(weights, violation * violation) / mass +
                  0.5 * L2 * np.dot(beta, beta))
    score_gradient = -2.0 * encoded * violation
    gradient = np.empty_like(theta)
    gradient[:N_FEATURES] = x.T @ (weights * score_gradient) / mass + L2 * beta
    gradient[N_FEATURES] = np.dot(weights, score_gradient) / mass
    return value, gradient


def oracle(x: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if minimize is None:
        raise RuntimeError("SciPy is required for the independent linear-SVM oracle")
    result = minimize(
        lambda theta: objective_and_gradient(theta, x, labels, weights),
        np.zeros(N_FEATURES + 1, dtype=np.float64), method="L-BFGS-B", jac=True,
        options={"maxiter": 1500, "ftol": 1.0e-15, "gtol": 1.0e-9,
                 "maxls": 50},
    )
    if not result.success:
        raise RuntimeError(f"independent SVM oracle failed: {result.message}")
    theta = result.x
    scores = x @ theta[:N_FEATURES] + theta[N_FEATURES]
    predicted = np.where(scores >= 0.0, CLASSES[1], CLASSES[0]).astype(np.int64)
    return theta, scores, predicted


def parse_fortran(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.full(N_SAMPLES, -999, dtype=np.int64)
    scores = np.full(N_SAMPLES, np.nan, dtype=np.float64)
    predicted = np.full(N_SAMPLES, -999, dtype=np.int64)
    classes = np.full(2, -999, dtype=np.int64)
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            quantity = record["quantity"]
            row = int(record["row"]) - 1
            value = float(record["value"])
            if quantity == "label":
                labels[row] = int(value)
            elif quantity == "score":
                scores[row] = value
            elif quantity == "prediction":
                predicted[row] = int(value)
            elif quantity == "class":
                classes[int(record["row"]) - 1] = int(value)
            else:
                raise RuntimeError(f"unknown FortML quantity {quantity!r}")
    if (np.any(labels == -999) or np.any(predicted == -999) or
            np.isnan(scores).any() or np.any(classes == -999)):
        raise RuntimeError("FortML omitted a linear-SVM output")
    return labels, scores, predicted, classes


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "scipy_version": "unavailable" if scipy is None else scipy.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }


def make_row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "workload": "linear_svm_classifier", "phase": "", "backend": "",
        "device": "cpu", "status": "", "n_samples": N_SAMPLES,
        "n_features": N_FEATURES, "seconds_per_operation": "", "accuracy": "",
        "max_abs_error": "", "oracle": "", "notes": "",
    }
    row.update(details)
    row.update(values)
    return row


def parse_timing(stdout: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in stdout.splitlines():
        fields = line.strip().split(",")
        if len(fields) == 4 and fields[0] in {"linear_svm_fit", "linear_svm_predict"}:
            values[fields[0]] = float(fields[-1])
    return values


def run_numpy(details: dict[str, str]) -> list[dict[str, Any]]:
    x, labels, weights = fixture()
    theta, scores, predicted = oracle(x, labels, weights)
    started = time.perf_counter()
    for _ in range(PREDICTION_REPETITIONS):
        x @ theta[:N_FEATURES] + theta[N_FEATURES]
    seconds = (time.perf_counter() - started) / PREDICTION_REPETITIONS
    return [make_row(
        details, phase="fit_predict", backend="numpy_oracle", status="pass",
        seconds_per_operation=seconds, accuracy=float(np.mean(predicted == labels)),
        max_abs_error=0.0,
        oracle="independent SciPy L-BFGS-B weighted squared-hinge solve",
        notes="NumPy affine prediction checksum; ordinary hinge boundary is separate",
    )]


def run_fortml(root: Path, fortml: Path, details: dict[str, str]) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                   env=environment, check=True)
    x, labels, weights = fixture()
    theta, expected_scores, expected_predicted = oracle(x, labels, weights)
    with tempfile.TemporaryDirectory(dir="/mnt/storage/code/lazy-fortran/fortml/build") as directory:
        output = Path(directory) / "linear_svm_oracle.csv"
        environment["FORTML_BENCH_LINEAR_SVM_ORACLE"] = str(output)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", "fortml_bench_linear_svm"],
            cwd=fortml, env=environment, capture_output=True, text=True, check=True,
        )
        actual_labels, actual_scores, actual_predicted, actual_classes = parse_fortran(output)
    if not np.array_equal(actual_labels, labels):
        raise RuntimeError("FortML linear-SVM fixture labels differ from NumPy")
    if not np.array_equal(actual_classes, CLASSES):
        raise RuntimeError(f"FortML class order differs: {actual_classes} != {CLASSES}")
    error = max(float(np.max(np.abs(actual_scores - expected_scores))),
                float(np.max(actual_predicted != expected_predicted)))
    if error > 2.0e-4:
        raise RuntimeError(f"FortML linear-SVM oracle mismatch {error:.3e}")
    timing = parse_timing(completed.stdout)
    accuracy = float(np.mean(actual_predicted == labels))
    return [
        make_row(details, phase="fit", backend="fortml_cpu", status="pass",
                 seconds_per_operation=timing.get("linear_svm_fit", ""),
                 accuracy=accuracy, max_abs_error=error,
                 oracle="NumPy/SciPy complete labels/classes/signed-margin oracle",
                 notes="FortOpt L-BFGS-B weighted squared hinge"),
        make_row(details, phase="predict", backend="fortml_cpu", status="pass",
                 seconds_per_operation=timing.get("linear_svm_predict", ""),
                 accuracy=accuracy, max_abs_error=error,
                 oracle="NumPy/SciPy complete labels/classes/signed-margin oracle",
                 notes="fixed fitted affine margin prediction"),
        make_row(details, phase="predict", backend="fortml_cuda", device="cuda",
                 status="unavailable", oracle="typed_device_contract",
                 notes="no resident linear-SVM CUDA kernel; FORTNUM_NOT_IMPLEMENTED"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/linear_svm.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    details = metadata(root, args.fortml.resolve(), args.output.resolve())
    rows = run_numpy(details) + run_fortml(root, args.fortml.resolve(), details)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
