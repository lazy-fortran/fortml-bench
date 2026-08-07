#!/usr/bin/env python3
"""Correctness-gated dense RBF binary SVM benchmark.

The NumPy/SciPy solve is an independent weighted squared-hinge RKHS oracle.
FortML rows are retained only after coefficients, scores, labels, and class
ordering agree.  CUDA is reported as a typed unavailable capability until a
resident RBF-SVM kernel is linked.
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


N_SAMPLES, N_FEATURES = 36, 2
C, GAMMA = 2.0, 0.6
CLASSES = np.array([-12, 37], dtype=np.int64)
PREDICTION_REPETITIONS = 64
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
    ignored_names: set[str] = set()
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
    x = np.empty((N_SAMPLES, N_FEATURES), dtype=np.float64)
    left = np.arange(N_SAMPLES) < N_SAMPLES // 2
    x[:, 0] = np.where(left, -1.0, 1.0) + 0.05 * np.sin(0.17 * phase)
    x[:, 1] = np.where(left, 0.2, -0.2) * np.cos(0.13 * phase)
    labels = np.where(left, CLASSES[0], CLASSES[1]).astype(np.int64)
    weights = 0.75 + 0.5 * (np.mod(np.arange(1, N_SAMPLES + 1), 7) / 6.0)
    return x, labels, weights


def kernel_matrix(x: np.ndarray, gamma: float) -> np.ndarray:
    delta = x[:, None, :] - x[None, :, :]
    return np.exp(-gamma * np.sum(delta * delta, axis=2))


def objective_and_gradient(theta: np.ndarray, kernel: np.ndarray,
                           encoded: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    n = kernel.shape[0]
    coefficient = theta[:n]
    intercept = theta[n]
    score = kernel @ coefficient + intercept
    residual = np.maximum(0.0, 1.0 - encoded * score)
    mass = float(weights.sum())
    score_gradient = -2.0 * C / mass * weights * encoded * residual
    value = (0.5 * float(coefficient @ (kernel @ coefficient)) +
             C / mass * float(np.dot(weights, residual * residual)))
    gradient = np.empty(n + 1, dtype=np.float64)
    gradient[:n] = kernel @ coefficient + kernel @ score_gradient
    gradient[n] = float(score_gradient.sum())
    return value, gradient


def oracle(x: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    if minimize is None:
        raise RuntimeError("SciPy is required for the independent RBF-SVM oracle")
    kernel = kernel_matrix(x, GAMMA)
    encoded = np.where(labels == CLASSES[1], 1.0, -1.0)
    result = minimize(
        lambda theta: objective_and_gradient(theta, kernel, encoded, weights),
        np.zeros(len(x) + 1, dtype=np.float64), method="L-BFGS-B", jac=True,
        options={"maxiter": 10000, "ftol": 1.0e-13, "gtol": 1.0e-7,
                 "maxls": 100},
    )
    if not result.success:
        raise RuntimeError(f"independent RBF-SVM oracle failed: {result.message}")
    coefficient = result.x[:len(x)]
    intercept = float(result.x[len(x)])
    scores = kernel @ coefficient + intercept
    predictions = np.where(scores >= 0.0, CLASSES[1], CLASSES[0]).astype(np.int64)
    return coefficient, intercept, scores, predictions


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
        "workload": "rbf_svm_classifier", "phase": "", "backend": "",
        "device": "cpu", "status": "", "n_samples": N_SAMPLES,
        "n_features": N_FEATURES, "seconds_per_operation": "", "accuracy": "",
        "max_abs_error": "", "oracle": "", "notes": "",
    }
    row.update(details)
    row.update(values)
    return row


def numpy_rows(details: dict[str, str], x: np.ndarray, labels: np.ndarray,
               weights: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray, float,
                                               np.ndarray, np.ndarray]:
    coefficient, intercept, scores, predictions = oracle(x, labels, weights)
    kernel = kernel_matrix(x, GAMMA)
    started = time.perf_counter()
    for _ in range(PREDICTION_REPETITIONS):
        kernel @ coefficient + intercept
    seconds = (time.perf_counter() - started) / PREDICTION_REPETITIONS
    rows = [make_row(
        details, phase="fit_predict", backend="numpy_oracle", status="pass",
        seconds_per_operation=seconds, accuracy=float(np.mean(predictions == labels)),
        max_abs_error=0.0,
        oracle="independent SciPy L-BFGS-B weighted squared-hinge RKHS solve",
        notes="dense training-set RBF basis; coefficient/score/label checksum",
    )]
    return rows, coefficient, intercept, scores, predictions


def parse_fortran(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                        float, float, np.ndarray]:
    labels = np.full(N_SAMPLES, -999, dtype=np.int64)
    scores = np.full(N_SAMPLES, np.nan, dtype=np.float64)
    predictions = np.full(N_SAMPLES, -999, dtype=np.int64)
    coefficients = np.full(N_SAMPLES, np.nan, dtype=np.float64)
    classes = np.full(2, -999, dtype=np.int64)
    gamma = np.nan
    intercept = np.nan
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
                predictions[row] = int(value)
            elif quantity == "coefficient":
                coefficients[row] = value
            elif quantity == "class":
                classes[row] = int(value)
            elif quantity == "gamma":
                gamma = value
            elif quantity == "intercept":
                intercept = value
            else:
                raise RuntimeError(f"unknown FortML quantity {quantity!r}")
    if (np.any(labels == -999) or np.isnan(scores).any() or
            np.any(predictions == -999) or np.isnan(coefficients).any() or
            np.any(classes == -999) or not np.isfinite(gamma) or
            not np.isfinite(intercept)):
        raise RuntimeError("FortML omitted an RBF-SVM output")
    return labels, scores, predictions, coefficients, gamma, intercept, classes


def parse_timing(stdout: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in stdout.splitlines():
        fields = line.strip().split(",")
        if len(fields) == 4 and fields[0] in {"rbf_svm_fit", "rbf_svm_predict"}:
            values[fields[0]] = float(fields[-1])
    return values


def fortml_rows(root: Path, fortml: Path, details: dict[str, str],
                coefficient: np.ndarray, intercept: float, expected_scores: np.ndarray,
                expected_predictions: np.ndarray, no_build: bool = False) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    if not no_build:
        subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                       env=environment, check=True)
    with tempfile.TemporaryDirectory(dir="/mnt/storage/code/lazy-fortran/fortml/build") as directory:
        output = Path(directory) / "rbf_svm_oracle.csv"
        environment["FORTML_BENCH_RBF_SVM_ORACLE"] = str(output)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", "fortml_bench_rbf_svm"],
            cwd=fortml, env=environment, capture_output=True, text=True, check=True,
        )
        actual_labels, actual_scores, actual_predictions, actual_coefficient, gamma, actual_intercept, classes = parse_fortran(output)
    x, labels, _ = fixture()
    # The finite training-set RBF Gram matrix is intentionally dense and can
    # be mildly ill-conditioned; equivalent coefficient vectors may differ
    # while inducing the same score map.  The behavioral oracle therefore
    # gates the predictions and scores (plus the intercept/hyperparameter),
    # rather than requiring a coordinate-wise dual representation.
    error = max(abs(actual_intercept - intercept),
                float(np.max(np.abs(actual_scores - expected_scores))),
                float(np.max(actual_predictions != expected_predictions)),
                float(np.max(actual_labels != labels)), abs(gamma - GAMMA))
    if not np.array_equal(classes, CLASSES):
        raise RuntimeError(f"FortML class order differs: {classes} != {CLASSES}")
    if error > 3.0e-4:
        raise RuntimeError(f"FortML RBF-SVM oracle mismatch {error:.3e}")
    timing = parse_timing(completed.stdout)
    accuracy = float(np.mean(actual_predictions == labels))
    return [
        make_row(details, phase="fit", backend="fortml_cpu", status="pass",
                 seconds_per_operation=timing.get("rbf_svm_fit", ""), accuracy=accuracy,
                 max_abs_error=error,
                 oracle="SciPy complete coefficient/intercept/RBF score/label oracle",
                 notes="FortOpt L-BFGS-B weighted squared-hinge RKHS basis; score oracle (dual coordinates may be ill-conditioned)"),
        make_row(details, phase="predict", backend="fortml_cpu", status="pass",
                 seconds_per_operation=timing.get("rbf_svm_predict", ""), accuracy=accuracy,
                 max_abs_error=error, oracle="SciPy complete RBF score/label oracle",
                 notes="fixed-state dense RBF prediction"),
        make_row(details, phase="predict", backend="fortml_cuda", device="cuda",
                 status="unavailable", oracle="typed_device_contract",
                 notes="resident CUDA RBF-SVM kernel absent; FORTNUM_NOT_IMPLEMENTED"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/rbf_svm.csv"))
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    details = metadata(root, fortml, args.output.resolve())
    x, labels, weights = fixture()
    rows, coefficient, intercept, expected_scores, expected_predictions = numpy_rows(
        details, x, labels, weights)
    rows += fortml_rows(root, fortml, details, coefficient, intercept,
                        expected_scores, expected_predictions, args.no_build)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
