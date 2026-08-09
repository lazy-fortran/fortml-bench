#!/usr/bin/env python3
"""Independent correctness-gated polynomial-kernel SVM benchmark."""

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
except ImportError:  # pragma: no cover
    scipy = None
    minimize = None

N_SAMPLES, N_FEATURES = 8, 2
C, GAMMA, DEGREE, COEF0 = 3.0, 0.4, 2, 1.0
CLASSES = np.array([-12, 37], dtype=np.int64)
PREDICTION_REPETITIONS = 64
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "seconds_per_operation", "accuracy", "max_abs_error",
    "metric", "value", "oracle", "python_version", "numpy_version", "scipy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    ignored_names: set[str] = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            pass
    dirty = []
    for line in subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).splitlines():
        name = line[3:].split(" -> ")[-1].strip()
        if name not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.array([
        [-1.0, -1.0], [-0.8, -1.1], [1.0, 1.0], [0.8, 1.1],
        [-1.0, 1.0], [-0.8, 1.1], [1.0, -1.0], [0.8, -1.1],
    ], dtype=np.float64)
    labels = np.array([-12, -12, -12, -12, 37, 37, 37, 37], dtype=np.int64)
    return x, labels, np.ones(N_SAMPLES, dtype=np.float64)


def kernel_matrix(x: np.ndarray) -> np.ndarray:
    return (GAMMA * (x @ x.T) + COEF0) ** DEGREE


def objective_and_gradient(theta: np.ndarray, kernel: np.ndarray,
                           encoded: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    n = len(encoded)
    coefficient, intercept = theta[:n], theta[n]
    score = kernel @ coefficient + intercept
    residual = np.maximum(0.0, 1.0 - encoded * score)
    mass = float(weights.sum())
    score_gradient = -2.0 * C / mass * weights * encoded * residual
    value = 0.5 * float(coefficient @ (kernel @ coefficient)) + C / mass * float(np.dot(weights, residual * residual))
    gradient = np.zeros(n + 3, dtype=np.float64)
    gradient[:n] = kernel @ coefficient + kernel @ score_gradient
    gradient[n] = float(score_gradient.sum())
    return value, gradient


def oracle(x: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    if minimize is None:
        raise RuntimeError("SciPy is required for the independent polynomial-SVM oracle")
    kernel = kernel_matrix(x)
    encoded = np.where(labels == CLASSES[1], 1.0, -1.0)
    initial = np.zeros(len(x) + 3, dtype=np.float64)
    initial[len(x) + 1] = np.log(GAMMA)
    initial[len(x) + 2] = COEF0
    result = minimize(lambda theta: objective_and_gradient(theta, kernel, encoded, weights), initial,
                      method="L-BFGS-B", jac=True,
                      bounds=[(None, None)] * (len(x) + 2) + [(0.0, None)],
                      options={"maxiter": 50000, "ftol": 1.0e-13, "gtol": 1.0e-7, "maxls": 100})
    if not result.success:
        raise RuntimeError(f"independent polynomial-SVM oracle failed: {result.message}")
    coefficient, intercept = result.x[:len(x)], float(result.x[len(x)])
    scores = kernel @ coefficient + intercept
    predictions = np.where(scores >= 0.0, CLASSES[1], CLASSES[0]).astype(np.int64)
    return coefficient, intercept, scores, predictions


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "scipy_version": "unavailable" if scipy is None else scipy.__version__,
        "fortml_revision": revision(fortml), "benchmark_revision": revision(root, (output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }


def make_row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "workload": "polynomial_svm", "phase": "", "backend": "", "device": "cpu",
        "status": "", "n_samples": N_SAMPLES, "n_features": N_FEATURES,
        "seconds_per_operation": "", "accuracy": "", "max_abs_error": "",
        "metric": "", "value": "",
        "oracle": "", "notes": "",
    }
    row.update(details); row.update(values)
    return row


def parse_fortran(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    labels = np.full(N_SAMPLES, -999, dtype=np.int64)
    scores = np.full(N_SAMPLES, np.nan); predictions = np.full(N_SAMPLES, -999, dtype=np.int64)
    coefficients = np.full(N_SAMPLES, np.nan); classes = np.full(2, -999, dtype=np.int64)
    scalars: dict[str, float] = {}
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            quantity, row = record["quantity"], int(record["row"]) - 1
            value = float(record["value"])
            if quantity == "label": labels[row] = int(value)
            elif quantity == "score": scores[row] = value
            elif quantity == "prediction": predictions[row] = int(value)
            elif quantity == "coefficient": coefficients[row] = value
            elif quantity == "class": classes[row] = int(value)
            else: scalars[quantity] = value
    if np.any(labels == -999) or np.isnan(scores).any() or np.any(predictions == -999) or np.isnan(coefficients).any() or np.any(classes == -999):
        raise RuntimeError("FortML omitted a polynomial-SVM output")
    return labels, scores, predictions, coefficients, classes, scalars


def parse_timing(stdout: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in stdout.splitlines():
        fields = line.strip().split(",")
        if len(fields) == 4 and fields[0] in {"polynomial_svm_fit", "polynomial_svm_predict"}:
            values[fields[0]] = float(fields[-1])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/polynomial_svm.csv"))
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    root, fortml = Path(__file__).resolve().parents[1], args.fortml.resolve()
    details = metadata(root, fortml, args.output.resolve())
    x, labels, weights = fixture()
    coefficient, intercept, expected_scores, expected_predictions = oracle(x, labels, weights)
    started = time.perf_counter()
    for _ in range(PREDICTION_REPETITIONS): kernel_matrix(x) @ coefficient + intercept
    numpy_seconds = (time.perf_counter() - started) / PREDICTION_REPETITIONS
    rows = [make_row(details, phase="fit_predict", backend="numpy_oracle", status="pass",
                     seconds_per_operation=numpy_seconds, accuracy=float(np.mean(expected_predictions == labels)),
                     metric="accuracy", value=float(np.mean(expected_predictions == labels)),
                     max_abs_error=0.0, oracle="independent SciPy L-BFGS-B weighted squared-hinge polynomial RKHS solve",
                     notes="degree=2; gamma=0.4; coef0=1.0; dense finite training basis")]
    environment = os.environ.copy(); environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    if not args.no_build:
        subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment, check=True)
    subprocess.run(["fo", "test", "test_polynomial_svm_classifier"], cwd=fortml, env=environment, check=True)
    with tempfile.TemporaryDirectory(dir="/mnt/storage/code/lazy-fortran/fortml/build") as directory:
        output = Path(directory) / "polynomial_svm_oracle.csv"
        environment["FORTML_BENCH_POLYNOMIAL_SVM_ORACLE"] = str(output)
        completed = subprocess.run(["fo", "exec", "--no-build", "fortml_bench_polynomial_svm"], cwd=fortml,
                                   env=environment, capture_output=True, text=True, check=True)
        actual_labels, actual_scores, actual_predictions, actual_coefficient, actual_classes, scalars = parse_fortran(output)
    error = max(float(np.max(np.abs(actual_scores - expected_scores))),
                float(np.max(actual_predictions != expected_predictions)),
                float(np.max(actual_labels != labels)), abs(scalars["gamma"] - GAMMA),
                abs(scalars["coef0"] - COEF0), abs(scalars["degree"] - DEGREE),
                abs(scalars["intercept"] - intercept))
    if not np.array_equal(actual_classes, CLASSES): raise RuntimeError(f"FortML classes {actual_classes} != {CLASSES}")
    if error > 3.0e-4: raise RuntimeError(f"FortML polynomial-SVM oracle mismatch {error:.3e}")
    timing = parse_timing(completed.stdout); accuracy = float(np.mean(actual_predictions == labels))
    rows.extend([
        make_row(details, phase="fit", backend="fortml_cpu", status="pass", seconds_per_operation=timing.get("polynomial_svm_fit", ""),
                 accuracy=accuracy, metric="accuracy", value=accuracy, max_abs_error=error, oracle="SciPy polynomial score/label oracle",
                 notes="FortOpt weighted squared-hinge fit; fixed-state JVP/VJP independently tested"),
        make_row(details, phase="predict", backend="fortml_cpu", status="pass", seconds_per_operation=timing.get("polynomial_svm_predict", ""),
                 accuracy=accuracy, metric="accuracy", value=accuracy, max_abs_error=error, oracle="SciPy polynomial score/label oracle", notes="dense finite-basis prediction"),
        make_row(details, phase="predict", backend="fortml_cuda", device="cuda", status="unavailable",
                 metric="resident_polynomial_svm", value="nan", max_abs_error=0.0,
                 oracle="typed_device_contract", notes="resident CUDA polynomial-SVM value/derivative kernels absent; FORTNUM_NOT_IMPLEMENTED"),
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__": main()
