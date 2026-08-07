#!/usr/bin/env python3
"""Benchmark the weighted ordinal cumulative-logit classifier.

SciPy's independent L-BFGS-B implementation supplies the behavioral oracle;
the FortML app must emit the complete ordered probability matrix and labels
before a timing row is retained.  CUDA is recorded as an explicit capability
boundary and is never replaced by a host timing.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
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


N_SAMPLES = 192
N_FEATURES = 4
N_CLASSES = 4
L2 = 5.0e-2
PREDICT_REPETITIONS = 64
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_classes", "repetitions", "seconds_per_operation",
    "accuracy", "max_abs_error", "oracle", "python_version", "numpy_version",
    "scipy_version", "fortml_revision", "benchmark_revision", "compiler", "flags",
    "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    value = subprocess.check_output(
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
    return value + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = np.arange(1, N_SAMPLES + 1, dtype=np.float64)
    columns = np.arange(1, N_FEATURES + 1, dtype=np.float64)[None, :]
    x = np.sin(0.021 * phase[:, None] + 0.083 * columns)
    x += 0.15 * np.cos(0.011 * phase[:, None] * columns)
    score = 0.7 * x[:, 0] - 0.3 * x[:, 1] + 0.2 * np.sin(0.13 * phase)
    labels = np.select(
        [score < -0.25, score < 0.05, score < 0.30],
        [-9, 4, 13], default=42,
    ).astype(np.int64)
    weights = 0.75 + 0.5 * (np.mod(np.arange(1, N_SAMPLES + 1), 7) / 6.0)
    return x, labels, weights


def sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def raw_thresholds(raw: np.ndarray) -> np.ndarray:
    thresholds = np.empty(raw.size, dtype=np.float64)
    thresholds[0] = raw[0]
    thresholds[1:] = thresholds[0] + np.cumsum(np.exp(raw[1:]))
    return thresholds


def objective_and_gradient(theta: np.ndarray, x: np.ndarray, classes: np.ndarray,
                           encoded: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    n_features = x.shape[1]
    n_thresholds = classes.size - 1
    beta = theta[:n_features]
    intercept = theta[n_features]
    raw = theta[n_features + 1:]
    thresholds = raw_thresholds(raw)
    eta = x @ beta + intercept
    q = sigmoid(thresholds[None, :] - eta[:, None])
    qp = q * (1.0 - q)
    probability = np.empty(x.shape[0], dtype=np.float64)
    dp_eta = np.empty_like(probability)
    dp_threshold = np.zeros((x.shape[0], n_thresholds), dtype=np.float64)
    first = encoded == 0
    last = encoded == n_thresholds
    probability[first] = q[first, 0]
    dp_eta[first] = -qp[first, 0]
    dp_threshold[first, 0] = qp[first, 0]
    probability[last] = 1.0 - q[last, -1]
    dp_eta[last] = qp[last, -1]
    dp_threshold[last, -1] = -qp[last, -1]
    for index in range(1, n_thresholds):
        middle = encoded == index
        probability[middle] = q[middle, index] - q[middle, index - 1]
        dp_eta[middle] = -qp[middle, index] + qp[middle, index - 1]
        dp_threshold[middle, index] = qp[middle, index]
        dp_threshold[middle, index - 1] = -qp[middle, index - 1]
    probability = np.maximum(probability, np.finfo(np.float64).tiny)
    weight_sum = weights.sum()
    value = float((-weights * np.log(probability)).sum() / weight_sum +
                 0.5 * L2 * np.dot(beta, beta))
    residual = -weights / (weight_sum * probability)
    eta_gradient = residual * dp_eta
    gradient = np.empty_like(theta)
    gradient[:n_features] = x.T @ eta_gradient + L2 * beta
    gradient[n_features] = eta_gradient.sum()
    threshold_gradient = (residual[:, None] * dp_threshold).sum(axis=0)
    gradient[n_features + 1] = threshold_gradient.sum()
    if n_thresholds > 1:
        gradient[n_features + 2:] = np.exp(raw[1:]) * np.cumsum(
            threshold_gradient[:0:-1]
        )[::-1]
    return value, gradient


def oracle(x: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if minimize is None:
        raise RuntimeError("SciPy is required for the ordinal independent oracle")
    classes = np.unique(labels)
    encoded = np.searchsorted(classes, labels)
    cumulative = np.cumsum([weights[encoded == j].sum() for j in range(classes.size - 1)])
    cumulative = np.clip(cumulative / weights.sum(), 0.05, 0.95)
    initial_thresholds = np.log(cumulative / (1.0 - cumulative))
    raw = np.empty(classes.size - 1)
    raw[0] = initial_thresholds[0]
    raw[1:] = np.log(np.maximum(np.diff(initial_thresholds), 1.0e-3))
    initial = np.r_[np.zeros(x.shape[1] + 1), raw]
    result = minimize(
        lambda value: objective_and_gradient(value, x, classes, encoded, weights),
        initial, method="L-BFGS-B", jac=True,
        options={"maxiter": 1000, "ftol": 1.0e-15, "gtol": 1.0e-9, "maxls": 50},
    )
    if not result.success:
        raise RuntimeError(f"independent ordinal oracle failed: {result.message}")
    theta = result.x
    thresholds = raw_thresholds(theta[x.shape[1] + 1:])
    eta = x @ theta[:x.shape[1]] + theta[x.shape[1]]
    q = sigmoid(thresholds[None, :] - eta[:, None])
    probabilities = np.column_stack((q[:, 0], np.diff(q, axis=1), 1.0 - q[:, -1]))
    predicted = classes[np.argmax(probabilities, axis=1)]
    return probabilities, predicted


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "scipy_version": "unavailable" if scipy is None else scipy.__version__,
        "fortml_revision": revision(fortml), "benchmark_revision": revision(root, (output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "workload": "ordinal_logistic_classifier", "phase": "", "backend": "",
        "device": "cpu", "status": "", "n_samples": N_SAMPLES,
        "n_features": N_FEATURES, "n_classes": N_CLASSES, "repetitions": "",
        "seconds_per_operation": "", "accuracy": "", "max_abs_error": "",
        "oracle": "", "notes": "",
    }
    result.update(details)
    result.update(values)
    return result


def run_numpy(details: dict[str, str]) -> list[dict[str, Any]]:
    x, labels, weights = fixture()
    probabilities, predicted = oracle(x, labels, weights)
    started = time.perf_counter()
    for _ in range(PREDICT_REPETITIONS):
        oracle(x, labels, weights)
    seconds = (time.perf_counter() - started) / PREDICT_REPETITIONS
    accuracy = float(np.mean(predicted == labels))
    return [row(details, phase="fit_predict", backend="numpy_oracle", status="pass",
                repetitions=PREDICT_REPETITIONS, seconds_per_operation=seconds,
                accuracy=accuracy, max_abs_error=0.0,
                oracle="independent SciPy L-BFGS-B cumulative-logit oracle",
                notes="weighted objective and complete ordered probability matrix")]


def parse_fortran(stdout: str) -> dict[str, float]:
    values: dict[str, float] = {}
    pattern = re.compile(r"^(ordinal_logistic_(?:fit|predict)),[^,]+,[^,]+,[^,]+,(.+)$")
    for line in stdout.splitlines():
        match = pattern.match(line.strip())
        if match:
            values[match.group(1)] = float(match.group(2))
    return values


def read_fortran(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.full(N_SAMPLES, -2, dtype=np.int64)
    predicted = np.full(N_SAMPLES, -2, dtype=np.int64)
    probabilities = np.full((N_SAMPLES, N_CLASSES), np.nan, dtype=np.float64)
    classes = np.full(N_CLASSES, -2, dtype=np.int64)
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            quantity = record["quantity"]
            i = int(record["row"]) - 1
            j = int(record["column"]) - 1
            value = float(record["value"]) if record["value"] is not None else None
            if quantity == "label":
                labels[i] = int(value)
            elif quantity == "prediction":
                predicted[i] = int(value)
            elif quantity == "probability":
                probabilities[i, j] = value
            elif quantity == "class":
                classes[i] = int(record["column"])
            else:
                raise RuntimeError(f"unknown FortML quantity {quantity!r}")
    if np.any(labels == -2) or np.any(predicted == -2) or np.isnan(probabilities).any() or np.any(classes == -2):
        raise RuntimeError("FortML omitted ordinal outputs")
    return labels, predicted, probabilities, classes


def run_fortml(fortml: Path, details: dict[str, str]) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment, check=True)
    x, labels, weights = fixture()
    expected_probabilities, expected_predicted = oracle(x, labels, weights)
    with tempfile.TemporaryDirectory(dir="/mnt/storage/code/lazy-fortran/fortml/build") as directory:
        oracle_path = Path(directory) / "ordinal_oracle.csv"
        environment["FORTML_BENCH_ORDINAL_ORACLE"] = str(oracle_path)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", "fortml_bench_ordinal_logistic"],
            cwd=fortml, env=environment, capture_output=True, text=True, check=True,
        )
        actual_labels, actual_predicted, actual_probabilities, actual_classes = read_fortran(oracle_path)
    if not np.array_equal(actual_labels, labels) or not np.array_equal(actual_classes, np.unique(labels)):
        raise RuntimeError("FortML ordinal fixture or class order differs from NumPy")
    error = max(float(np.max(np.abs(actual_probabilities - expected_probabilities))),
                float(np.max(actual_predicted != expected_predicted)))
    if error > 3.0e-5:
        raise RuntimeError(f"FortML ordinal oracle mismatch {error:.3e}")
    values = parse_fortran(completed.stdout)
    accuracy = float(np.mean(actual_predicted == labels))
    return [
        row(details, phase="fit", backend="fortml", status="pass" if "ordinal_logistic_fit" in values else "parse_failed",
            repetitions=1, seconds_per_operation=values.get("ordinal_logistic_fit", float("nan")),
            accuracy=accuracy, max_abs_error=error,
            oracle="independent SciPy L-BFGS-B cumulative-logit oracle",
            notes="complete-array release app; weighted fit"),
        row(details, phase="predict", backend="fortml", status="pass" if "ordinal_logistic_predict" in values else "parse_failed",
            repetitions=PREDICT_REPETITIONS, seconds_per_operation=values.get("ordinal_logistic_predict", float("nan")),
            accuracy=accuracy, max_abs_error=error,
            oracle="independent SciPy L-BFGS-B cumulative-logit oracle",
            notes="ordered probabilities and labels checked before timing"),
    ]


def device_refusal_rows(details: dict[str, str]) -> list[dict[str, Any]]:
    return [row({**details, "device": "cuda"}, phase=phase, backend="fortml", status="unavailable",
                oracle="FortML device capability boundary; no CUDA execution",
                notes="device_supported(CUDA)=false; no resident ordinal kernel")
            for phase in ("predict", "predict_proba")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/ordinal_logistic.csv"))
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    details = metadata(root, arguments.fortml.resolve(), arguments.output)
    rows = run_numpy(details) + run_fortml(arguments.fortml.resolve(), details) + device_refusal_rows(details)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: item.get(field, "") for field in FIELDS} for item in rows)


if __name__ == "__main__":
    main()
