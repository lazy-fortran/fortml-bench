#!/usr/bin/env python3
"""Benchmark dense multilabel-indicator logistic heads.

The NumPy implementation is an independent damped-Newton oracle for the
declared per-label weighted logistic objective.  FortML must emit every
probability and hard indicator before a timing row is retained.  CUDA is an
explicit capability boundary; no host fallback is timed as accelerator work.
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


N_SAMPLES = 192
N_FEATURES = 4
N_LABELS = 3
L2 = 5.0e-2
FIT_REPETITIONS = 8
PREDICT_REPETITIONS = 64

FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_labels", "repetitions", "seconds_per_operation",
    "hamming_accuracy", "max_abs_error", "oracle", "python_version",
    "numpy_version", "sklearn_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
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


def package_version(name: str) -> str:
    try:
        module = __import__(name)
    except ImportError:
        return "unavailable"
    return str(getattr(module, "__version__", "installed"))


def fixture() -> tuple[np.ndarray, np.ndarray]:
    phase = np.arange(1, N_SAMPLES + 1, dtype=np.float64)
    columns = np.arange(1, N_FEATURES + 1, dtype=np.float64)[None, :]
    x = np.sin(0.021 * phase[:, None] + 0.083 * columns)
    x += 0.15 * np.cos(0.011 * phase[:, None] * columns)
    indicators = np.empty((N_SAMPLES, N_LABELS), dtype=np.int64)
    score = 0.8 * x[:, 0] - 0.45 * x[:, 1] + 0.15 * np.sin(0.13 * phase)
    indicators[:, 0] = score > 0.0
    score = -0.35 * x[:, 0] + 0.7 * x[:, 2] - 0.1 * np.cos(0.09 * phase + 0.3)
    indicators[:, 1] = score > 0.0
    score = 0.3 * x[:, 1] + 0.55 * x[:, 3] + 0.2 * np.sin(0.07 * phase + 0.5)
    indicators[:, 2] = score > 0.0
    return x, indicators.astype(np.int64)


def sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def objective(theta: np.ndarray, design: np.ndarray, target: np.ndarray) -> float:
    score = design @ theta
    value = np.where(
        target > 0.5, np.logaddexp(0.0, -score), np.logaddexp(0.0, score)
    ).sum() / target.size
    return float(value + 0.5 * L2 * np.dot(theta[:-1], theta[:-1]))


def binary_newton(x: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.column_stack((x, np.ones(x.shape[0])))
    theta = np.zeros(design.shape[1], dtype=np.float64)
    regularizer = np.diag(np.r_[np.full(x.shape[1], L2), 0.0])
    for _ in range(200):
        probability = sigmoid(design @ theta)
        residual = probability - target
        curvature = probability * (1.0 - probability)
        gradient = design.T @ residual / target.size + regularizer @ theta
        hessian = (design.T * curvature) @ design / target.size + regularizer
        hessian += 1.0e-10 * np.eye(design.shape[1])
        direction = np.linalg.solve(hessian, gradient)
        value = objective(theta, design, target)
        step = 1.0
        while step > 1.0e-10:
            candidate = theta - step * direction
            if objective(candidate, design, target) <= value:
                theta = candidate
                break
            step *= 0.5
        if np.max(np.abs(gradient)) < 1.0e-10:
            break
    return theta


def oracle(x: np.ndarray, indicators: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack((x, np.ones(x.shape[0])))
    probabilities = np.empty((x.shape[0], indicators.shape[1]), dtype=np.float64)
    for label in range(indicators.shape[1]):
        theta = binary_newton(x, indicators[:, label].astype(np.float64))
        probabilities[:, label] = sigmoid(design @ theta)
    return probabilities, (probabilities >= 0.5).astype(np.int64)


def checked_metrics(indicators: np.ndarray, probabilities: np.ndarray,
                    predicted: np.ndarray) -> dict[str, float]:
    if probabilities.shape != indicators.shape or predicted.shape != indicators.shape:
        raise RuntimeError("multilabel outputs have invalid shapes")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0) or \
            np.any(probabilities > 1.0):
        raise RuntimeError("multilabel probabilities are invalid")
    if not np.isin(predicted, [0, 1]).all():
        raise RuntimeError("multilabel predictions are not binary indicators")
    return {"hamming_accuracy": float(np.mean(predicted == indicators))}


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": package_version("sklearn"),
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "workload": "multilabel_logistic_classifier", "phase": "", "backend": "",
        "device": "cpu", "status": "", "n_samples": N_SAMPLES,
        "n_features": N_FEATURES, "n_labels": N_LABELS, "repetitions": "",
        "seconds_per_operation": "", "hamming_accuracy": "", "max_abs_error": "",
        "oracle": "", "notes": "",
    }
    result.update(details)
    result.update(values)
    return result


def parse_fortran(stdout: str) -> dict[str, float]:
    values: dict[str, float] = {}
    pattern = re.compile(r"^(multilabel_logistic_(?:fit|predict)),[^,]+,[^,]+,[^,]+,(.+)$")
    for line in stdout.splitlines():
        match = pattern.match(line.strip())
        if match:
            values[match.group(1)] = float(match.group(2))
    return values


def read_fortran(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.full((N_SAMPLES, N_LABELS), -1, dtype=np.int64)
    predicted = np.full_like(labels, -1)
    probabilities = np.full(labels.shape, np.nan, dtype=np.float64)
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            i = int(record["row"]) - 1
            j = int(record["column"]) - 1
            value = float(record["value"])
            if record["quantity"] == "label":
                labels[i, j] = int(value)
            elif record["quantity"] == "prediction":
                predicted[i, j] = int(value)
            elif record["quantity"] == "probability":
                probabilities[i, j] = value
            else:
                raise RuntimeError(f"unknown FortML quantity {record['quantity']!r}")
    if np.any(labels < 0) or np.any(predicted < 0) or np.isnan(probabilities).any():
        raise RuntimeError("FortML omitted multilabel outputs")
    return labels, predicted, probabilities


def run_numpy(details: dict[str, str]) -> list[dict[str, Any]]:
    x, indicators = fixture()
    expected_probabilities, expected_predicted = oracle(x, indicators)
    started = time.perf_counter()
    for _ in range(FIT_REPETITIONS):
        oracle(x, indicators)
    fit_seconds = (time.perf_counter() - started) / FIT_REPETITIONS
    started = time.perf_counter()
    for _ in range(PREDICT_REPETITIONS):
        expected_probabilities, expected_predicted = oracle(x, indicators)
    predict_seconds = (time.perf_counter() - started) / PREDICT_REPETITIONS
    metrics = checked_metrics(indicators, expected_probabilities, expected_predicted)
    return [
        row(details, phase="fit", backend="numpy_oracle", status="pass",
            repetitions=FIT_REPETITIONS, seconds_per_operation=fit_seconds,
            **metrics, max_abs_error=0.0,
            oracle="independent NumPy per-label logistic/Newton oracle",
            notes="behavioral reference; no FortML calls"),
        row(details, phase="predict", backend="numpy_oracle", status="pass",
            repetitions=PREDICT_REPETITIONS, seconds_per_operation=predict_seconds,
            **metrics, max_abs_error=0.0,
            oracle="independent NumPy per-label logistic/Newton oracle",
            notes="recomputes the heads for an independent timing lane"),
    ]


def run_fortml(fortml: Path, details: dict[str, str]) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment, check=True)
    x, indicators = fixture()
    expected_probabilities, expected_predicted = oracle(x, indicators)
    with tempfile.TemporaryDirectory(dir="/mnt/storage/code/lazy-fortran/fortml/build") as directory:
        oracle_path = Path(directory) / "multilabel_oracle.csv"
        environment["FORTML_BENCH_MULTILABEL_ORACLE"] = str(oracle_path)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", "fortml_bench_multilabel_logistic"],
            cwd=fortml, env=environment, capture_output=True, text=True, check=True,
        )
        actual_labels, actual_predicted, actual_probabilities = read_fortran(oracle_path)
    if not np.array_equal(actual_labels, indicators):
        raise RuntimeError("FortML multilabel fixture differs from NumPy")
    error = max(
        float(np.max(np.abs(actual_probabilities - expected_probabilities))),
        float(np.max(actual_predicted != expected_predicted)),
    )
    if error > 3.0e-5:
        raise RuntimeError(f"FortML multilabel oracle mismatch {error:.3e}")
    metrics = checked_metrics(indicators, actual_probabilities, actual_predicted)
    values = parse_fortran(completed.stdout)
    return [
        row(details, phase="fit", backend="fortml", status="pass" if "multilabel_logistic_fit" in values else "parse_failed",
            repetitions=1, seconds_per_operation=values.get("multilabel_logistic_fit", float("nan")),
            **metrics, max_abs_error=error,
            oracle="independent NumPy per-label logistic/Newton oracle",
            notes="complete-array release app; no GPU fallback"),
        row(details, phase="predict", backend="fortml", status="pass" if "multilabel_logistic_predict" in values else "parse_failed",
            repetitions=PREDICT_REPETITIONS,
            seconds_per_operation=values.get("multilabel_logistic_predict", float("nan")),
            **metrics, max_abs_error=error,
            oracle="independent NumPy per-label logistic/Newton oracle",
            notes="positive probability matrix and hard indicator matrix checked"),
    ]


def run_sklearn(details: dict[str, str]) -> list[dict[str, Any]]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.multioutput import MultiOutputClassifier
    except ImportError as error:
        return [row(details, phase="fit", backend="sklearn", status="unavailable", notes=str(error))]
    x, indicators = fixture()
    model = MultiOutputClassifier(LogisticRegression(C=1.0 / L2, max_iter=1000))
    started = time.perf_counter()
    model.fit(x, indicators)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predicted = model.predict(x)
    predict_seconds = time.perf_counter() - started
    probabilities = np.column_stack([value[:, 1] for value in model.predict_proba(x)])
    metrics = checked_metrics(indicators, probabilities, predicted)
    return [
        row(details, phase="fit", backend="sklearn", status="pass", repetitions=1,
            seconds_per_operation=fit_seconds, **metrics, max_abs_error="",
            oracle="contextual scikit-learn MultiOutputClassifier",
            notes="CPU contextual timing"),
        row(details, phase="predict", backend="sklearn", status="pass", repetitions=1,
            seconds_per_operation=predict_seconds, **metrics, max_abs_error="",
            oracle="contextual scikit-learn MultiOutputClassifier",
            notes="positive probability columns checked"),
    ]


def device_refusal_rows(details: dict[str, str]) -> list[dict[str, Any]]:
    cuda_details = {**details, "device": "cuda"}
    return [
        row(cuda_details, phase=phase, backend="fortml", status="unavailable",
            oracle="FortML device capability boundary; no CUDA execution",
            notes="device_supported(CUDA)=false; no resident multilabel kernel")
        for phase in ("predict", "predict_proba", "predict_proba_jvp",
                      "predict_proba_vjp", "predict_proba_parameter_jvp",
                      "predict_proba_parameter_vjp")
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/multilabel_logistic.csv"))
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    details = metadata(root, arguments.fortml.resolve(), arguments.output)
    rows = (run_numpy(details) + run_fortml(arguments.fortml.resolve(), details) +
            run_sklearn(details) + device_refusal_rows(details))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: item.get(field, "") for field in FIELDS} for item in rows)


if __name__ == "__main__":
    main()
