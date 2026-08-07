#!/usr/bin/env python3
"""Benchmark the deterministic FortML one-vs-one logistic classifier.

The NumPy implementation is an independent pairwise logistic/Newton oracle.
It checks the complete FortML output array, including the explicit pairwise
vote probability policy.  A scikit-learn row is contextual: its pairwise
coupling probability policy is intentionally not treated as the FortML oracle.
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


N_SAMPLES = 256
N_FEATURES = 6
CLASS_LABELS = np.array([-7, 3, 11, 42], dtype=np.int64)
L2 = 5.0e-2
N_PAIRS = 6
FIT_REPETITIONS = 8
PREDICT_REPETITIONS = 64

FIELDS = (
    "workload",
    "phase",
    "backend",
    "device",
    "status",
    "n_samples",
    "n_features",
    "n_classes",
    "n_pairs",
    "repetitions",
    "seconds_per_operation",
    "accuracy",
    "probability_normalization_error",
    "max_abs_error",
    "oracle",
    "python_version",
    "numpy_version",
    "sklearn_version",
    "fortml_revision",
    "benchmark_revision",
    "compiler",
    "flags",
    "notes",
)


def revision(repository: Path, ignored_paths: tuple[Path, ...] = ()) -> str:
    value = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    status_lines = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines()
    ignored = {path.resolve() for path in ignored_paths}
    dirty = []
    for line in status_lines:
        path_text = line[3:].split(" -> ")[-1].strip()
        if (repository / path_text).resolve() not in ignored:
            dirty.append(line)
    return value + ("+dirty" if dirty else "")


def package_version(name: str) -> str:
    try:
        module = __import__(name)
    except ImportError:
        return "unavailable"
    return str(getattr(module, "__version__", "installed"))


def fixture() -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(1, N_SAMPLES + 1, dtype=np.float64)[:, None]
    columns = np.arange(1, N_FEATURES + 1, dtype=np.float64)[None, :]
    x = np.sin(0.017 * rows + 0.071 * columns)
    x += 0.2 * np.cos(0.009 * rows * columns)
    phase = rows[:, 0]
    scores = np.column_stack(
        (
            0.4 * x[:, 0] - 0.2 * x[:, 1] + 0.1 * x[:, 2],
            -0.1 * x[:, 0] + 0.5 * x[:, 1] - 0.2 * x[:, 3],
            0.2 * x[:, 2] + 0.3 * x[:, 4] - 0.4 * x[:, 5],
            -0.3 * x[:, 0] + 0.2 * x[:, 3] + 0.4 * x[:, 5],
        )
    )
    bias = np.column_stack(
        (
            0.3 * np.sin(0.11 * phase),
            0.3 * np.cos(0.11 * phase),
            0.3 * np.sin(0.13 * phase + 1.0),
            0.2 * np.cos(0.07 * phase + 0.4),
        )
    )
    return x, CLASS_LABELS[np.argmax(scores + bias, axis=1)]


def sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def binary_objective(
    theta: np.ndarray, design: np.ndarray, target: np.ndarray
) -> float:
    score = design @ theta
    data = np.sum(
        np.where(target > 0.5, np.logaddexp(0.0, -score), np.logaddexp(0.0, score))
    )
    return float(data / target.size + 0.5 * L2 * np.sum(theta[:-1] ** 2))


def binary_newton(x: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.column_stack((x, np.ones(x.shape[0])))
    theta = np.zeros(design.shape[1], dtype=np.float64)
    regularizer = np.diag(np.r_[np.full(x.shape[1], L2), 0.0])
    for _ in range(120):
        probability = sigmoid(design @ theta)
        residual = probability - target
        gradient = design.T @ residual / target.size + regularizer @ theta
        hessian = (
            (design.T * (probability * (1.0 - probability))) @ design / target.size
        )
        hessian += regularizer + 1.0e-10 * np.eye(design.shape[1])
        direction = np.linalg.solve(hessian, gradient)
        value = binary_objective(theta, design, target)
        step = 1.0
        while step > 1.0e-10:
            candidate = theta - step * direction
            if binary_objective(candidate, design, target) <= value:
                theta = candidate
                break
            step *= 0.5
        if np.linalg.norm(gradient, ord=np.inf) < 1.0e-10:
            break
    return theta


def ovo_oracle(x: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.zeros((x.shape[0], CLASS_LABELS.size), dtype=np.float64)
    pair_count = 0
    for negative in range(CLASS_LABELS.size - 1):
        for positive in range(negative + 1, CLASS_LABELS.size):
            selected = (labels == CLASS_LABELS[negative]) | (
                labels == CLASS_LABELS[positive]
            )
            pair_x = x[selected]
            pair_labels = (labels[selected] == CLASS_LABELS[positive]).astype(
                np.float64
            )
            theta = binary_newton(pair_x, pair_labels)
            pair_probability = sigmoid(
                np.column_stack((x, np.ones(x.shape[0]))) @ theta
            )
            probabilities[:, negative] += 1.0 - pair_probability
            probabilities[:, positive] += pair_probability
            pair_count += 1
    probabilities /= float(pair_count)
    predicted = CLASS_LABELS[np.argmax(probabilities, axis=1)]
    return probabilities, predicted


def checked_metrics(
    labels: np.ndarray, predicted: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    if predicted.shape != labels.shape or probabilities.shape != (
        N_SAMPLES,
        CLASS_LABELS.size,
    ):
        raise RuntimeError("OVO output shape differs from the fixture")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise RuntimeError("OVO probabilities are not finite/nonnegative")
    normalization_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1.0)))
    if normalization_error > 2.0e-14:
        raise RuntimeError(
            f"OVO probability normalization error {normalization_error:.3e}"
        )
    if not np.isin(predicted, CLASS_LABELS).all():
        raise RuntimeError("OVO predictions contain an unknown class")
    return {
        "accuracy": float(np.mean(predicted == labels)),
        "probability_normalization_error": normalization_error,
    }


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": package_version("sklearn"),
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
        "device": "cpu",
    }


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        **details,
        "workload": "ovo_logistic_classifier",
        "phase": "",
        "backend": "",
        "status": "",
        "n_samples": N_SAMPLES,
        "n_features": N_FEATURES,
        "n_classes": CLASS_LABELS.size,
        "n_pairs": N_PAIRS,
        "repetitions": "",
        "seconds_per_operation": "",
        "accuracy": "",
        "probability_normalization_error": "",
        "max_abs_error": "",
        "oracle": "",
        "notes": "",
    }
    result.update(values)
    return result


def parse_fortran(stdout: str) -> dict[str, float]:
    values: dict[str, float] = {}
    pattern = re.compile(
        r"^(ovo_logistic_(?:fit|predict)),([^,]+),([^,]+),([^,]+),([^,]+),(.+)$"
    )
    for line in stdout.splitlines():
        match = pattern.match(line.strip())
        if match:
            values[match.group(1)] = float(match.group(6))
    return values


def read_oracle(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.zeros(N_SAMPLES, dtype=np.int64)
    predicted = np.zeros(N_SAMPLES, dtype=np.int64)
    probabilities = np.full((N_SAMPLES, CLASS_LABELS.size), np.nan)
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            index = int(record["row"]) - 1
            column = int(record["column"]) - 1
            value = float(record["value"])
            if record["quantity"] == "label":
                labels[index] = int(value)
            elif record["quantity"] == "prediction":
                predicted[index] = int(value)
            elif record["quantity"] == "probability":
                probabilities[index, column] = value
            else:
                raise RuntimeError(
                    f"unknown FortML OVO oracle quantity {record['quantity']!r}"
                )
    if np.isnan(probabilities).any():
        raise RuntimeError("FortML OVO oracle omitted probabilities")
    return labels, predicted, probabilities


def run_fortml(fortml: Path, details: dict[str, str]) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update(
        {"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"}
    )
    subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment, check=True
    )
    x, labels = fixture()
    expected_probabilities, expected_prediction = ovo_oracle(x, labels)
    with tempfile.TemporaryDirectory() as directory:
        oracle_path = Path(directory) / "ovo_oracle.csv"
        environment["FORTML_BENCH_OVO_ORACLE"] = str(oracle_path)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", "fortml_bench_ovo_classifier"],
            cwd=fortml,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        actual_labels, actual_prediction, actual_probabilities = read_oracle(
            oracle_path
        )
    if not np.array_equal(actual_labels, labels):
        raise RuntimeError("FortML OVO fixture labels differ from NumPy")
    error = max(
        float(np.max(np.abs(actual_probabilities - expected_probabilities))),
        float(np.max(actual_prediction != expected_prediction)),
    )
    if error > 3.0e-5:
        raise RuntimeError(f"FortML OVO oracle mismatch {error:.3e}")
    metrics = checked_metrics(labels, actual_prediction, actual_probabilities)
    values = parse_fortran(completed.stdout)
    rows = []
    for phase, key in (
        ("fit", "ovo_logistic_fit"),
        ("predict", "ovo_logistic_predict"),
    ):
        rows.append(
            row(
                details,
                phase=phase,
                backend="fortml",
                status="pass" if key in values else "parse_failed",
                repetitions=1 if phase == "fit" else PREDICT_REPETITIONS,
                seconds_per_operation=values.get(key, float("nan")),
                **metrics,
                max_abs_error=error,
                oracle="independent NumPy pairwise logistic/Newton vote oracle",
                notes="pair order lexicographic; probability policy divides pair votes by n_pairs",
            )
        )
    return rows


def run_numpy(details: dict[str, str]) -> list[dict[str, Any]]:
    x, labels = fixture()
    started = time.perf_counter()
    for _ in range(FIT_REPETITIONS):
        probabilities, predicted = ovo_oracle(x, labels)
    fit_seconds = (time.perf_counter() - started) / FIT_REPETITIONS
    started = time.perf_counter()
    for _ in range(PREDICT_REPETITIONS):
        probabilities, predicted = ovo_oracle(x, labels)
    predict_seconds = (time.perf_counter() - started) / PREDICT_REPETITIONS
    metrics = checked_metrics(labels, predicted, probabilities)
    return [
        row(
            details,
            phase="fit",
            backend="numpy_oracle",
            status="pass",
            repetitions=FIT_REPETITIONS,
            seconds_per_operation=fit_seconds,
            **metrics,
            max_abs_error=0.0,
            oracle="independent NumPy pairwise logistic/Newton vote oracle",
            notes="behavioral reference; no FortML implementation calls",
        ),
        row(
            details,
            phase="predict",
            backend="numpy_oracle",
            status="pass",
            repetitions=PREDICT_REPETITIONS,
            seconds_per_operation=predict_seconds,
            **metrics,
            max_abs_error=0.0,
            oracle="independent NumPy pairwise logistic/Newton vote oracle",
            notes="recomputes deterministic pair models for an independent timing lane",
        ),
    ]


def run_sklearn(details: dict[str, str]) -> list[dict[str, Any]]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsOneClassifier
    except ImportError as error:
        return [
            row(
                details,
                phase="fit",
                backend="sklearn",
                status="unavailable",
                notes=str(error),
            )
        ]
    x, labels = fixture()
    model = OneVsOneClassifier(
        LogisticRegression(C=1.0 / L2, max_iter=1000, solver="lbfgs")
    )
    started = time.perf_counter()
    model.fit(x, labels)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predicted = model.predict(x)
    predict_seconds = time.perf_counter() - started
    probabilities = np.zeros((N_SAMPLES, CLASS_LABELS.size))
    probabilities[np.arange(N_SAMPLES), np.searchsorted(CLASS_LABELS, predicted)] = 1.0
    metrics = checked_metrics(labels, predicted, probabilities)
    return [
        row(
            details,
            phase="fit",
            backend="sklearn",
            status="pass",
            repetitions=1,
            seconds_per_operation=fit_seconds,
            **metrics,
            max_abs_error="",
            oracle="contextual scikit-learn OneVsOneClassifier",
            notes="CPU contextual timing; sklearn pairwise coupling policy is separate",
        ),
        row(
            details,
            phase="predict",
            backend="sklearn",
            status="pass",
            repetitions=1,
            seconds_per_operation=predict_seconds,
            **metrics,
            max_abs_error="",
            oracle="contextual scikit-learn OneVsOneClassifier",
            notes="hard one-hot probabilities only for shared label metrics",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/ovo_logistic.csv"))
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    details = metadata(root, arguments.fortml.resolve(), arguments.output)
    rows = (
        run_numpy(details)
        + run_fortml(arguments.fortml.resolve(), details)
        + run_sklearn(details)
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: item.get(field, "") for field in FIELDS} for item in rows
        )


if __name__ == "__main__":
    main()
