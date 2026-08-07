#!/usr/bin/env python3
"""Benchmark the binary logistic slice against an independent NumPy oracle."""

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

import numpy as np


N_SAMPLES = 1024
N_FEATURES = 8
L2 = 1.0e-3
CLASS_LABELS = np.array([-3, 7], dtype=np.int64)


def inputs() -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(1, N_SAMPLES + 1, dtype=np.float64)[:, None]
    columns = np.arange(1, N_FEATURES + 1, dtype=np.float64)[None, :]
    x = np.sin(0.013 * rows + 0.071 * columns)
    x += 0.2 * np.cos(0.009 * rows * columns)
    true = -0.15 + x @ np.sin(0.17 * np.arange(1, N_FEATURES + 1))
    labels = np.where(true >= 0.0, 7, -3).astype(np.int64)
    return x, labels


def git_revision(repository: Path, ignored_paths: tuple[Path, ...] = ()) -> str:
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines()
    ignored = {path.resolve() for path in ignored_paths}
    dirty = []
    for line in status:
        path_text = line[3:].split(" -> ")[-1].strip()
        if (repository / path_text).resolve() not in ignored:
            dirty.append(line)
    return revision + ("+dirty" if dirty else "")


def parse_fortran(stdout: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in stdout.splitlines():
        match = re.match(
            r"^(logistic_fit|logistic_predict|logistic_accuracy),([^,]+),([^,]+),([^,]+),(.+)$",
            line,
        )
        if match:
            values[match.group(1)] = float(match.group(5))
        match = re.match(r"^logistic_accuracy,(.*)$", line)
        if match:
            values["logistic_accuracy"] = float(match.group(1))
    return values


def read_fortran_oracle(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = np.zeros(N_SAMPLES, dtype=np.int64)
    predicted = np.zeros(N_SAMPLES, dtype=np.int64)
    probabilities = np.zeros((N_SAMPLES, 2), dtype=np.float64)
    target_seen = np.zeros(N_SAMPLES, dtype=bool)
    prediction_seen = np.zeros(N_SAMPLES, dtype=bool)
    probability_seen = np.zeros((N_SAMPLES, 2), dtype=bool)
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            row = int(record["row"]) - 1
            column = int(record["column"]) - 1
            value = float(record["value"])
            if not 0 <= row < N_SAMPLES:
                raise RuntimeError("FortML oracle contains an invalid row")
            if record["quantity"] == "target":
                targets[row] = int(value)
                target_seen[row] = True
            elif record["quantity"] == "prediction":
                predicted[row] = int(value)
                prediction_seen[row] = True
            elif record["quantity"] == "probability":
                if not 0 <= column < 2:
                    raise RuntimeError("FortML oracle contains an invalid class column")
                probabilities[row, column] = value
                probability_seen[row, column] = True
            else:
                raise RuntimeError("FortML oracle contains an unknown quantity")
    if not target_seen.all() or not prediction_seen.all() or not probability_seen.all():
        raise RuntimeError("FortML oracle is incomplete")
    return targets, predicted, probabilities


def checked_metrics(
    labels: np.ndarray, predicted: np.ndarray, probabilities: np.ndarray
) -> dict[str, object]:
    if labels.shape != (N_SAMPLES,) or predicted.shape != labels.shape:
        raise RuntimeError("classification metric label shape is invalid")
    if probabilities.shape != (N_SAMPLES, 2):
        raise RuntimeError("classification probability shape is invalid")
    if not np.isfinite(probabilities).all():
        raise RuntimeError("classification probabilities are not finite")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise RuntimeError("classification probabilities lie outside [0,1]")
    if not np.isin(labels, CLASS_LABELS).all():
        raise RuntimeError("classification targets contain an unknown label")
    if not np.isin(predicted, CLASS_LABELS).all():
        raise RuntimeError("classification predictions contain an unknown label")

    normalization_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1.0)))
    target_columns = (labels == CLASS_LABELS[1]).astype(np.int64)
    selected = probabilities[np.arange(N_SAMPLES), target_columns]
    selected = np.clip(selected, np.finfo(np.float64).eps, 1.0)
    numpy_accuracy = float(np.mean(predicted == labels))
    numpy_log_loss = float(-np.mean(np.log(selected)))
    numpy_confusion = np.array(
        [
            [
                np.count_nonzero(
                    (labels == CLASS_LABELS[actual])
                    & (predicted == CLASS_LABELS[reported])
                )
                for reported in range(2)
            ]
            for actual in range(2)
        ],
        dtype=np.int64,
    )

    from sklearn.metrics import accuracy_score, confusion_matrix, log_loss

    sklearn_accuracy = float(accuracy_score(labels, predicted))
    sklearn_log_loss = float(log_loss(labels, probabilities, labels=CLASS_LABELS))
    sklearn_confusion = confusion_matrix(labels, predicted, labels=CLASS_LABELS)
    metric_error = max(
        abs(numpy_accuracy - sklearn_accuracy),
        abs(numpy_log_loss - sklearn_log_loss),
        float(np.max(np.abs(numpy_confusion - sklearn_confusion))),
    )
    if normalization_error > 1.0e-14 or metric_error > 5.0e-14:
        raise RuntimeError("NumPy and scikit classification metrics disagree")
    return {
        "accuracy": numpy_accuracy,
        "log_loss": numpy_log_loss,
        "true_negative": int(numpy_confusion[0, 0]),
        "false_positive": int(numpy_confusion[0, 1]),
        "false_negative": int(numpy_confusion[1, 0]),
        "true_positive": int(numpy_confusion[1, 1]),
        "probability_normalization_error": normalization_error,
        "metric_max_abs_error": metric_error,
    }


def run_fortran(fortml: Path) -> list[dict[str, object]]:
    env = os.environ.copy()
    env.update({"FO_FC": env.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=env, check=True)
    with tempfile.TemporaryDirectory() as directory:
        oracle_path = Path(directory) / "fortml_classification_oracle.csv"
        env["FORTML_BENCH_ORACLE"] = str(oracle_path)
        started = time.perf_counter()
        completed = subprocess.run(
            ["fo", "exec", "--no-build", "fortml_bench_logistic"],
            cwd=fortml,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        wall = time.perf_counter() - started
        oracle_targets, predicted, probabilities = read_fortran_oracle(oracle_path)
    values = parse_fortran(completed.stdout)
    expected_targets = inputs()[1]
    if not np.array_equal(oracle_targets, expected_targets):
        raise RuntimeError("FortML oracle targets differ from the NumPy fixture")
    metrics = checked_metrics(expected_targets, predicted, probabilities)
    if abs(float(metrics["accuracy"]) - values["logistic_accuracy"]) > 5.0e-15:
        raise RuntimeError("FortML reported accuracy differs from full-output metrics")
    rows = []
    for phase, key in (("fit", "logistic_fit"), ("predict", "logistic_predict")):
        rows.append(
            {
                "backend": "fortml",
                "phase": phase,
                "status": "pass" if key in values else "parse_failed",
                "n_samples": N_SAMPLES,
                "n_features": N_FEATURES,
                "seconds_per_operation": values.get(key, float("nan")),
                "accuracy": values.get("logistic_accuracy", float("nan")),
                "oracle": "FortML internal invariant plus NumPy labels",
                "notes": f"fo wall time including process startup={wall:.6e}s",
            }
        )
    rows.append(
        {
            "backend": "fortml",
            "phase": "metrics",
            "status": "pass",
            "n_samples": N_SAMPLES,
            "n_features": N_FEATURES,
            **metrics,
            "oracle": "independent NumPy full-output metrics cross-checked by sklearn.metrics",
            "notes": "class order -3,7; confusion rows=true and columns=predicted",
        }
    )
    return rows


def run_sklearn(x: np.ndarray, labels: np.ndarray) -> list[dict[str, object]]:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return [{"backend": "sklearn", "phase": "fit", "status": "unavailable"}]
    started = time.perf_counter()
    model = LogisticRegression(
        C=1.0 / L2, fit_intercept=True, solver="lbfgs", max_iter=500
    )
    model.fit(x, labels)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predicted = model.predict(x)
    probabilities = model.predict_proba(x)
    predict_seconds = time.perf_counter() - started
    accuracy = float(np.mean(predicted == labels))
    metrics = checked_metrics(labels, predicted, probabilities)
    return [
        {
            "backend": "sklearn",
            "phase": "fit",
            "status": "pass",
            "n_samples": N_SAMPLES,
            "n_features": N_FEATURES,
            "seconds_per_operation": fit_seconds,
            "accuracy": accuracy,
            "oracle": "NumPy generated labels",
            "notes": f"probability normalization error={np.max(np.abs(probabilities.sum(axis=1) - 1.0)):.3e}",
        },
        {
            "backend": "sklearn",
            "phase": "predict",
            "status": "pass",
            "n_samples": N_SAMPLES,
            "n_features": N_FEATURES,
            "seconds_per_operation": predict_seconds,
            "accuracy": accuracy,
            "oracle": "NumPy generated labels",
            "notes": "single-process CPU reference",
        },
        {
            "backend": "sklearn",
            "phase": "metrics",
            "status": "pass",
            "n_samples": N_SAMPLES,
            "n_features": N_FEATURES,
            **metrics,
            "oracle": "independent NumPy full-output metrics cross-checked by sklearn.metrics",
            "notes": "class order -3,7; confusion rows=true and columns=predicted",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/classification_workloads.csv")
    )
    args = parser.parse_args()
    x, labels = inputs()
    rows = run_sklearn(x, labels)
    rows.extend(run_fortran(args.fortml.resolve()))
    output_path = args.output.resolve()
    metadata = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "benchmark_revision": git_revision(Path.cwd(), (output_path,)),
        "fortml_revision": git_revision(args.fortml.resolve()),
    }
    try:
        import sklearn

        metadata["sklearn_version"] = sklearn.__version__
    except ImportError:
        metadata["sklearn_version"] = "unavailable"
    for row in rows:
        row.update(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "backend",
        "phase",
        "status",
        "n_samples",
        "n_features",
        "seconds_per_operation",
        "accuracy",
        "log_loss",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "probability_normalization_error",
        "metric_max_abs_error",
        "oracle",
        "notes",
        "python_version",
        "numpy_version",
        "sklearn_version",
        "fortml_revision",
        "benchmark_revision",
    ]
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
