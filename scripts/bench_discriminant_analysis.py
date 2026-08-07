#!/usr/bin/env python3
"""Correctness-gated weighted LDA/QDA benchmark.

The NumPy implementation below is deliberately independent of FortML: it
forms weighted class moments, applies the declared diagonal regularization,
and evaluates the Gaussian discriminants directly.  The release Fortran app
must emit every probability, class, and prediction before timing rows are
retained.  CUDA is an explicit typed refusal until a resident kernel exists.
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

import numpy as np


N_SAMPLES, N_FEATURES, N_QUERY, N_CLASSES = 120, 3, 9, 3
REG_PARAM = 3.0e-2
PREDICTION_REPETITIONS = 128
CLASSES = np.array([-17, 4, 23], dtype=np.int64)
FIELDS = (
    "workload", "model", "phase", "backend", "device", "status",
    "n_samples", "n_features", "n_query", "seconds_per_operation",
    "accuracy", "max_abs_error", "oracle", "python_version", "numpy_version",
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


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    index = np.arange(1, N_SAMPLES + 1, dtype=np.float64)
    x = np.empty((N_SAMPLES, N_FEATURES), dtype=np.float64)
    for j in range(N_FEATURES):
        x[:, j] = (np.sin(0.031 * index + 0.19 * (j + 1)) +
                   0.11 * np.cos(0.017 * index * (j + 1)))
    score = 0.8 * x[:, 0] - 0.45 * x[:, 1] + 0.25 * x[:, 2]
    labels = np.where(score < -0.25, -17, np.where(score < 0.28, 4, 23)).astype(np.int64)
    weights = 0.7 + 0.6 * (np.mod(index, 7.0) / 6.0)
    query = np.column_stack((
        -1.1 + 2.2 * np.arange(N_QUERY, dtype=np.float64) / (N_QUERY - 1),
        np.sin(0.4 * np.arange(1, N_QUERY + 1, dtype=np.float64)),
        np.cos(0.3 * np.arange(1, N_QUERY + 1, dtype=np.float64)),
    ))
    tangent = np.column_stack((
        0.1 * np.cos(0.2 * np.arange(1, N_QUERY + 1, dtype=np.float64)),
        -0.1 * np.sin(0.3 * np.arange(1, N_QUERY + 1, dtype=np.float64)),
        np.full(N_QUERY, 0.04),
    ))
    return x, labels, weights, query, tangent


def fit_oracle(x: np.ndarray, labels: np.ndarray, weights: np.ndarray,
                model: str) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    means = np.empty((N_CLASSES, N_FEATURES), dtype=np.float64)
    counts = np.empty(N_CLASSES, dtype=np.float64)
    for c, label in enumerate(CLASSES):
        mask = labels == label
        counts[c] = np.sum(weights[mask])
        means[c] = np.sum(weights[mask, None] * x[mask], axis=0) / counts[c]
    covariances: list[np.ndarray] = []
    if model == "lda":
        covariance = np.zeros((N_FEATURES, N_FEATURES), dtype=np.float64)
        for row, label in enumerate(labels):
            delta = x[row] - means[np.flatnonzero(CLASSES == label)[0]]
            covariance += weights[row] * np.outer(delta, delta)
        covariance /= np.sum(weights)
        covariance = (1.0 - REG_PARAM) * covariance + REG_PARAM * np.eye(N_FEATURES)
        covariances = [covariance for _ in range(N_CLASSES)]
    else:
        for c, label in enumerate(CLASSES):
            covariance = np.zeros((N_FEATURES, N_FEATURES), dtype=np.float64)
            for row in np.flatnonzero(labels == label):
                delta = x[row] - means[c]
                covariance += weights[row] * np.outer(delta, delta)
            covariance = covariance / counts[c]
            covariances.append((1.0 - REG_PARAM) * covariance +
                               REG_PARAM * np.eye(N_FEATURES))
    return means, counts / counts.sum(), covariances


def predict_oracle(x: np.ndarray, means: np.ndarray, prior: np.ndarray,
                   covariances: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    log_joint = np.empty((x.shape[0], N_CLASSES), dtype=np.float64)
    for c, covariance in enumerate(covariances):
        precision = np.linalg.inv(covariance)
        _, logdet = np.linalg.slogdet(covariance)
        delta = x - means[c]
        log_joint[:, c] = (np.log(prior[c]) - 0.5 * (N_FEATURES * np.log(2.0 * np.pi) +
                           logdet + np.einsum("ni,ij,nj->n", delta, precision, delta)))
    shifted = log_joint - np.max(log_joint, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities, CLASSES[np.argmax(probabilities, axis=1)]


def parse_app(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for model in ("lda", "qda"):
        probabilities = np.full((N_QUERY, N_CLASSES), np.nan)
        predicted = np.full(N_QUERY, -999, dtype=np.int64)
        classes = np.full(N_CLASSES, -999, dtype=np.int64)
        result[model] = (classes, probabilities, predicted)
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            model = record["model"]
            quantity = record["quantity"]
            row = int(record["row"]) - 1
            column = int(record["column"]) - 1
            value = float(record["value"])
            classes, probabilities, predicted = result[model]
            if quantity == "class":
                classes[row] = int(value)
            elif quantity == "probability":
                probabilities[row, column] = value
            elif quantity == "prediction":
                predicted[row] = int(value)
            elif quantity in {"mean", "covariance"}:
                # Additional fitted-state diagnostics are retained by the
                # release app but are not needed for the prediction oracle.
                continue
            else:
                raise RuntimeError(f"unknown quantity {quantity!r}")
    for model, (classes, probabilities, predicted) in result.items():
        if (not np.array_equal(classes, CLASSES) or np.isnan(probabilities).any() or
                np.any(predicted == -999)):
            raise RuntimeError(f"FortML omitted {model} discriminant output")
    return result


def timings(stdout: str) -> dict[str, tuple[float, float, float]]:
    result: dict[str, tuple[float, float, float]] = {}
    for line in stdout.splitlines():
        fields = line.strip().split(",")
        if len(fields) == 7 and fields[0] in {"lda_fit_predict", "qda_fit_predict"}:
            result[fields[0][:3]] = (float(fields[3]), float(fields[4]), float(fields[5]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/discriminant_analysis.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    x, labels, weights, query, _ = fixture()
    metadata = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (args.output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }

    def row(**values: object) -> dict[str, object]:
        output: dict[str, object] = {
            "workload": "discriminant_analysis", "model": "", "phase": "",
            "backend": "", "device": "cpu", "status": "", "n_samples": N_SAMPLES,
            "n_features": N_FEATURES, "n_query": N_QUERY, "seconds_per_operation": "",
            "accuracy": "", "max_abs_error": "", "oracle": "", "notes": "",
            **metadata,
        }
        output.update(values)
        return output

    rows: list[dict[str, object]] = []
    for model in ("lda", "qda"):
        means, prior, covariances = fit_oracle(x, labels, weights, model)
        expected_probabilities, expected_predictions = predict_oracle(
            query, means, prior, covariances,
        )
        started = time.perf_counter()
        for _ in range(PREDICTION_REPETITIONS):
            predict_oracle(query, means, prior, covariances)
        oracle_seconds = (time.perf_counter() - started) / PREDICTION_REPETITIONS
        rows.append(row(model=model, phase="fit_predict", backend="numpy_oracle", status="pass",
                        seconds_per_operation=oracle_seconds, accuracy=1.0,
                        max_abs_error=0.0,
                        oracle="independent NumPy weighted Gaussian discriminant oracle",
                        notes="direct weighted moments and stabilized log-sum-exp"))

    environment = os.environ.copy()
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment, check=True)
    with tempfile.TemporaryDirectory(dir=fortml / "build") as directory:
        output = Path(directory) / "discriminant.csv"
        environment["FORTML_BENCH_DISCRIMINANT_OUTPUT"] = str(output)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", "fortml_bench_discriminant_analysis"],
            cwd=fortml, env=environment, capture_output=True, text=True, check=True,
        )
        actual = parse_app(output)
    measured = timings(completed.stdout)
    for model in ("lda", "qda"):
        means, prior, covariances = fit_oracle(x, labels, weights, model)
        expected_probabilities, expected_predictions = predict_oracle(
            query, means, prior, covariances,
        )
        _, actual_probabilities, actual_predictions = actual[model]
        error = max(float(np.max(np.abs(actual_probabilities - expected_probabilities))),
                    float(np.max(actual_predictions != expected_predictions)))
        if error > 3.0e-11:
            raise RuntimeError(f"{model} NumPy oracle mismatch: {error:.3e}")
        fit_seconds, predict_seconds, jvp_seconds = measured[model]
        common = dict(model=model, backend="fortml_cpu", status="pass", accuracy=1.0,
                      max_abs_error=error,
                      oracle="independent NumPy weighted Gaussian discriminant oracle")
        rows.extend([
            row(**common, phase="fit", seconds_per_operation=fit_seconds,
                notes="complete class/probability/prediction oracle passed"),
            row(**common, phase="predict", seconds_per_operation=predict_seconds,
                notes="complete class/probability/prediction oracle passed"),
            row(**common, phase="input_jvp", seconds_per_operation=jvp_seconds,
                notes="fixed-state probability JVP"),
            row(model=model, phase="predict", backend="fortml_cuda", device="cuda",
                status="unavailable", oracle="typed_device_contract",
                notes="no resident discriminant-analysis CUDA kernel; FORTNUM_NOT_IMPLEMENTED"),
        ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
