#!/usr/bin/env python3
"""Correctness-gated multiclass softmax-temperature calibration benchmark."""

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
N_CLASSES = 3
L2 = 0.05
FIT_REPETITIONS = 8
PREDICT_REPETITIONS = 128
FIELDS = (
    "workload", "method", "phase", "backend", "device", "status",
    "n_samples", "n_classes", "repetitions", "seconds_per_operation",
    "accuracy", "probability_normalization_error", "temperature",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    value = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return value + ("+dirty" if dirty else "")


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = np.arange(1, N_SAMPLES + 1, dtype=np.float64)
    first = 1.2 * np.sin(0.031 * index) + 0.17 * np.cos(0.013 * index)
    second = 1.1 * np.cos(0.027 * index + 0.4) + 0.12 * np.sin(0.017 * index)
    third = 0.9 * np.sin(0.019 * index + 1.1) - 0.18 * np.cos(0.011 * index)
    scores = np.column_stack((first, second, third))
    class_labels = np.array([-4, 17, 91], dtype=np.int64)
    labels = class_labels[np.argmax(scores, axis=1)]
    return scores, labels, class_labels


def softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    scaled = scores / temperature
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def fit_temperature(scores: np.ndarray, labels: np.ndarray,
                    classes: np.ndarray) -> float:
    encoded = np.searchsorted(classes, labels)
    alpha = 1.0
    probabilities = softmax(scores, 1.0 / alpha)

    def objective(value: float) -> float:
        return float(np.mean(-np.log(softmax(scores, 1.0 / value)[
            np.arange(scores.shape[0]), encoded])) + 0.5 * L2 * value * value)

    current = objective(alpha)
    for _ in range(500):
        mean_score = np.sum(probabilities * scores, axis=1)
        mean_square = np.sum(probabilities * scores * scores, axis=1)
        target_score = scores[np.arange(scores.shape[0]), encoded]
        gradient = float(np.mean(mean_score - target_score) + L2 * alpha)
        hessian = float(np.mean(mean_square - mean_score * mean_score) + L2)
        if hessian <= 0.0 or not np.isfinite(hessian):
            raise RuntimeError("multiclass temperature oracle has invalid curvature")
        step = gradient / hessian
        scale = 1.0
        candidate = max(np.sqrt(np.finfo(np.float64).tiny), alpha - scale * step)
        candidate_value = objective(candidate)
        while candidate_value > current and scale > 1.0e-8:
            scale *= 0.5
            candidate = max(np.sqrt(np.finfo(np.float64).tiny), alpha - scale * step)
            candidate_value = objective(candidate)
        step_norm = abs(candidate - alpha) / max(1.0, abs(alpha))
        alpha, current = candidate, candidate_value
        probabilities = softmax(scores, 1.0 / alpha)
        if step_norm <= 1.0e-11 or abs(gradient) <= 1.0e-11:
            break
    else:
        raise RuntimeError("multiclass temperature oracle did not converge")
    return 1.0 / alpha


def row(details: dict[str, str], phase: str, backend: str, status: str,
        **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {field: "" for field in FIELDS}
    result.update(details)
    result.update({
        "workload": "multiclass_probability_calibration",
        "method": "temperature", "phase": phase, "backend": backend,
        "device": "cpu", "status": status, "n_samples": N_SAMPLES,
        "n_classes": N_CLASSES, "oracle": "", "notes": "",
    })
    result.update(values)
    return result


def checked(scores: np.ndarray, labels: np.ndarray, classes: np.ndarray,
            temperature: float, probabilities: np.ndarray,
            predictions: np.ndarray) -> dict[str, float]:
    expected = softmax(scores, temperature)
    error = float(np.max(np.abs(probabilities - expected)))
    normalization_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1.0)))
    if error > 4.0e-8 or normalization_error > 4.0e-14:
        raise RuntimeError(f"multiclass calibration oracle mismatch {error:.3e}")
    expected_predictions = classes[np.argmax(expected, axis=1)]
    if not np.array_equal(predictions, expected_predictions):
        raise RuntimeError("multiclass calibration predictions disagree with probabilities")
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "probability_normalization_error": normalization_error,
        "temperature": temperature,
        "max_abs_error": error,
    }


def run_numpy(details: dict[str, str]) -> list[dict[str, Any]]:
    scores, labels, classes = fixture()
    temperature = fit_temperature(scores, labels, classes)
    probabilities = softmax(scores, temperature)
    predictions = classes[np.argmax(probabilities, axis=1)]
    metrics = checked(scores, labels, classes, temperature, probabilities, predictions)
    started = time.perf_counter()
    for _ in range(FIT_REPETITIONS):
        fit_temperature(scores, labels, classes)
    fit_seconds = (time.perf_counter() - started) / FIT_REPETITIONS
    started = time.perf_counter()
    for _ in range(PREDICT_REPETITIONS):
        softmax(scores, temperature)
    predict_seconds = (time.perf_counter() - started) / PREDICT_REPETITIONS
    return [
        row(details, "fit", "numpy_oracle", "pass", repetitions=FIT_REPETITIONS,
            seconds_per_operation=fit_seconds, **metrics,
            oracle="independent NumPy weighted softmax-NLL Newton oracle",
            notes="positive inverse-temperature solve with L2=0.05"),
        row(details, "predict", "numpy_oracle", "pass", repetitions=PREDICT_REPETITIONS,
            seconds_per_operation=predict_seconds, **metrics,
            oracle="independent NumPy stable softmax oracle",
            notes="sorted class columns; no FortML calls"),
    ]


def read_fortran(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    labels = np.full(N_SAMPLES, -2, dtype=np.int64)
    predictions = np.full(N_SAMPLES, -2, dtype=np.int64)
    probabilities = np.full((N_SAMPLES, N_CLASSES), np.nan, dtype=np.float64)
    classes = np.full(N_CLASSES, -2, dtype=np.int64)
    temperature = np.nan
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            quantity = record["quantity"]
            row_index = int(record["row"]) - 1
            column = int(record["column"]) - 1
            value = float(record["value"])
            if quantity == "class":
                classes[row_index] = int(value)
            elif quantity == "label":
                labels[row_index] = int(value)
            elif quantity == "prediction":
                predictions[row_index] = int(value)
            elif quantity == "probability":
                probabilities[row_index, column] = value
            elif quantity == "temperature":
                temperature = value
            else:
                raise RuntimeError(f"unknown FortML quantity {quantity!r}")
    if (np.any(labels == -2) or np.any(predictions == -2) or
            np.any(classes == -2) or np.isnan(probabilities).any() or
            not np.isfinite(temperature)):
        raise RuntimeError("FortML omitted multiclass calibration outputs")
    return labels, predictions, probabilities, classes, temperature


def run_fortml(fortml: Path, details: dict[str, str]) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                           env=environment, capture_output=True, text=True)
    if build.returncode != 0:
        return [row(details, "fit", "fortml", "unavailable",
                    oracle="FortML release-app protocol", notes="fo build failed")]
    scores, labels, classes = fixture()
    expected_temperature = fit_temperature(scores, labels, classes)
    with tempfile.TemporaryDirectory(dir="/mnt/storage", prefix="fortml-multical-") as directory:
        oracle_path = Path(directory) / "multiclass_calibration.csv"
        check_environment = dict(environment)
        check_environment["FORTML_BENCH_MULTICLASS_CALIBRATION_ORACLE"] = str(oracle_path)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", "fortml_bench_multiclass_probability_calibration"],
            cwd=fortml, env=check_environment, capture_output=True, text=True,
        )
        if completed.returncode != 0 or not oracle_path.is_file():
            return [row(details, "fit", "fortml", "unavailable",
                        oracle="FortML release-app protocol",
                        notes="release app did not emit complete output")]
        actual_labels, actual_predictions, actual_probabilities, actual_classes, actual_temperature = read_fortran(oracle_path)
    if not np.array_equal(actual_labels, labels) or not np.array_equal(actual_classes, classes):
        raise RuntimeError("FortML multiclass calibration fixture or class order differs")
    metrics = checked(scores, labels, classes, expected_temperature,
                      actual_probabilities, actual_predictions)
    temperature_error = abs(actual_temperature - expected_temperature)
    error = max(metrics["max_abs_error"], temperature_error)
    if error > 4.0e-8:
        raise RuntimeError(f"FortML multiclass calibration mismatch {error:.3e}")
    metrics["temperature"] = actual_temperature
    metrics["max_abs_error"] = error
    timings: dict[str, float] = {}
    pattern = re.compile(r"^multiclass_probability_calibration_(fit|predict),temperature,\s*(.+)$")
    for line in completed.stdout.splitlines():
        match = pattern.match(line.strip())
        if match:
            timings[match.group(1)] = float(match.group(2))
    return [
        row(details, "fit", "fortml", "pass" if "fit" in timings else "parse_failed",
            repetitions=FIT_REPETITIONS, seconds_per_operation=timings.get("fit", float("nan")),
            **metrics,
            oracle="independent NumPy weighted softmax-NLL Newton oracle",
            notes="complete class/label/probability release array checked before timing"),
        row(details, "predict", "fortml", "pass" if "predict" in timings else "parse_failed",
            repetitions=PREDICT_REPETITIONS, seconds_per_operation=timings.get("predict", float("nan")),
            **metrics,
            oracle="independent NumPy stable softmax oracle",
            notes="complete probability/prediction release array checked before timing"),
    ]


def device_refusal(details: dict[str, str]) -> list[dict[str, Any]]:
    return [row(details, "device_capability", "fortml", "unavailable", device="cuda",
                oracle="FortML CUDA capability boundary; no resident calibration kernel",
                notes="device_supported(CUDA)=false; no host fallback")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/multiclass_probability_calibration.csv"))
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    details = metadata(root, arguments.fortml.resolve(), arguments.output)
    rows = run_numpy(details) + run_fortml(arguments.fortml.resolve(), details) + device_refusal(details)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: item.get(field, "") for field in FIELDS} for item in rows)


if __name__ == "__main__":
    main()
