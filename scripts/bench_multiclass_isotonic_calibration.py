#!/usr/bin/env python3
"""Correctness-gated weighted multiclass isotonic calibration benchmark."""

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
FIT_REPETITIONS = 8
PREDICT_REPETITIONS = 128
NOT_IMPLEMENTED = 3
FIELDS = (
    "workload", "method", "phase", "backend", "device", "status",
    "n_samples", "n_classes", "repetitions", "seconds_per_operation",
    "accuracy", "probability_normalization_error", "knot_count",
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


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    index = np.arange(1, N_SAMPLES + 1, dtype=np.float64)
    first = 1.2 * np.sin(0.031 * index) + 0.17 * np.cos(0.013 * index)
    second = 1.1 * np.cos(0.027 * index + 0.4) + 0.12 * np.sin(0.017 * index)
    third = 0.9 * np.sin(0.019 * index + 1.1) - 0.18 * np.cos(0.011 * index)
    scores = np.column_stack((first, second, third))
    classes = np.array([-4, 17, 91], dtype=np.int64)
    labels = classes[np.argmax(scores, axis=1)]
    weights = 0.7 + 0.05 * (np.arange(1, N_SAMPLES + 1) % 9)
    return scores, labels, weights, classes


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=1, keepdims=True)


def pava_column(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray,
                class_label: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores, kind="stable")
    unique_x: list[float] = []
    unique_w: list[float] = []
    unique_y: list[float] = []
    for index in order:
        if weights[index] <= 0.0:
            continue
        score = float(scores[index])
        if not unique_x or score != unique_x[-1]:
            unique_x.append(score)
            unique_w.append(float(weights[index]))
            unique_y.append(float(weights[index]) * (labels[index] == class_label))
        else:
            unique_w[-1] += float(weights[index])
            unique_y[-1] += float(weights[index]) * (labels[index] == class_label)
    blocks: list[list[float]] = []
    for x_value, weight, target_mass in zip(unique_x, unique_w, unique_y):
        blocks.append([x_value, weight, target_mass])
        while len(blocks) >= 2:
            previous, current = blocks[-2], blocks[-1]
            if previous[2] / previous[1] <= current[2] / current[1]:
                break
            merged_weight = previous[1] + current[1]
            previous[0] = (previous[1] * previous[0] + current[1] * current[0]) / merged_weight
            previous[1] = merged_weight
            previous[2] += current[2]
            blocks.pop()
    knots = np.array([block[0] for block in blocks], dtype=np.float64)
    values = np.array([block[2] / block[1] for block in blocks], dtype=np.float64)
    return knots, values


def isotonic_oracle(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray,
                    classes: np.ndarray) -> tuple[np.ndarray, int]:
    raw = softmax(scores)
    probabilities = np.empty_like(raw)
    knot_count = 0
    for column, class_label in enumerate(classes):
        knots, values = pava_column(raw[:, column], labels, weights, int(class_label))
        knot_count += knots.size
        probabilities[:, column] = np.interp(
            raw[:, column], knots, values, left=values[0], right=values[-1],
        )
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return probabilities, knot_count


def row(details: dict[str, str], phase: str, backend: str, status: str,
        **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {field: "" for field in FIELDS}
    result.update(details)
    result.update({
        "workload": "multiclass_isotonic_calibration", "method": "isotonic",
        "phase": phase, "backend": backend, "device": "cpu", "status": status,
        "n_samples": N_SAMPLES, "n_classes": N_CLASSES, "oracle": "", "notes": "",
    })
    result.update(values)
    return result


def checked(scores: np.ndarray, labels: np.ndarray, classes: np.ndarray,
            probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    expected, knot_count = isotonic_oracle(scores, labels, fixture()[2], classes)
    error = float(np.max(np.abs(probabilities - expected)))
    normalization_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1.0)))
    if error > 4.0e-12 or normalization_error > 4.0e-14:
        raise RuntimeError(f"multiclass isotonic oracle mismatch {error:.3e}")
    expected_predictions = classes[np.argmax(expected, axis=1)]
    if not np.array_equal(predictions, expected_predictions):
        raise RuntimeError("FortML isotonic predictions disagree with probabilities")
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "probability_normalization_error": normalization_error,
        "knot_count": knot_count,
        "max_abs_error": error,
    }


def run_numpy(details: dict[str, str]) -> list[dict[str, Any]]:
    scores, labels, weights, classes = fixture()
    probabilities, knot_count = isotonic_oracle(scores, labels, weights, classes)
    predictions = classes[np.argmax(probabilities, axis=1)]
    metrics = checked(scores, labels, classes, probabilities, predictions)
    started = time.perf_counter()
    for _ in range(FIT_REPETITIONS):
        isotonic_oracle(scores, labels, weights, classes)
    fit_seconds = (time.perf_counter() - started) / FIT_REPETITIONS
    started = time.perf_counter()
    for _ in range(PREDICT_REPETITIONS):
        isotonic_oracle(scores, labels, weights, classes)[0]
    predict_seconds = (time.perf_counter() - started) / PREDICT_REPETITIONS
    return [
        row(details, "fit", "numpy_oracle", "pass", repetitions=FIT_REPETITIONS,
            seconds_per_operation=fit_seconds, **metrics,
            oracle="independent NumPy weighted one-vs-rest PAVA oracle",
            notes=f"weighted simplex normalization; knot_count={knot_count}"),
        row(details, "predict", "numpy_oracle", "pass", repetitions=PREDICT_REPETITIONS,
            seconds_per_operation=predict_seconds, **metrics,
            oracle="independent NumPy stable-softmax/interpolation oracle",
            notes="no FortML calls"),
    ]


def read_fortran(path: Path) -> dict[str, Any]:
    labels = np.full(N_SAMPLES, -2, dtype=np.int64)
    predictions = np.full(N_SAMPLES, -2, dtype=np.int64)
    scores = np.full((N_SAMPLES, N_CLASSES), np.nan)
    probabilities = np.full((N_SAMPLES, N_CLASSES), np.nan)
    weights = np.full(N_SAMPLES, np.nan)
    classes = np.full(N_CLASSES, -2, dtype=np.int64)
    statuses: dict[str, int] = {}
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
            elif quantity == "score":
                scores[row_index, column] = value
            elif quantity == "probability":
                probabilities[row_index, column] = value
            elif quantity == "weight":
                weights[row_index] = value
            elif quantity.endswith("_status"):
                statuses[quantity] = int(value)
            else:
                raise RuntimeError(f"unknown FortML quantity {quantity!r}")
    if (np.any(labels == -2) or np.any(predictions == -2) or np.any(classes == -2) or
            np.isnan(scores).any() or np.isnan(probabilities).any() or np.isnan(weights).any()):
        raise RuntimeError("FortML omitted multiclass isotonic outputs")
    return {"labels": labels, "predictions": predictions, "scores": scores,
            "probabilities": probabilities, "weights": weights, "classes": classes,
            "statuses": statuses}


def run_fortml(fortml: Path, details: dict[str, str]) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                           env=environment, capture_output=True, text=True)
    if build.returncode != 0:
        return [row(details, "fit", "fortml", "unavailable",
                    oracle="FortML release-app protocol", notes="fo build failed")]
    scores, labels, weights, classes = fixture()
    with tempfile.TemporaryDirectory(dir="/mnt/storage", prefix="fortml-isotonic-") as directory:
        oracle_path = Path(directory) / "multiclass_isotonic.csv"
        check_environment = dict(environment)
        check_environment["FORTML_BENCH_MULTICLASS_ISOTONIC_ORACLE"] = str(oracle_path)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", "fortml_bench_multiclass_isotonic_calibration"],
            cwd=fortml, env=check_environment, capture_output=True, text=True,
        )
        if completed.returncode != 0 or not oracle_path.is_file():
            return [row(details, "fit", "fortml", "unavailable",
                        oracle="FortML release-app protocol",
                        notes="release app did not emit complete output")]
        actual = read_fortran(oracle_path)
    if (not np.array_equal(actual["labels"], labels) or
            not np.array_equal(actual["classes"], classes) or
            np.max(np.abs(actual["scores"] - scores)) > 2.0e-14 or
            np.max(np.abs(actual["weights"] - weights)) > 2.0e-14):
        raise RuntimeError("FortML isotonic fixture differs from NumPy fixture")
    metrics = checked(scores, labels, classes, actual["probabilities"], actual["predictions"])
    statuses = actual["statuses"]
    if any(statuses.get(name) != NOT_IMPLEMENTED for name in
           ("jvp_status", "vjp_status", "cuda_status")):
        raise RuntimeError(f"unexpected refusal status codes: {statuses}")
    timings: dict[str, float] = {}
    pattern = re.compile(r"^multiclass_probability_calibration_(fit|predict),isotonic,\s*(.+)$")
    for line in completed.stdout.splitlines():
        match = pattern.match(line.strip())
        if match:
            timings[match.group(1)] = float(match.group(2))
    return [
        row(details, "fit", "fortml", "pass" if "fit" in timings else "parse_failed",
            repetitions=FIT_REPETITIONS, seconds_per_operation=timings.get("fit", float("nan")),
            **metrics, oracle="independent NumPy weighted one-vs-rest PAVA oracle",
            notes="complete fixture/probability release array checked before timing"),
        row(details, "predict", "fortml", "pass" if "predict" in timings else "parse_failed",
            repetitions=PREDICT_REPETITIONS,
            seconds_per_operation=timings.get("predict", float("nan")), **metrics,
            oracle="independent NumPy stable-softmax/interpolation oracle",
            notes="typed JVP/VJP/CUDA refusals also checked"),
        row(details, "derivative_refusal", "fortml", "pass", device="cpu",
            repetitions=1, seconds_per_operation=0.0, **metrics,
            oracle="typed active-set derivative boundary", notes="JVP/VJP status=FORTNUM_NOT_IMPLEMENTED"),
        row(details, "device_capability", "fortml", "unavailable", device="cuda",
            repetitions=1, seconds_per_operation=0.0, **metrics,
            oracle="FortML CUDA capability boundary", notes="no resident isotonic kernel; no host fallback"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/multiclass_isotonic_calibration.csv"))
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    details = metadata(root, arguments.fortml.resolve(), arguments.output)
    rows = run_numpy(details) + run_fortml(arguments.fortml.resolve(), details)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: item.get(field, "") for field in FIELDS} for item in rows)


if __name__ == "__main__":
    main()
