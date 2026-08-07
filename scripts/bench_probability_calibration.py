#!/usr/bin/env python3
"""Correctness-gated sigmoid and isotonic calibration benchmark.

The NumPy implementations below are independent behavioral oracles.  The
FortML release app must emit every calibrated probability and label for both
methods before a timing row is retained.  CUDA is represented by an explicit
capability boundary; no host fallback is timed as GPU execution.
"""

from __future__ import annotations

import argparse
import csv
import math
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
PREDICT_REPETITIONS = 128
FIT_REPETITIONS = 8
L2 = 0.1
METHODS = ("sigmoid", "isotonic")
FIELDS = (
    "workload", "method", "phase", "backend", "device", "status",
    "n_samples", "repetitions", "seconds_per_operation", "accuracy",
    "probability_normalization_error", "max_abs_error", "oracle",
    "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
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
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray]:
    phase = np.arange(1, N_SAMPLES + 1, dtype=np.float64)
    scores = 1.5 * np.sin(0.071 * phase) + 0.35 * np.cos(0.013 * phase)
    scores += 0.002 * phase
    labels = np.where(scores + 0.2 * np.sin(0.17 * phase) > 0.0, 42, 10)
    return scores, labels.astype(np.int64)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def sigmoid_objective(theta: np.ndarray, scores: np.ndarray, target: np.ndarray) -> float:
    eta = theta[0] * scores + theta[1]
    value = np.where(
        eta >= 0.0,
        np.log1p(np.exp(-eta)) + (1.0 - target) * eta,
        np.log1p(np.exp(eta)) - target * eta,
    ).sum()
    return float(value + 0.5 * L2 * np.dot(theta, theta))


def sigmoid_oracle(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    target = (labels == 42).astype(np.float64)
    theta = np.array([0.0, math.log(target.mean() / (1.0 - target.mean()))])
    objective = sigmoid_objective(theta, scores, target)
    for _ in range(500):
        probability = sigmoid(theta[0] * scores + theta[1])
        residual = probability - target
        curvature = np.maximum(probability * (1.0 - probability), 1.0e-14)
        gradient = np.array([
            np.dot(residual, scores) + L2 * theta[0],
            residual.sum() + L2 * theta[1],
        ])
        hessian = np.array([
            [np.dot(curvature, scores * scores) + L2, np.dot(curvature, scores)],
            [np.dot(curvature, scores), curvature.sum() + L2],
        ])
        direction = np.linalg.solve(hessian, gradient)
        step = 1.0
        candidate = theta - step * direction
        candidate_objective = sigmoid_objective(candidate, scores, target)
        while candidate_objective > objective and step > 1.0e-8:
            step *= 0.5
            candidate = theta - step * direction
            candidate_objective = sigmoid_objective(candidate, scores, target)
        step_norm = np.max(np.abs(candidate - theta)) / max(1.0, np.max(np.abs(theta)))
        theta, objective = candidate, candidate_objective
        if step_norm <= 1.0e-10 or np.max(np.abs(gradient)) <= 1.0e-10:
            break
    positive = sigmoid(theta[0] * scores + theta[1])
    return np.column_stack((1.0 - positive, positive))


def isotonic_oracle(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    target = (labels == 42).astype(np.float64)
    order = np.argsort(scores, kind="mergesort")
    unique_scores: list[float] = []
    unique_weights: list[float] = []
    unique_positive: list[float] = []
    for index in order:
        score = float(scores[index])
        if not unique_scores or score != unique_scores[-1]:
            unique_scores.append(score)
            unique_weights.append(1.0)
            unique_positive.append(float(target[index]))
        else:
            unique_weights[-1] += 1.0
            unique_positive[-1] += float(target[index])
    block_x: list[float] = []
    block_w: list[float] = []
    block_y: list[float] = []
    for score, weight, positive in zip(unique_scores, unique_weights, unique_positive):
        block_x.append(score)
        block_w.append(weight)
        block_y.append(positive)
        while len(block_x) >= 2 and block_y[-2] / block_w[-2] > block_y[-1] / block_w[-1]:
            weight_sum = block_w[-2] + block_w[-1]
            block_x[-2] = (block_w[-2] * block_x[-2] + block_w[-1] * block_x[-1]) / weight_sum
            block_w[-2] = weight_sum
            block_y[-2] += block_y[-1]
            del block_x[-1], block_w[-1], block_y[-1]
    knots = np.asarray(block_x)
    values = np.asarray(block_y) / np.asarray(block_w)
    positive = np.interp(scores, knots, values, left=values[0], right=values[-1])
    return np.column_stack((1.0 - positive, positive))


def labels_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.where(probabilities[:, 1] > probabilities[:, 0], 42, 10).astype(np.int64)


def checked_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    if probabilities.shape != (N_SAMPLES, 2) or not np.isfinite(probabilities).all():
        raise RuntimeError("calibration probabilities have invalid shape or values")
    normalization_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1.0)))
    if normalization_error > 2.0e-14:
        raise RuntimeError(f"probability normalization error {normalization_error:.3e}")
    predicted = labels_from_probabilities(probabilities)
    return {
        "accuracy": float(np.mean(predicted == labels)),
        "probability_normalization_error": normalization_error,
    }


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def row(details: dict[str, str], method: str, phase: str, backend: str,
        status: str, **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {field: "" for field in FIELDS}
    result.update(details)
    result.update({
        "workload": "probability_calibration", "method": method,
        "phase": phase, "backend": backend, "device": "cpu", "status": status,
        "n_samples": N_SAMPLES, "repetitions": "", "oracle": "",
        "notes": "",
    })
    result.update(values)
    return result


def parse_timing(stdout: str) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    pattern = re.compile(r"^probability_calibration_(fit|predict),([^,]+),\s*(.+)$")
    for line in stdout.splitlines():
        match = pattern.match(line.strip())
        if match:
            values[(match.group(1), match.group(2).strip())] = float(match.group(3))
    return values


def read_fortran_oracle(path: Path) -> dict[tuple[str, str, int, int], float]:
    values: dict[tuple[str, str, int, int], float] = {}
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            key = (record["method"].strip(), record["quantity"].strip(),
                   int(record["row"]), int(record["column"]))
            if key in values:
                raise RuntimeError(f"duplicate FortML calibration oracle key {key}")
            values[key] = float(record["value"])
    expected = 2 * N_SAMPLES * 4
    if len(values) != expected:
        raise RuntimeError(f"FortML calibration oracle has {len(values)} rows, expected {expected}")
    return values


def run_fortml(fortml: Path, details: dict[str, str]) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment, check=True)
    scores, labels = fixture()
    expected = {"sigmoid": sigmoid_oracle(scores, labels), "isotonic": isotonic_oracle(scores, labels)}
    with tempfile.TemporaryDirectory(dir="/mnt/storage/code/lazy-fortran/fortml/build") as directory:
        oracle_path = Path(directory) / "calibration_oracle.csv"
        environment["FORTML_BENCH_CALIBRATION_ORACLE"] = str(oracle_path)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", "fortml_bench_probability_calibration"],
            cwd=fortml, env=environment, capture_output=True, text=True, check=True,
        )
        actual = read_fortran_oracle(oracle_path)
    rows: list[dict[str, Any]] = []
    timings = parse_timing(completed.stdout)
    for method in METHODS:
        actual_probabilities = np.empty((N_SAMPLES, 2), dtype=np.float64)
        actual_labels = np.empty(N_SAMPLES, dtype=np.int64)
        actual_predictions = np.empty(N_SAMPLES, dtype=np.int64)
        for index in range(1, N_SAMPLES + 1):
            actual_labels[index - 1] = int(actual[(method, "label", index, 1)])
            actual_predictions[index - 1] = int(actual[(method, "prediction", index, 1)])
            for column in (1, 2):
                actual_probabilities[index - 1, column - 1] = actual[(method, "probability", index, column)]
        if not np.array_equal(actual_labels, labels):
            raise RuntimeError(f"FortML {method} fixture labels differ from NumPy")
        if not np.array_equal(actual_predictions, labels_from_probabilities(actual_probabilities)):
            raise RuntimeError("FortML calibration prediction/probability row mismatch")
        error = float(np.max(np.abs(actual_probabilities - expected[method])))
        if error > 3.0e-7:
            raise RuntimeError(f"FortML {method} calibration oracle mismatch {error:.3e}")
        metrics = checked_metrics(labels, actual_probabilities)
        for phase in ("fit", "predict"):
            rows.append(row(
                details, method, phase, "fortml", "pass" if (phase, method) in timings else "parse_failed",
                device="cpu", repetitions=1 if phase == "fit" else PREDICT_REPETITIONS,
                seconds_per_operation=timings.get((phase, method), float("nan")),
                **metrics, max_abs_error=error,
                oracle="independent NumPy sigmoid/PAVA calibration oracle",
                notes="complete-array release app; no GPU fallback",
            ))
    return rows


def run_numpy(details: dict[str, str]) -> list[dict[str, Any]]:
    scores, labels = fixture()
    rows: list[dict[str, Any]] = []
    for method, oracle in (("sigmoid", sigmoid_oracle), ("isotonic", isotonic_oracle)):
        probabilities = oracle(scores, labels)
        metrics = checked_metrics(labels, probabilities)
        started = time.perf_counter()
        for _ in range(FIT_REPETITIONS):
            oracle(scores, labels)
        fit_seconds = (time.perf_counter() - started) / FIT_REPETITIONS
        started = time.perf_counter()
        for _ in range(PREDICT_REPETITIONS):
            oracle(scores, labels)
        predict_seconds = (time.perf_counter() - started) / PREDICT_REPETITIONS
        for phase, seconds in (("fit", fit_seconds), ("predict", predict_seconds)):
            rows.append(row(
                details, method, phase, "numpy_oracle", "pass",
                repetitions=FIT_REPETITIONS if phase == "fit" else PREDICT_REPETITIONS,
                seconds_per_operation=seconds, **metrics, max_abs_error=0.0,
                oracle="independent NumPy sigmoid/PAVA calibration oracle",
                notes="behavioral reference; no FortML calls",
            ))
    return rows


def device_refusal_rows(details: dict[str, str]) -> list[dict[str, Any]]:
    return [row(
        details, method, "device_capability", "fortml", "unavailable", device="cuda",
        oracle="FortML CUDA capability boundary; no resident calibration kernel",
        notes="device_supported(CUDA)=false; host fallback is forbidden",
    ) for method in METHODS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/probability_calibration.csv"))
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
