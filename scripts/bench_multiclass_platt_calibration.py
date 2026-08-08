#!/usr/bin/env python3
"""Correctness-gated weighted multiclass one-vs-rest Platt benchmark."""

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
NOT_IMPLEMENTED = 3
FIELDS = (
    "workload", "method", "phase", "backend", "device", "status",
    "n_samples", "n_classes", "repetitions", "seconds_per_operation",
    "accuracy", "probability_normalization_error", "parameter_count",
    "max_abs_error", "max_parameter_error", "oracle", "python_version",
    "numpy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
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


def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return np.where(np.asarray(value) >= 0.0, 1.0 / (1.0 + np.exp(-np.asarray(value))),
                    np.exp(np.asarray(value)) / (1.0 + np.exp(np.asarray(value))))


def fit_column(scores: np.ndarray, encoded: np.ndarray, weights: np.ndarray,
               class_index: int) -> tuple[float, float, float]:
    total = float(np.sum(weights))
    positive = float(np.sum(weights[encoded == class_index]))
    mean = np.clip(positive / total, 1.0e-8, 1.0 - 1.0e-8)
    slope = 0.0
    intercept = float(np.log(mean / (1.0 - mean)))

    def objective(value_slope: float, value_intercept: float) -> float:
        eta = value_slope * scores + value_intercept
        target = (encoded == class_index).astype(np.float64)
        value = np.where(eta >= 0.0, np.log1p(np.exp(-eta)) + (1.0 - target) * eta,
                         np.log1p(np.exp(eta)) - target * eta)
        return float(np.sum(weights * value) + 0.5 * L2 *
                     (value_slope * value_slope + value_intercept * value_intercept))

    current = objective(slope, intercept)
    converged = False
    for _ in range(500):
        eta = slope * scores + intercept
        probability = np.asarray(sigmoid(eta))
        target = (encoded == class_index).astype(np.float64)
        residual = probability - target
        curvature = np.maximum(probability * (1.0 - probability), 1.0e-14)
        gradient_slope = float(L2 * slope + np.sum(weights * residual * scores))
        gradient_intercept = float(L2 * intercept + np.sum(weights * residual))
        hessian_ss = float(L2 + np.sum(weights * curvature * scores * scores))
        hessian_si = float(np.sum(weights * curvature * scores))
        hessian_ii = float(L2 + np.sum(weights * curvature))
        determinant = hessian_ss * hessian_ii - hessian_si * hessian_si
        if determinant <= np.finfo(np.float64).tiny or not np.isfinite(determinant):
            raise RuntimeError("Platt oracle has singular Newton system")
        step_slope = (hessian_ii * gradient_slope - hessian_si * gradient_intercept) / determinant
        step_intercept = (-hessian_si * gradient_slope + hessian_ss * gradient_intercept) / determinant
        scale = 1.0
        trial_slope = slope - scale * step_slope
        trial_intercept = intercept - scale * step_intercept
        candidate = objective(trial_slope, trial_intercept)
        while candidate > current and scale > 1.0e-8:
            scale *= 0.5
            trial_slope = slope - scale * step_slope
            trial_intercept = intercept - scale * step_intercept
            candidate = objective(trial_slope, trial_intercept)
        step_norm = max(abs(trial_slope - slope), abs(trial_intercept - intercept)) / max(
            1.0, abs(slope), abs(intercept))
        slope, intercept, current = trial_slope, trial_intercept, candidate
        if step_norm <= 1.0e-11 or max(abs(gradient_slope), abs(gradient_intercept)) <= 1.0e-11:
            converged = True
            break
    if not converged:
        raise RuntimeError("Platt oracle did not converge")
    return slope, intercept, current


def platt_oracle(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray,
                 classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = softmax(scores)
    encoded = np.searchsorted(classes, labels)
    slopes = np.empty(classes.size)
    intercepts = np.empty(classes.size)
    for index in range(classes.size):
        slopes[index], intercepts[index], _ = fit_column(raw[:, index], encoded, weights, index)
    calibrated = np.asarray(sigmoid(raw * slopes[None, :] + intercepts[None, :]))
    probabilities = calibrated / np.sum(calibrated, axis=1, keepdims=True)
    return probabilities, np.column_stack((slopes, intercepts)).reshape(-1)


def row(details: dict[str, str], phase: str, backend: str, status: str,
        **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {field: "" for field in FIELDS}
    result.update(details)
    result.update({
        "workload": "multiclass_platt_calibration", "method": "sigmoid",
        "phase": phase, "backend": backend, "device": "cpu", "status": status,
        "n_samples": N_SAMPLES, "n_classes": N_CLASSES, "oracle": "", "notes": "",
    })
    result.update(values)
    return result


def checked(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray,
            classes: np.ndarray, probabilities: np.ndarray,
            parameters: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    expected, expected_parameters = platt_oracle(scores, labels, weights, classes)
    error = float(np.max(np.abs(probabilities - expected)))
    parameter_error = float(np.max(np.abs(parameters - expected_parameters)))
    normalization_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1.0)))
    if error > 5.0e-9 or parameter_error > 5.0e-9 or normalization_error > 4.0e-14:
        raise RuntimeError(
            f"multiclass Platt oracle mismatch p={error:.3e} theta={parameter_error:.3e}"
        )
    expected_predictions = classes[np.argmax(expected, axis=1)]
    if not np.array_equal(predictions, expected_predictions):
        raise RuntimeError("FortML Platt predictions disagree with probabilities")
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "probability_normalization_error": normalization_error,
        "parameter_count": float(parameters.size),
        "max_abs_error": error,
        "max_parameter_error": parameter_error,
    }


def run_numpy(details: dict[str, str]) -> list[dict[str, Any]]:
    scores, labels, weights, classes = fixture()
    probabilities, parameters = platt_oracle(scores, labels, weights, classes)
    predictions = classes[np.argmax(probabilities, axis=1)]
    metrics = checked(scores, labels, weights, classes, probabilities, parameters, predictions)
    started = time.perf_counter()
    for _ in range(FIT_REPETITIONS):
        platt_oracle(scores, labels, weights, classes)
    fit_seconds = (time.perf_counter() - started) / FIT_REPETITIONS
    started = time.perf_counter()
    for _ in range(PREDICT_REPETITIONS):
        raw = softmax(scores)
        np.asarray(sigmoid(raw * parameters[::2][None, :] + parameters[1::2][None, :]))
    predict_seconds = (time.perf_counter() - started) / PREDICT_REPETITIONS
    return [
        row(details, "fit", "numpy_oracle", "pass", repetitions=FIT_REPETITIONS,
            seconds_per_operation=fit_seconds, **metrics,
            oracle="independent NumPy weighted one-vs-rest sigmoid Newton oracle",
            notes="stable softmax inputs; interleaved slope/intercept parameters"),
        row(details, "predict", "numpy_oracle", "pass", repetitions=PREDICT_REPETITIONS,
            seconds_per_operation=predict_seconds, **metrics,
            oracle="independent NumPy sigmoid/simplex oracle", notes="no FortML calls"),
    ]


def read_fortran(path: Path) -> dict[str, Any]:
    labels = np.full(N_SAMPLES, -2, dtype=np.int64)
    predictions = np.full(N_SAMPLES, -2, dtype=np.int64)
    scores = np.full((N_SAMPLES, N_CLASSES), np.nan)
    probabilities = np.full((N_SAMPLES, N_CLASSES), np.nan)
    weights = np.full(N_SAMPLES, np.nan)
    classes = np.full(N_CLASSES, -2, dtype=np.int64)
    parameters = np.full(2 * N_CLASSES, np.nan)
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
            elif quantity == "slope":
                parameters[2 * row_index] = value
            elif quantity == "intercept":
                parameters[2 * row_index + 1] = value
            elif quantity.endswith("_status"):
                statuses[quantity] = int(value)
            else:
                raise RuntimeError(f"unknown FortML quantity {quantity!r}")
    if (np.any(labels == -2) or np.any(predictions == -2) or np.any(classes == -2) or
            np.isnan(scores).any() or np.isnan(probabilities).any() or np.isnan(weights).any() or
            np.isnan(parameters).any()):
        raise RuntimeError("FortML omitted multiclass Platt outputs")
    return {"labels": labels, "predictions": predictions, "scores": scores,
            "probabilities": probabilities, "weights": weights, "classes": classes,
            "parameters": parameters, "statuses": statuses}


def run_fortml(fortml: Path, details: dict[str, str]) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                           env=environment, capture_output=True, text=True)
    if build.returncode != 0:
        return [row(details, "fit", "fortml", "unavailable",
                    oracle="FortML release-app protocol", notes="fo build failed")]
    scores, labels, weights, classes = fixture()
    expected_parameters = platt_oracle(scores, labels, weights, classes)[1]
    with tempfile.TemporaryDirectory(dir="/mnt/storage", prefix="fortml-platt-") as directory:
        oracle_path = Path(directory) / "multiclass_platt.csv"
        check_environment = dict(environment)
        check_environment["FORTML_BENCH_MULTICLASS_PLATT_ORACLE"] = str(oracle_path)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", "fortml_bench_multiclass_platt_calibration"],
            cwd=fortml, env=check_environment, capture_output=True, text=True,
        )
        if completed.returncode != 0 or not oracle_path.is_file():
            return [row(details, "fit", "fortml", "unavailable",
                        oracle="FortML release-app protocol",
                        notes="release app did not emit complete output")]
        actual = read_fortran(oracle_path)
    if (not np.array_equal(actual["labels"], labels) or not np.array_equal(actual["classes"], classes) or
            np.max(np.abs(actual["scores"] - scores)) > 2.0e-14 or
            np.max(np.abs(actual["weights"] - weights)) > 2.0e-14):
        raise RuntimeError("FortML Platt fixture differs from NumPy fixture")
    metrics = checked(scores, labels, weights, classes, actual["probabilities"],
                      actual["parameters"], actual["predictions"])
    statuses = actual["statuses"]
    expected_refusals = ("cuda_status",)
    if any(statuses.get(name) != NOT_IMPLEMENTED for name in expected_refusals):
        raise RuntimeError(f"unexpected refusal status codes: {statuses}")
    if any(statuses.get(name) != 0 for name in
           ("jvp_status", "vjp_status", "parameter_jvp_status", "parameter_vjp_status")):
        raise RuntimeError(f"unexpected smooth derivative status codes: {statuses}")
    timings: dict[str, float] = {}
    pattern = re.compile(r"^multiclass_probability_calibration_(fit|predict),sigmoid,\s*(.+)$")
    for line in completed.stdout.splitlines():
        match = pattern.match(line.strip())
        if match:
            timings[match.group(1)] = float(match.group(2))
    return [
        row(details, "fit", "fortml", "pass" if "fit" in timings else "parse_failed",
            repetitions=FIT_REPETITIONS, seconds_per_operation=timings.get("fit", float("nan")),
            **metrics, oracle="independent NumPy weighted one-vs-rest sigmoid Newton oracle",
            notes="complete fixture/parameter/probability release array checked before timing"),
        row(details, "predict", "fortml", "pass" if "predict" in timings else "parse_failed",
            repetitions=PREDICT_REPETITIONS,
            seconds_per_operation=timings.get("predict", float("nan")), **metrics,
            oracle="independent NumPy sigmoid/simplex oracle",
            notes="smooth JVP/VJP and typed CUDA refusal also checked"),
        row(details, "derivative_products", "fortml", "pass", device="cpu", repetitions=1,
            seconds_per_operation=0.0, **metrics,
            oracle="smooth input/parameter product contract", notes="all four JVP/VJP calls returned FORTNUM_OK"),
        row(details, "device_capability", "fortml", "unavailable", device="cuda", repetitions=1,
            seconds_per_operation=0.0, **metrics,
            oracle="FortML CUDA capability boundary", notes="no resident calibration kernel; no host fallback"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/multiclass_platt_calibration.csv"))
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
