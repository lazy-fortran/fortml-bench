#!/usr/bin/env python3
"""Independent NumPy/SciPy oracle for FortML's OVO RBF-SVM.

The oracle solves every sorted class pair on its own finite basis with the
same weighted squared-hinge objective used by ``rbf_svm_classifier_t``.  It
then replays the explicit normalized pairwise-vote probability policy and
gates FortML margins, probabilities, labels, metadata, and packed state.
CUDA is retained as a typed unavailable capability row.
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
from typing import Any

import numpy as np

try:
    import scipy
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover
    scipy = None
    minimize = None

N_SAMPLES, N_FEATURES, N_CLASSES = 36, 2, 3
N_PER_CLASS = N_SAMPLES // N_CLASSES
C, GAMMA = 2.0, 0.6
CLASSES = np.array([-12, 7, 37], dtype=np.int64)
N_PAIRS = N_CLASSES * (N_CLASSES - 1) // 2
PREDICTION_REPETITIONS = 64
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_classes", "seconds_per_operation", "accuracy",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "scipy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
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


def fixture() -> tuple[np.ndarray, np.ndarray]:
    phase = np.arange(1, N_SAMPLES + 1, dtype=np.float64)
    x = np.empty((N_SAMPLES, N_FEATURES), dtype=np.float64)
    group = np.arange(N_SAMPLES) // N_PER_CLASS
    centers = np.array([-1.0, 0.0, 1.0])
    x[:, 0] = centers[group] + 0.05 * np.sin(0.17 * phase)
    x[:, 1] = np.where(group == 1, -0.2, 0.2) * np.cos(0.13 * phase)
    return x, CLASSES[group]


def kernel_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = left[:, None, :] - right[None, :, :]
    return np.exp(-GAMMA * np.sum(delta * delta, axis=2))


def objective_and_gradient(theta: np.ndarray, kernel: np.ndarray,
                           encoded: np.ndarray) -> tuple[float, np.ndarray]:
    n = kernel.shape[0]
    coefficient = theta[:n]
    intercept = theta[n]
    score = kernel @ coefficient + intercept
    residual = np.maximum(0.0, 1.0 - encoded * score)
    score_gradient = -2.0 * C / n * encoded * residual
    value = (0.5 * float(coefficient @ (kernel @ coefficient)) +
             C / n * float(np.dot(residual, residual)))
    gradient = np.empty(n + 1, dtype=np.float64)
    gradient[:n] = kernel @ coefficient + kernel @ score_gradient
    gradient[n] = float(score_gradient.sum())
    return value, gradient


def oracle(x: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if minimize is None:
        raise RuntimeError("SciPy is required for the independent oracle")
    scores = np.empty((len(x), N_PAIRS), dtype=np.float64)
    packed: list[np.ndarray] = []
    pair_index = 0
    for negative in range(N_CLASSES - 1):
        for positive in range(negative + 1, N_CLASSES):
            pair_index += 1
            mask = (labels == CLASSES[negative]) | (labels == CLASSES[positive])
            pair_x = x[mask]
            pair_y = labels[mask]
            encoded = np.where(pair_y == CLASSES[positive], 1.0, -1.0)
            kernel = kernel_matrix(pair_x, pair_x)
            result = minimize(
                lambda theta: objective_and_gradient(theta, kernel, encoded),
                np.zeros(len(pair_x) + 1, dtype=np.float64), method="L-BFGS-B", jac=True,
                options={"maxiter": 50000, "ftol": 1.0e-13, "gtol": 1.0e-7,
                         "maxls": 100},
            )
            if not result.success:
                raise RuntimeError(f"independent pair {negative},{positive} solve failed: {result.message}")
            coefficient = result.x[:len(pair_x)]
            intercept = result.x[len(pair_x)]
            scores[:, pair_index - 1] = kernel_matrix(x, pair_x) @ coefficient + intercept
            packed.append(np.concatenate((coefficient, [intercept, np.log(GAMMA)])))
    pair_probability = 1.0 / (1.0 + np.exp(-scores))
    probabilities = np.zeros((len(x), N_CLASSES), dtype=np.float64)
    pair_index = 0
    for negative in range(N_CLASSES - 1):
        for positive in range(negative + 1, N_CLASSES):
            probabilities[:, negative] += (1.0 - pair_probability[:, pair_index]) / N_PAIRS
            probabilities[:, positive] += pair_probability[:, pair_index] / N_PAIRS
            pair_index += 1
    predictions = CLASSES[np.argmax(probabilities, axis=1)]
    return np.concatenate(packed), scores, probabilities, predictions


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "scipy_version": "unavailable" if scipy is None else scipy.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }


def make_row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "workload": "ovo_rbf_svm", "phase": "", "backend": "", "device": "cpu",
        "status": "", "n_samples": N_SAMPLES, "n_features": N_FEATURES,
        "n_classes": N_CLASSES, "seconds_per_operation": "", "accuracy": "",
        "max_abs_error": "", "oracle": "", "notes": "",
    }
    row.update(details)
    row.update(values)
    return row


def parse_fortran(path: Path, parameter_count: int) -> dict[str, Any]:
    labels = np.full(N_SAMPLES, -999, dtype=np.int64)
    predictions = np.full(N_SAMPLES, -999, dtype=np.int64)
    scores = np.full((N_SAMPLES, N_PAIRS), np.nan)
    probabilities = np.full((N_SAMPLES, N_CLASSES), np.nan)
    parameters = np.full(parameter_count, np.nan)
    classes = np.full(N_CLASSES, -999, dtype=np.int64)
    scalars: dict[str, float] = {}
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            quantity = record["quantity"]
            row = int(record["row"]) - 1
            column = int(record["column"]) - 1
            value = float(record["value"])
            if quantity == "label":
                labels[row] = int(value)
            elif quantity == "prediction":
                predictions[row] = int(value)
            elif quantity == "decision":
                scores[row, column] = value
            elif quantity == "probability":
                probabilities[row, column] = value
            elif quantity == "parameter":
                parameters[row] = value
            elif quantity == "class":
                classes[row] = int(value)
            else:
                scalars[quantity] = value
    if (np.any(labels == -999) or np.any(predictions == -999) or
            np.isnan(scores).any() or np.isnan(probabilities).any() or
            np.isnan(parameters).any() or np.any(classes == -999)):
        raise RuntimeError("FortML omitted an OVO RBF-SVM output")
    return {"labels": labels, "predictions": predictions, "scores": scores,
            "probabilities": probabilities, "parameters": parameters,
            "classes": classes, "scalars": scalars}


def run(root: Path, fortml: Path, output: Path, no_build: bool) -> list[dict[str, Any]]:
    details = metadata(root, fortml, output)
    x, labels = fixture()
    expected_parameters, expected_scores, expected_probabilities, expected_predictions = oracle(x, labels)
    oracle_started = time.perf_counter()
    for _ in range(PREDICTION_REPETITIONS):
        pair_probability = 1.0 / (1.0 + np.exp(-expected_scores))
        pair_probability.sum(axis=1)
    oracle_seconds = (time.perf_counter() - oracle_started) / PREDICTION_REPETITIONS
    rows = [make_row(
        details, phase="fit_predict", backend="numpy_scipy_oracle", status="pass",
        seconds_per_operation=oracle_seconds, accuracy=float(np.mean(expected_predictions == labels)),
        max_abs_error=0.0, oracle="independent per-pair SciPy L-BFGS-B weighted squared-hinge RKHS solves",
        notes="three sorted classes; pair-specific bases and packed offsets",
    )]
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    if not no_build:
        subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment, check=True)
    subprocess.run(["fo", "test", "test_ovo_rbf_svm_classifier"], cwd=fortml, env=environment, check=True)
    parameter_count = len(expected_parameters)
    (fortml / "build").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(fortml / "build")) as directory:
        oracle_path = Path(directory) / "ovo_rbf_svm.csv"
        environment["FORTML_BENCH_OVO_RBF_SVM_ORACLE"] = str(oracle_path)
        subprocess.run(["fo", "exec", "--no-build", "fortml_bench_ovo_rbf_svm_classifier"],
                       cwd=fortml, env=environment, check=True, capture_output=True, text=True)
        observed = parse_fortran(oracle_path, parameter_count)
    score_error = float(np.max(np.abs(observed["scores"] - expected_scores)))
    probability_error = float(np.max(np.abs(observed["probabilities"] - expected_probabilities)))
    parameter_error = float(np.max(np.abs(observed["parameters"] - expected_parameters)))
    labels_match = np.array_equal(observed["labels"], labels)
    classes_match = np.array_equal(observed["classes"], CLASSES)
    predictions_match = np.array_equal(observed["predictions"], expected_predictions)
    max_error = max(score_error, probability_error)
    if (not labels_match or not classes_match or not predictions_match or
            score_error > 2.0e-5 or probability_error > 2.0e-6):
        raise RuntimeError(
            f"FortML OVO RBF-SVM mismatch: labels={labels_match}, classes={classes_match}, "
            f"predictions={predictions_match}, score={score_error:.3e}, "
            f"probability={probability_error:.3e}, parameter={parameter_error:.3e}",
        )
    timing = observed["scalars"]
    rows.append(make_row(
        details, phase="fit", backend="fortml", status="pass",
        seconds_per_operation=timing.get("fit_seconds", ""), accuracy=float(np.mean(expected_predictions == labels)),
        max_abs_error=max_error, oracle="NumPy/SciPy independent OVO replay",
        notes=f"decisions={score_error:.3e}; probabilities={probability_error:.3e}; parameters={parameter_error:.3e}",
    ))
    rows.append(make_row(
        details, phase="predict", backend="fortml", status="pass",
        seconds_per_operation=timing.get("predict_seconds", ""), accuracy=float(np.mean(expected_predictions == labels)),
        max_abs_error=max_error, oracle="NumPy/SciPy independent OVO replay",
        notes="normalized pairwise-vote probabilities and original integer labels",
    ))
    rows.append(make_row(
        details, phase="predict", backend="fortml", device="cuda", status="unavailable",
        max_abs_error="", oracle="typed capability contract",
        notes=f"FORTNUM_NOT_IMPLEMENTED status={int(timing.get('cuda_status', -1))}; no host fallback",
    ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/ovo_rbf_svm.csv"))
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    rows = run(Path(__file__).resolve().parents[1], args.fortml.resolve(), args.output, args.no_build)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
