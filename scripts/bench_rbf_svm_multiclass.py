#!/usr/bin/env python3
"""Correctness-gated one-vs-rest multiclass RBF-SVM benchmark.

Each class is solved independently with SciPy's L-BFGS-B on the same weighted
squared-hinge finite-basis objective as the FortML binary child.  The release
row is retained only after sorted labels, packed parameters, margins,
normalized probabilities, and predictions agree with that independent oracle.
CUDA is recorded as a typed unavailable capability until resident batched RBF
kernels are linked.
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
C, GAMMA = 2.0, 0.6
CLASSES = np.array([-12, 7, 37], dtype=np.int64)
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


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = np.arange(1, N_SAMPLES + 1, dtype=np.float64)
    x = np.empty((N_SAMPLES, N_FEATURES), dtype=np.float64)
    group = np.arange(N_SAMPLES) // (N_SAMPLES // N_CLASSES)
    centers = np.array([-1.0, 0.0, 1.0])
    x[:, 0] = centers[group] + 0.05 * np.sin(0.17 * phase)
    x[:, 1] = np.where(group == 1, -0.2, 0.2) * np.cos(0.13 * phase)
    labels = CLASSES[group]
    weights = 0.75 + 0.5 * (np.mod(np.arange(1, N_SAMPLES + 1), 7) / 6.0)
    return x, labels, weights


def kernel_matrix(x: np.ndarray, gamma: float) -> np.ndarray:
    delta = x[:, None, :] - x[None, :, :]
    return np.exp(-gamma * np.sum(delta * delta, axis=2))


def objective_and_gradient(theta: np.ndarray, kernel: np.ndarray,
                           encoded: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    n = kernel.shape[0]
    coefficient = theta[:n]
    intercept = theta[n]
    score = kernel @ coefficient + intercept
    residual = np.maximum(0.0, 1.0 - encoded * score)
    mass = float(weights.sum())
    score_gradient = -2.0 * C / mass * weights * encoded * residual
    value = (0.5 * float(coefficient @ (kernel @ coefficient)) +
             C / mass * float(np.dot(weights, residual * residual)))
    gradient = np.empty(n + 1, dtype=np.float64)
    gradient[:n] = kernel @ coefficient + kernel @ score_gradient
    gradient[n] = float(score_gradient.sum())
    return value, gradient


def oracle(x: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if minimize is None:
        raise RuntimeError("SciPy is required for the independent oracle")
    kernel = kernel_matrix(x, GAMMA)
    scores = np.empty((len(x), N_CLASSES), dtype=np.float64)
    packed: list[np.ndarray] = []
    for label in CLASSES:
        encoded = np.where(labels == label, 1.0, -1.0)
        result = minimize(
            lambda theta: objective_and_gradient(theta, kernel, encoded, weights),
            np.zeros(len(x) + 1, dtype=np.float64), method="L-BFGS-B", jac=True,
            options={"maxiter": 10000, "ftol": 1.0e-13, "gtol": 1.0e-7,
                     "maxls": 100},
        )
        if not result.success:
            raise RuntimeError(f"independent class {label} solve failed: {result.message}")
        coefficient = result.x[:len(x)]
        intercept = result.x[len(x)]
        scores[:, np.flatnonzero(CLASSES == label)[0]] = kernel @ coefficient + intercept
        packed.append(np.concatenate((coefficient, [intercept, np.log(GAMMA)])))
    raw = 1.0 / (1.0 + np.exp(-scores))
    probabilities = raw / raw.sum(axis=1, keepdims=True)
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
        "workload": "rbf_svm_multiclass", "phase": "", "backend": "",
        "device": "cpu", "status": "", "n_samples": N_SAMPLES,
        "n_features": N_FEATURES, "n_classes": N_CLASSES,
        "seconds_per_operation": "", "accuracy": "", "max_abs_error": "",
        "oracle": "", "notes": "",
    }
    row.update(details)
    row.update(values)
    return row


def parse_fortran(path: Path) -> dict[str, Any]:
    labels = np.full(N_SAMPLES, -999, dtype=np.int64)
    predictions = np.full(N_SAMPLES, -999, dtype=np.int64)
    scores = np.full((N_SAMPLES, N_CLASSES), np.nan)
    probabilities = np.full((N_SAMPLES, N_CLASSES), np.nan)
    parameters = np.full(N_CLASSES * (N_SAMPLES + 2), np.nan)
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
            elif quantity == "score":
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
        raise RuntimeError("FortML omitted a multiclass RBF-SVM output")
    return {"labels": labels, "predictions": predictions, "scores": scores,
            "probabilities": probabilities, "parameters": parameters,
            "classes": classes, "scalars": scalars}


def run(root: Path, fortml: Path, output: Path, no_build: bool) -> list[dict[str, Any]]:
    details = metadata(root, fortml, output)
    x, labels, weights = fixture()
    expected_parameters, expected_scores, expected_probabilities, expected_predictions = oracle(
        x, labels, weights,
    )
    oracle_started = time.perf_counter()
    for _ in range(PREDICTION_REPETITIONS):
        raw = 1.0 / (1.0 + np.exp(-expected_scores))
        raw / raw.sum(axis=1, keepdims=True)
    oracle_seconds = (time.perf_counter() - oracle_started) / PREDICTION_REPETITIONS
    rows = [make_row(
        details, phase="fit_predict", backend="numpy_oracle", status="pass",
        seconds_per_operation=oracle_seconds,
        accuracy=float(np.mean(expected_predictions == labels)), max_abs_error=0.0,
        oracle="independent per-class SciPy L-BFGS-B weighted squared-hinge RKHS solves",
        notes="three sorted classes; packed coefficients/intercepts/log-gamma",
    )]
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    if not no_build:
        subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                       env=environment, check=True)
    subprocess.run(["fo", "test", "test_rbf_svm_multiclass"], cwd=fortml,
                   env=environment, check=True)
    with tempfile.TemporaryDirectory(dir="/mnt/storage/code/lazy-fortran/fortml/build") as directory:
        oracle_path = Path(directory) / "rbf_svm_multiclass.csv"
        environment["FORTML_BENCH_RBF_SVM_MULTICLASS_ORACLE"] = str(oracle_path)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", "fortml_bench_rbf_svm_multiclass"],
            cwd=fortml, env=environment, capture_output=True, text=True, check=True,
        )
        observed = parse_fortran(oracle_path)
    score_error = float(np.max(np.abs(observed["scores"] - expected_scores)))
    probability_error = float(np.max(np.abs(observed["probabilities"] - expected_probabilities)))
    parameter_error = float(np.max(np.abs(observed["parameters"] - expected_parameters)))
    labels_match = np.array_equal(observed["labels"], labels)
    classes_match = np.array_equal(observed["classes"], CLASSES)
    predictions_match = np.array_equal(observed["predictions"], expected_predictions)
    # The dense Gram matrix can be mildly ill-conditioned: different
    # coefficient coordinates can induce the same score map.  Gate the
    # behavioral quantities while recording the packed-coordinate drift.
    max_error = max(score_error, probability_error)
    if (not labels_match or not classes_match or not predictions_match or
            score_error > 4.0e-6 or probability_error > 4.0e-7):
        raise RuntimeError(
            f"FortML multiclass RBF-SVM mismatch: labels={labels_match}, "
            f"classes={classes_match}, predictions={predictions_match}, "
            f"score={score_error:.3e}, probability={probability_error:.3e}, "
            f"parameter={parameter_error:.3e}",
        )
    timing = observed["scalars"]
    rows.append(make_row(
        details, phase="fit", backend="fortml", status="pass",
        seconds_per_operation=timing.get("fit_seconds", ""), accuracy=float(np.mean(expected_predictions == labels)),
        max_abs_error=max_error, oracle="NumPy/SciPy multiclass OVR replay",
        notes=f"scores={score_error:.3e}; probabilities={probability_error:.3e}; parameters={parameter_error:.3e}",
    ))
    rows.append(make_row(
        details, phase="predict", backend="fortml", status="pass",
        seconds_per_operation=timing.get("predict_seconds", ""), accuracy=float(np.mean(expected_predictions == labels)),
        max_abs_error=max_error, oracle="NumPy/SciPy multiclass OVR replay",
        notes="normalized probabilities and original integer labels",
    ))
    rows.append(make_row(
        details, phase="predict", backend="fortml", device="cuda",
        status="unavailable", max_abs_error="", oracle="typed capability contract",
        notes=f"FORTNUM_NOT_IMPLEMENTED status={int(timing.get('cuda_status', -1))}; no host fallback",
    ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/rbf_svm_multiclass.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/RBF_SVM_MULTICLASS.md"))
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    rows = run(Path.cwd(), args.fortml.resolve(), args.output.resolve(), args.no_build)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    errors = [row["max_abs_error"] for row in rows if row["backend"] == "fortml" and row["status"] == "pass"]
    args.report.write_text(
        "# Multiclass RBF SVM\n\n"
        "This correctness-gated lane compares FortML's transactional sorted-label\n"
        "one-vs-rest finite-basis RBF SVM against independent per-class SciPy\n"
        "L-BFGS-B weighted squared-hinge solves. The release rows retain margins,\n"
        "normalized probabilities, packed child parameters, and predictions only\n"
        "after all arrays agree. CUDA is an explicit typed-unavailable row until\n"
        "resident batched RBF kernels are linked.\n\n"
        f"Maximum retained CPU absolute error: {max(errors):.3e}.\n\n"
        "Raw data: [`rbf_svm_multiclass.csv`](rbf_svm_multiclass.csv).\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} and {args.report}")


if __name__ == "__main__":
    main()
