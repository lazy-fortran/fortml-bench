#!/usr/bin/env python3
"""Correctness-gated benchmark for the weighted softmax FortOpt objective.

The release application emits packed value/gradient/HVP products and a bounded
L-BFGS-B fit record.  NumPy reconstructs the weighted cross-entropy objective
independently, including the optional L2 hyperparameter coordinate, before a
CPU row is retained.  CUDA is a typed unavailable record because this
objective has no resident device implementation yet.
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_classes", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)

X = np.array(
    [[-1.0, -0.8], [0.2, 1.3], [0.7, -0.4], [1.1, 0.2], [-0.3, 0.9]],
    dtype=np.float64,
)
LABELS = np.array([-3, 4, 9, 4, -3], dtype=np.int64)
CLASSES = np.array([-3, 4, 9], dtype=np.int64)
SAMPLE_WEIGHT = np.array([1.0, 2.0, 0.5, 1.2, 0.7], dtype=np.float64)
CLASS_WEIGHT = np.array([1.0, 2.0, 3.0], dtype=np.float64)
PARAMETERS = np.array(
    [0.20, -0.10, -0.30, 0.15, 0.25, -0.20,
     0.10, -0.05, 0.30, 0.35],
    dtype=np.float64,
)
DIRECTION = np.array(
    [-0.13, 0.08, 0.11, -0.09, 0.07, 0.12,
     -0.06, 0.15, -0.10, 0.04],
    dtype=np.float64,
)
L2 = float(PARAMETERS[-1])
N_FEATURES = X.shape[1]
N_CLASSES = CLASSES.size
COEFFICIENT_OFFSET = N_FEATURES * N_CLASSES
TARGET_INDEX = np.searchsorted(CLASSES, LABELS)
WEIGHTS = SAMPLE_WEIGHT * CLASS_WEIGHT[TARGET_INDEX]
WEIGHT_SUM = float(WEIGHTS.sum())


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    """Return HEAD and mark non-output working-tree edits as dirty."""
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def oracle(parameters: np.ndarray, direction: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return independent value, gradient, and HVP for the packed objective."""
    coefficients = parameters[:COEFFICIENT_OFFSET].reshape(
        (N_FEATURES, N_CLASSES), order="F",
    )
    intercept = parameters[COEFFICIENT_OFFSET:COEFFICIENT_OFFSET + N_CLASSES]
    logits = X @ coefficients + intercept
    maximum = logits.max(axis=1, keepdims=True)
    exponential = np.exp(logits - maximum)
    probabilities = exponential / exponential.sum(axis=1, keepdims=True)
    value = float(
        np.sum(WEIGHTS * (maximum[:, 0] - logits[np.arange(X.shape[0]), TARGET_INDEX]
                          + np.log(exponential.sum(axis=1))))
        / WEIGHT_SUM
        + 0.5 * parameters[-1] * np.sum(coefficients * coefficients)
    )
    residual = WEIGHTS[:, None] / WEIGHT_SUM * (
        probabilities - np.eye(N_CLASSES)[TARGET_INDEX]
    )
    gradient = np.empty_like(parameters)
    gradient[:COEFFICIENT_OFFSET] = (X.T @ residual).reshape(-1, order="F")
    gradient[COEFFICIENT_OFFSET:COEFFICIENT_OFFSET + N_CLASSES] = residual.sum(axis=0)
    gradient[:COEFFICIENT_OFFSET] += parameters[-1] * parameters[:COEFFICIENT_OFFSET]
    gradient[-1] = 0.5 * np.sum(coefficients * coefficients)

    dcoefficients = direction[:COEFFICIENT_OFFSET].reshape(
        (N_FEATURES, N_CLASSES), order="F",
    )
    dintercept = direction[COEFFICIENT_OFFSET:COEFFICIENT_OFFSET + N_CLASSES]
    dlogits = X @ dcoefficients + dintercept
    dprobabilities = probabilities * (
        dlogits - np.sum(probabilities * dlogits, axis=1, keepdims=True)
    )
    dresidual = WEIGHTS[:, None] / WEIGHT_SUM * dprobabilities
    hvp = np.empty_like(parameters)
    hvp[:COEFFICIENT_OFFSET] = (X.T @ dresidual).reshape(-1, order="F")
    hvp[COEFFICIENT_OFFSET:COEFFICIENT_OFFSET + N_CLASSES] = dresidual.sum(axis=0)
    hvp[:COEFFICIENT_OFFSET] += (
        parameters[-1] * direction[:COEFFICIENT_OFFSET]
        + direction[-1] * parameters[:COEFFICIENT_OFFSET]
    )
    hvp[-1] = np.dot(parameters[:COEFFICIENT_OFFSET], direction[:COEFFICIENT_OFFSET])
    return value, gradient, hvp


def parse_real(token: str) -> float:
    return float(token.strip().replace("D", "E").replace("d", "e"))


def parse_app(output: str) -> dict[str, list[str]]:
    records: dict[str, list[str]] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and fields[0].startswith("softmax_"):
            records[fields[0]] = fields
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/softmax_training.csv"),
    )
    parser.add_argument("--target", default="fortml_bench_softmax_training")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, check=True)
    started = time.perf_counter()
    completed = subprocess.run(
        ["fo", "exec", "--no-build", args.target], cwd=fortml,
        check=True, capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - started
    records = parse_app(completed.stdout)
    required = {"softmax_objective", "softmax_hvp", "softmax_fit", "softmax_cuda"}
    missing = required.difference(records)
    if missing:
        raise RuntimeError(f"release app omitted softmax fields: {sorted(missing)}")

    expected_value, expected_gradient, expected_hvp = oracle(PARAMETERS, DIRECTION)
    objective = records["softmax_objective"]
    if len(objective) != 13:
        raise RuntimeError(f"malformed softmax_objective row: {objective}")
    actual_value = parse_real(objective[1])
    actual_gradient_norm = parse_real(objective[2])
    actual_gradient = np.array([parse_real(token) for token in objective[3:]])
    objective_error = max(
        abs(actual_value - expected_value),
        abs(actual_gradient_norm - np.linalg.norm(expected_gradient)),
        float(np.max(np.abs(actual_gradient - expected_gradient))),
    )
    if objective_error > 3.0e-13:
        raise RuntimeError(f"softmax value/gradient oracle mismatch: {objective_error:.3e}")

    hvp = records["softmax_hvp"]
    if len(hvp) != 11:
        raise RuntimeError(f"malformed softmax_hvp row: {hvp}")
    actual_hvp = np.array([parse_real(token) for token in hvp[1:]])
    hvp_error = float(np.max(np.abs(actual_hvp - expected_hvp)))
    if hvp_error > 3.0e-13:
        raise RuntimeError(f"softmax HVP oracle mismatch: {hvp_error:.3e}")

    fit = records["softmax_fit"]
    if len(fit) != 5 or fit[1].upper() != "T":
        raise RuntimeError(f"softmax L-BFGS-B fit did not converge: {fit}")
    fit_objective = parse_real(fit[3])
    fit_gradient_norm = parse_real(fit[4])
    if not math.isfinite(fit_objective) or not math.isfinite(fit_gradient_norm):
        raise RuntimeError(f"nonfinite softmax fit record: {fit}")

    cuda = records["softmax_cuda"]
    if len(cuda) != 3 or cuda[1] != "unavailable" or int(cuda[2]) != 3:
        raise RuntimeError(f"softmax CUDA refusal changed unexpectedly: {cuda}")

    details = {
        "n_samples": str(X.shape[0]), "n_features": str(N_FEATURES),
        "n_classes": str(N_CLASSES), "oracle": "independent NumPy weighted softmax value/gradient/HVP",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (args.output.resolve(),)),
        "compiler": "gfortran", "flags": "-O3",
    }

    def row(**values: object) -> dict[str, object]:
        result: dict[str, object] = {field: "" for field in FIELDS}
        result.update(details)
        result.update({"workload": "softmax_training", "backend": "fortml",
                       "device": "cpu", "status": "pass"})
        result.update(values)
        return result

    rows = [
        row(phase="oracle", backend="numpy_oracle", seconds_per_operation="0.0",
            metric="objective", value=expected_value, max_abs_error="0.0",
            notes="weighted samples and sorted class weights; packed [coefficients, intercept, log2]"),
        row(phase="value_gradient", seconds_per_operation="0.0",
            metric="objective", value=actual_value, max_abs_error=objective_error,
            notes="release app value, gradient norm, and all ten gradient components"),
        row(phase="hvp", seconds_per_operation="0.0",
            metric="hvp_norm", value=float(np.linalg.norm(actual_hvp)),
            max_abs_error=hvp_error, notes="release app all ten exact HVP components"),
        row(phase="fit", seconds_per_operation=elapsed, metric="objective",
            value=fit_objective, max_abs_error="0.0",
            notes=f"bounded FortOpt L-BFGS-B; iterations={fit[2]}; grad_norm={fit_gradient_norm:.3e}"),
        row(phase="device_capability", device="cuda", status="unavailable",
            seconds_per_operation="", metric="", value="nan", max_abs_error="nan",
            oracle="typed_device_contract",
            notes="FORTNUM_NOT_IMPLEMENTED (code 3); no resident softmax-training CUDA path"),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
